"""USB 摄像头定时抓拍：历史归档 + latest 快照（供蓝牙传输）。"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

CameraHandler = Callable[[dict[str, Any]], None]

# Windows 上不同摄像头对后端敏感：
#   Integrated Camera → DirectShow 正常，MSMF 常黑屏
#   Logitech C920     → MSMF 正常，DirectShow 常纯色黑帧
_BACKEND_ORDER = ("msmf", "dshow")


def list_camera_names_windows() -> list[str]:
    """按系统枚举顺序返回摄像头友好名（常与 OpenCV index 一致，但不保证）。"""
    ps = (
        "Get-PnpDevice -Class Camera -Status OK "
        "| Select-Object -ExpandProperty FriendlyName"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("枚举摄像头失败: %s", exc)
        return []
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def backend_flag(name: str) -> int:
    import cv2

    mapping = {
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "any": cv2.CAP_ANY,
    }
    return mapping.get((name or "msmf").lower(), cv2.CAP_MSMF)


def _open_camera(
    index: int,
    backend: str,
    width: int,
    height: int,
    *,
    warmup: int = 6,
):
    """打开摄像头并预热；失败返回 None。"""
    import cv2

    t0 = time.time()
    cap = cv2.VideoCapture(index, backend_flag(backend))
    logger.info(
        "[camera] VideoCapture(index=%d, %s) opened=%s (%.1fs)",
        index,
        backend,
        cap.isOpened(),
        time.time() - t0,
    )
    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        return None
    try:
        # MSMF：必须在首帧之前设定分辨率；中途改分辨率会 Failed to select stream
        if width > 0 and height > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        best = 0.0
        for _ in range(max(warmup, 3)):
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            score = float(frame.std())
            if score > best:
                best = score
            if best >= 1.0:
                break
        if best < 1.0:
            logger.warning(
                "[camera] index=%d backend=%s 预热后仍为纯色/黑帧 (score=%.2f)",
                index,
                backend,
                best,
            )
            cap.release()
            time.sleep(0.3)
            return None
        return cap
    except Exception as exc:
        logger.warning("[camera] 打开异常 index=%d backend=%s: %s", index, backend, exc)
        try:
            cap.release()
        except Exception:
            pass
        time.sleep(0.3)
        return None


def resolve_camera(
    index: int | None,
    prefer_names: list[str] | None = None,
    *,
    backend: str | None = None,
    width: int = 1280,
    height: int = 720,
) -> tuple[int, str, str]:
    """解析 (index, display_name, backend)。保留供外部调试调用。"""
    names = list_camera_names_windows()
    prefer = prefer_names or ["C920", "Logitech", "HD Pro Webcam", "Logi"]
    backends = [backend] if backend else list(_BACKEND_ORDER)

    def label_for(i: int) -> str:
        return names[i] if 0 <= i < len(names) else f"camera_{i}"

    if index is not None:
        i = int(index)
        for b in backends:
            cap = _open_camera(i, b, width, height)
            if cap is not None:
                cap.release()
                time.sleep(0.4)
                return i, label_for(i), b
        return i, label_for(i), backends[0]

    prefer_idx = [
        i
        for i, n in enumerate(names)
        if any(p.lower() in n.lower() for p in prefer)
    ]
    ordered = prefer_idx + [i for i in range(max(len(names), 2)) if i not in prefer_idx]
    for i in ordered:
        for b in backends:
            cap = _open_camera(i, b, width, height)
            if cap is not None:
                cap.release()
                time.sleep(0.4)
                return i, label_for(i), b
    return 0, (names[0] if names else "camera_0"), backends[0]


class CameraCapture:
    """后台线程按间隔抓拍，写入历史目录并原子覆盖 latest 图片。"""

    def __init__(
        self,
        *,
        history_dir: str | Path,
        latest_path: str | Path,
        interval: float = 30.0,
        index: int | None = None,
        prefer_names: list[str] | None = None,
        width: int = 1280,
        height: int = 720,
        jpeg_quality: int = 85,
        backend: str | None = None,
        on_capture: CameraHandler | None = None,
        enabled: bool = True,
    ) -> None:
        self.history_dir = Path(history_dir)
        self.latest_path = Path(latest_path)
        self.interval = max(0.5, float(interval))
        self.index_cfg = index
        self.prefer_names = prefer_names
        self.width = int(width)
        self.height = int(height)
        self.jpeg_quality = max(1, min(100, int(jpeg_quality)))
        # None / "auto" → 自动在 msmf/dshow 间选择
        backend_raw = (backend or "auto").lower()
        self.backend_cfg: str | None = None if backend_raw in ("auto", "") else backend_raw
        self.on_capture = on_capture
        self.enabled = enabled
        self._latest_rel = "data/latest.jpg"
        self._history_rel = "data/camera/history"

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap = None
        self.camera_index = 0
        self.camera_name = "camera_0"
        self.backend = "msmf"
        self._last_meta: dict[str, Any] | None = None

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        project_root: Path,
        on_capture: CameraHandler | None = None,
    ) -> CameraCapture | None:
        cfg = config.get("camera") or {}
        if not cfg.get("enabled", False):
            return None
        history_rel = cfg.get("history_dir", "data/camera/history")
        latest_rel = cfg.get("latest_path", "data/latest.jpg")
        history = project_root / history_rel
        latest = project_root / latest_rel
        index = cfg.get("index", None)
        if index is not None:
            index = int(index)
        cap = cls(
            history_dir=history,
            latest_path=latest,
            interval=float(cfg.get("interval", 5.0)),
            index=index,
            prefer_names=list(cfg.get("prefer_names") or []),
            width=int(cfg.get("width", 1280)),
            height=int(cfg.get("height", 720)),
            jpeg_quality=int(cfg.get("jpeg_quality", 85)),
            backend=str(cfg.get("backend", "auto")),
            on_capture=on_capture,
            enabled=True,
        )
        cap._latest_rel = str(latest_rel).replace("\\", "/")
        cap._history_rel = str(history_rel).replace("\\", "/")
        return cap

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.latest_path.parent.mkdir(parents=True, exist_ok=True)
        # 选路/打开放到抓拍线程，避免主线程阻塞，并避免探测释放后再开导致占用冲突
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="usb-camera",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "[camera] 抓拍线程已启动 (interval=%.1fs) → latest=%s history=%s",
            self.interval,
            self.latest_path,
            self.history_dir,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(8.0, self.interval + 5.0))
        self._release_cap()
        logger.info("[camera] 已停止")

    def _ensure_open(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            return
        self._release_cap()

        names: list[str] = []
        if self.index_cfg is None or self.prefer_names:
            names = list_camera_names_windows()
        prefer = self.prefer_names or [
            "C920",
            "Logitech",
            "HD Pro Webcam",
            "Logi",
        ]
        backends = (
            [self.backend_cfg]
            if self.backend_cfg
            else list(_BACKEND_ORDER)
        )

        def label_for(i: int) -> str:
            return names[i] if 0 <= i < len(names) else f"camera_{i}"

        def is_prefer(i: int) -> bool:
            return any(p.lower() in label_for(i).lower() for p in prefer)

        if self.index_cfg is not None:
            ordered = [int(self.index_cfg)]
        else:
            prefer_idx = [i for i in range(len(names)) if is_prefer(i)]
            others = [
                i
                for i in range(max(len(names), 2))
                if i not in prefer_idx
            ]
            ordered = prefer_idx + others

        # 只打开一次：探测成功的句柄直接留用，避免 release 后再开冲突/卡住
        last_err = "无可用摄像头"
        for i in ordered:
            for b in backends:
                logger.info(
                    "[camera] 尝试打开 index=%d backend=%s (%s)",
                    i,
                    b,
                    label_for(i),
                )
                cap = _open_camera(
                    i, b, self.width, self.height, warmup=8
                )
                if cap is None:
                    last_err = f"index={i} backend={b} 无效画面或打不开"
                    continue
                self._cap = cap
                self.camera_index = i
                self.camera_name = label_for(i)
                self.backend = b
                logger.info(
                    "[camera] 已打开 name=%s index=%d backend=%s",
                    self.camera_name,
                    self.camera_index,
                    self.backend,
                )
                return

        raise RuntimeError(f"无法打开摄像头: {last_err}")

    def _release_cap(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
            time.sleep(0.3)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                meta = self._capture_once()
                self._last_meta = meta
                if self.on_capture and meta:
                    self.on_capture(meta)
                logger.info(
                    "[camera] 已抓拍 %s (%dx%d)",
                    meta.get("history_file") or meta.get("latest_path"),
                    meta.get("width", 0),
                    meta.get("height", 0),
                )
            except Exception as exc:
                logger.warning("[camera] 抓拍失败: %s", exc)
                self._release_cap()
            self._stop.wait(self.interval)
        self._release_cap()

    def _capture_once(self) -> dict[str, Any]:
        import cv2

        self._ensure_open()

        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._release_cap()
            self._ensure_open()
            ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("读帧失败")

        for _ in range(12):
            if frame is not None and float(frame.std()) >= 1.0:
                break
            ok, frame = self._cap.read()
            if not ok or frame is None:
                raise RuntimeError("读帧失败（画面全黑）")
        else:
            if frame is None or float(frame.std()) < 1.0:
                raise RuntimeError(
                    f"摄像头 index={self.camera_index} backend={self.backend} "
                    "持续输出纯色/黑帧，请检查镜头遮挡或改 config.yaml camera"
                )

        now = datetime.now(timezone.utc).astimezone()
        stamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        history_file = self.history_dir / f"{stamp}.jpg"
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]

        ok, buf = cv2.imencode(".jpg", frame, params)
        if not ok:
            raise RuntimeError("JPEG 编码失败")
        data = buf.tobytes()

        # 不用 cv2.imwrite：Windows 下中文路径常失败
        history_file.write_bytes(data)

        tmp = self.latest_path.with_suffix(self.latest_path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(self.latest_path)

        h, w = frame.shape[:2]
        return {
            "device": self.camera_name,
            "device_id": "usb_camera",
            "index": self.camera_index,
            "backend": self.backend,
            "timestamp": now.isoformat(timespec="milliseconds"),
            "width": int(w),
            "height": int(h),
            "latest_path": self._latest_rel,
            "history_file": history_file.name,
            "history_dir": self._history_rel,
        }
