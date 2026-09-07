"""全传感器最新读数快照：内存聚合，按间隔原子写入单文件供移动端拉取。"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .parser import ParsedFrame

logger = logging.getLogger(__name__)


class LatestSnapshot:
    """维护各传感器最新值，定时覆盖写入一个 JSON 文件。"""

    def __init__(
        self,
        path: str | Path,
        interval: float = 1.0,
        enabled: bool = True,
    ) -> None:
        self.path = Path(path)
        self.interval = max(0.1, float(interval))
        self.enabled = enabled
        self._lock = threading.Lock()
        self._sensors: dict[str, dict[str, Any]] = {}
        self._camera: dict[str, Any] | None = None
        self._last_write = 0.0
        self._dirty = False

        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def update(self, frame: ParsedFrame) -> None:
        if not self.enabled:
            return
        ts = frame.timestamp.isoformat(timespec="milliseconds")
        with self._lock:
            for r in frame.readings:
                entry: dict[str, Any] = {
                    "value": r.value,
                    "unit": r.unit,
                    "device_id": frame.device_id,
                    "timestamp": ts,
                }
                if r.voltage is not None:
                    entry["voltage"] = r.voltage
                if r.extras:
                    entry["extras"] = r.extras
                self._sensors[r.sensor] = entry
            self._dirty = True

    def update_camera(self, meta: dict[str, Any]) -> None:
        """摄像头抓拍后写入影像元数据（latest.jpg 路径等）。"""
        if not self.enabled:
            return
        with self._lock:
            self._camera = {
                "device": meta.get("device"),
                "device_id": meta.get("device_id", "usb_camera"),
                "timestamp": meta.get("timestamp"),
                "width": meta.get("width"),
                "height": meta.get("height"),
                "path": meta.get("latest_path"),
                "history_file": meta.get("history_file"),
            }
            self._dirty = True

    def maybe_flush(self) -> None:
        """主循环调用：间隔到达且有更新时写盘。"""
        if not self.enabled:
            return
        now = time.monotonic()
        with self._lock:
            if not self._dirty:
                return
            if now - self._last_write < self.interval:
                return
            payload = self._build_unlocked()
            self._dirty = False
            self._last_write = now
        self._atomic_write(payload)

    def flush(self) -> None:
        """强制写一次（退出时调用）。"""
        if not self.enabled:
            return
        with self._lock:
            if not self._sensors and not self._camera:
                return
            payload = self._build_unlocked()
            self._dirty = False
            self._last_write = time.monotonic()
        self._atomic_write(payload)

    def _build_unlocked(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")
        # 固定传感器顺序，便于移动端对比
        order = (
            "H2S",
            "NH3",
            "VOC",
            "CH3SH",
            "C2H5OH",
            "C2H4",
            "DHT1_T",
            "DHT1_H",
            "DHT2_T",
            "DHT2_H",
        )
        sensors: dict[str, Any] = {}
        for key in order:
            if key in self._sensors:
                sensors[key] = self._sensors[key]
        for key, val in self._sensors.items():
            if key not in sensors:
                sensors[key] = val
        payload: dict[str, Any] = {
            "updated_at": now,
            "sensor_count": len(sensors),
            "sensors": sensors,
        }
        if self._camera:
            payload["camera"] = self._camera
        return payload

    def _atomic_write(self, payload: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            text = json.dumps(payload, ensure_ascii=False, indent=2)
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            logger.warning("写入快照失败 %s: %s", self.path, exc)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
