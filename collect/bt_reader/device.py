"""蓝牙 SPP（Windows 虚拟 COM）设备连接与多设备并发读取。"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import serial
from serial import SerialException

from .parser import LineParser, ParsedFrame
from .spike_clean import OnlineSpikeFilter, SpikeCleanConfig
from .storage import DataStore

logger = logging.getLogger(__name__)

FrameHandler = Callable[[ParsedFrame], None]


class BluetoothDevice:
    """单台单片机：串口读线程 + 解析 + 落盘。"""

    def __init__(
        self,
        device_id: str,
        name: str,
        port: str,
        baudrate: int,
        parser: LineParser,
        store: DataStore,
        on_frame: FrameHandler | None = None,
        reconnect_delay: float = 3.0,
        sample_interval: float = 0.0,
        spike_filter: OnlineSpikeFilter | None = None,
    ) -> None:
        self.device_id = device_id
        self.name = name
        self.port = port
        self.baudrate = baudrate
        self.parser = parser
        self.store = store
        self.on_frame = on_frame
        self.reconnect_delay = reconnect_delay
        self.sample_interval = max(0.0, float(sample_interval))
        self.spike_filter = spike_filter

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ser: serial.Serial | None = None
        self._last_sample = 0.0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"bt-{self.device_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "[%s] 读线程已启动 (BT=%s %s @ %s, sample=%.0fs)",
            self.device_id,
            self.name,
            self.port,
            self.baudrate,
            self.sample_interval,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5)
        if self.spike_filter is not None:
            for frame in self.spike_filter.flush():
                self._emit_frame(frame)
        self.store.close()
        logger.info("[%s] 已停止", self.device_id)

    def _open(self) -> serial.Serial:
        return serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=1.0,
            write_timeout=1.0,
        )

    def _run(self) -> None:
        buf = bytearray()
        while not self._stop.is_set():
            try:
                if self._ser is None or not self._ser.is_open:
                    logger.info("[%s] 正在连接 %s ...", self.device_id, self.port)
                    self._ser = self._open()
                    buf.clear()
                    logger.info("[%s] 已连接", self.device_id)

                chunk = self._ser.read(256)
                if not chunk:
                    continue
                buf.extend(chunk)

                while True:
                    nl = buf.find(b"\n")
                    if nl < 0:
                        break
                    line_bytes = buf[: nl + 1]
                    del buf[: nl + 1]
                    line = line_bytes.decode("utf-8", errors="replace")
                    self._handle_line(line)

            except SerialException as exc:
                logger.warning("[%s] 串口异常: %s，%ss 后重连", self.device_id, exc, self.reconnect_delay)
                self._safe_close()
                self._stop.wait(self.reconnect_delay)
            except Exception as exc:
                logger.exception("[%s] 未预期错误: %s", self.device_id, exc)
                self._safe_close()
                self._stop.wait(self.reconnect_delay)

        self._safe_close()

    def _safe_close(self) -> None:
        if self._ser is not None:
            try:
                if self._ser.is_open:
                    self._ser.close()
            except Exception:
                pass
            self._ser = None

    def _handle_line(self, line: str) -> None:
        frame = self.parser.parse(self.device_id, line)
        if frame is None:
            stripped = line.strip()
            if not stripped:
                return
            # # 开头为板端状态/注释；其它无法解析的行升为 warning，避免“已连接却无输出”
            if stripped.startswith("#"):
                logger.info("[%s] %s", self.device_id, stripped)
            else:
                logger.warning("[%s] 无法解析(请确认已烧录统一协议固件): %s", self.device_id, stripped)
            return

        if self.sample_interval > 0:
            now = time.monotonic()
            if now - self._last_sample < self.sample_interval:
                return
            self._last_sample = now

        if self.spike_filter is not None:
            ready = self.spike_filter.push(frame)
            if ready is None:
                return
            frame = ready

        self._emit_frame(frame)

    def _emit_frame(self, frame: ParsedFrame) -> None:
        self.store.write(frame)
        if self.on_frame:
            try:
                self.on_frame(frame)
            except Exception:
                logger.exception("[%s] on_frame 回调异常", self.device_id)


class DeviceManager:
    """管理多台设备的启动 / 停止。"""

    def __init__(self) -> None:
        self._devices: dict[str, BluetoothDevice] = {}

    def add(self, device: BluetoothDevice) -> None:
        if device.device_id in self._devices:
            raise ValueError(f"设备 id 重复: {device.device_id}")
        self._devices[device.device_id] = device

    def start_all(self) -> None:
        for dev in self._devices.values():
            dev.start()

    def stop_all(self) -> None:
        for dev in self._devices.values():
            dev.stop()

    def suspend_stores(self) -> None:
        """关闭各设备落盘句柄，供 CSV 原地清洗后由后续 write 自动 reopen。"""
        for dev in self._devices.values():
            dev.store.suspend_writers()

    def clean_spike_csvs(
        self,
        *,
        min_abs_delta: float = 10.0,
        neighbor_factor: float = 0.4,
        max_passes: int = 3,
    ) -> dict[str, int]:
        results: dict[str, int] = {}
        for dev in self._devices.values():
            part = dev.store.clean_spike_csvs(
                min_abs_delta=min_abs_delta,
                neighbor_factor=neighbor_factor,
                max_passes=max_passes,
            )
            results.update(part)
            for path, n in part.items():
                logger.info("已修正尖峰/骤降 %d 处: %s", n, path)
        return results

    def data_dirs(self) -> list:
        from pathlib import Path

        return [Path(dev.store.data_dir) for dev in self._devices.values()]

    @property
    def devices(self) -> dict[str, BluetoothDevice]:
        return dict(self._devices)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        parsers: dict[str, LineParser],
        project_root: str | None = None,
        on_frame: FrameHandler | None = None,
        spike_cfg: SpikeCleanConfig | None = None,
    ) -> DeviceManager:
        from pathlib import Path

        root = Path(project_root) if project_root else Path.cwd()
        storage_cfg = config.get("storage", {})
        fmt = storage_cfg.get("format", "both")
        split = bool(storage_cfg.get("split_by_sensor", True))
        flush_interval = float(storage_cfg.get("flush_interval", 1.0))
        sample_interval = float(storage_cfg.get("sample_interval", 0.0))
        if spike_cfg is None:
            from .spike_clean import spike_clean_config_from_dict

            spike_cfg = spike_clean_config_from_dict(config.get("spike_clean"))

        manager = cls()
        for item in config.get("devices", []):
            if not item.get("enabled", True):
                continue
            parser_name = item["parser"]
            if parser_name not in parsers:
                raise KeyError(f"设备 {item['id']} 引用了未知解析器: {parser_name}")

            data_dir = root / item.get("data_dir", f"data/{item['id']}")
            store = DataStore(
                data_dir=data_dir,
                device_id=item["id"],
                fmt=fmt,
                split_by_sensor=split,
                flush_interval=flush_interval,
            )
            spike_filter = None
            if spike_cfg.enabled and spike_cfg.online:
                spike_filter = OnlineSpikeFilter(
                    min_abs_delta=spike_cfg.min_abs_delta,
                    neighbor_factor=spike_cfg.neighbor_factor,
                )
            device = BluetoothDevice(
                device_id=item["id"],
                name=item.get("name", item["id"]),
                port=item["port"],
                baudrate=int(item.get("baudrate", 115200)),
                parser=parsers[parser_name],
                store=store,
                on_frame=on_frame,
                sample_interval=float(item.get("sample_interval", sample_interval)),
                spike_filter=spike_filter,
            )
            manager.add(device)
        return manager
