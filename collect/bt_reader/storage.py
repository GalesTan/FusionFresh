"""本地文件实时落盘：CSV / JSONL，支持按传感器分文件。"""

from __future__ import annotations

import csv
import json
import logging
import threading
import time
from pathlib import Path
from typing import TextIO

from .parser import ParsedFrame, SensorReading

logger = logging.getLogger(__name__)


class DataStore:
    """每个设备一个 DataStore，线程安全写入。"""

    CSV_FIELDS = (
        "timestamp",
        "device_id",
        "sensor",
        "value",
        "unit",
        "voltage",
        "raw",
    )

    def __init__(
        self,
        data_dir: str | Path,
        device_id: str,
        fmt: str = "both",
        split_by_sensor: bool = True,
        flush_interval: float = 1.0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.device_id = device_id
        self.fmt = fmt.lower()
        self.split_by_sensor = split_by_sensor
        self.flush_interval = flush_interval

        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._csv_files: dict[str, TextIO] = {}
        self._csv_writers: dict[str, csv.DictWriter] = {}
        self._jsonl_files: dict[str, TextIO] = {}
        self._last_flush = time.monotonic()
        self._closed = False

    def write(self, frame: ParsedFrame) -> None:
        with self._lock:
            if self._closed:
                return
            for reading in frame.readings:
                if self.fmt in ("csv", "both"):
                    self._write_csv(frame, reading)
                if self.fmt in ("jsonl", "both"):
                    self._write_jsonl(frame, reading)
            self._maybe_flush()

    def _key(self, sensor: str) -> str:
        return sensor.upper() if self.split_by_sensor else "all_sensors"

    def _write_csv(self, frame: ParsedFrame, reading: SensorReading) -> None:
        key = self._key(reading.sensor)
        if key not in self._csv_writers:
            path = self.data_dir / f"{key}.csv"
            new_file = not path.exists() or path.stat().st_size == 0
            fh = path.open("a", encoding="utf-8", newline="")
            writer = csv.DictWriter(fh, fieldnames=self.CSV_FIELDS)
            if new_file:
                writer.writeheader()
            self._csv_files[key] = fh
            self._csv_writers[key] = writer

        self._csv_writers[key].writerow(
            {
                "timestamp": frame.timestamp.isoformat(timespec="milliseconds"),
                "device_id": frame.device_id,
                "sensor": reading.sensor,
                "value": reading.value,
                "unit": reading.unit,
                "voltage": "" if reading.voltage is None else reading.voltage,
                "raw": frame.raw,
            }
        )

    def _write_jsonl(self, frame: ParsedFrame, reading: SensorReading) -> None:
        key = self._key(reading.sensor)
        if key not in self._jsonl_files:
            path = self.data_dir / f"{key}.jsonl"
            self._jsonl_files[key] = path.open("a", encoding="utf-8")

        record = {
            "timestamp": frame.timestamp.isoformat(timespec="milliseconds"),
            "device_id": frame.device_id,
            "sensor": reading.sensor,
            "value": reading.value,
            "unit": reading.unit,
            "voltage": reading.voltage,
            "raw": frame.raw,
        }
        if reading.extras:
            record["extras"] = reading.extras
        self._jsonl_files[key].write(json.dumps(record, ensure_ascii=False) + "\n")

    def _maybe_flush(self) -> None:
        now = time.monotonic()
        if now - self._last_flush < self.flush_interval:
            return
        self._flush_unlocked()
        self._last_flush = now

    def _flush_unlocked(self) -> None:
        for fh in self._csv_files.values():
            fh.flush()
        for fh in self._jsonl_files.values():
            fh.flush()

    def flush(self) -> None:
        with self._lock:
            self._flush_unlocked()
            self._last_flush = time.monotonic()

    def suspend_writers(self) -> None:
        """关闭打开的文件句柄，便于外部原地重写 CSV；后续 write 会自动 reopen。"""
        with self._lock:
            if self._closed:
                return
            self._close_writers_unlocked()

    def clean_spike_csvs(
        self,
        *,
        min_abs_delta: float = 10.0,
        neighbor_factor: float = 0.4,
        max_passes: int = 3,
    ) -> dict[str, int]:
        """在写锁内关闭句柄并清洗本目录 CSV，避免与 write 并发冲突。"""
        from .spike_clean import clean_csv_file

        results: dict[str, int] = {}
        with self._lock:
            if self._closed:
                return results
            self._close_writers_unlocked()
            for csv_path in sorted(self.data_dir.glob("*.csv")):
                try:
                    n = clean_csv_file(
                        csv_path,
                        min_abs_delta=min_abs_delta,
                        neighbor_factor=neighbor_factor,
                        max_passes=max_passes,
                    )
                except OSError as exc:
                    logger.warning("清洗失败 %s: %s", csv_path, exc)
                    continue
                if n:
                    results[str(csv_path)] = n
        return results

    def _close_writers_unlocked(self) -> None:
        self._flush_unlocked()
        for fh in self._csv_files.values():
            fh.close()
        for fh in self._jsonl_files.values():
            fh.close()
        self._csv_files.clear()
        self._csv_writers.clear()
        self._jsonl_files.clear()

    def close(self) -> None:
        with self._lock:
            self._flush_unlocked()
            for fh in self._csv_files.values():
                fh.close()
            for fh in self._jsonl_files.values():
                fh.close()
            self._csv_files.clear()
            self._csv_writers.clear()
            self._jsonl_files.clear()
            self._closed = True
