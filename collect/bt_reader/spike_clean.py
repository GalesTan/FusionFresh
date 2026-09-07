"""传感器尖峰 / 骤降检测与前后邻域近似替代。

判定：相对前后点同时大幅偏离，且前后点彼此接近（典型单点毛刺）。
替代：value / voltage 用前后邻域线性插值（两点中点）。
"""

from __future__ import annotations

import csv
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .parser import ParsedFrame, SensorReading

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpikeCleanConfig:
    enabled: bool = True
    interval: float = 300.0  # 秒，主循环定期巡检间隔
    min_abs_delta: float = 10.0  # 最小绝对跳变（ppm / % 等）
    neighbor_factor: float = 0.4  # 前后跨度须 ≤ factor * min(两侧跳变)
    max_passes: int = 3  # 迭代清理轮数（处理交替毛刺）
    online: bool = True  # 写入前一拍延时校正


def spike_clean_config_from_dict(cfg: dict[str, Any] | None) -> SpikeCleanConfig:
    cfg = cfg or {}
    return SpikeCleanConfig(
        enabled=bool(cfg.get("enabled", True)),
        interval=float(cfg.get("interval", 300.0)),
        min_abs_delta=float(cfg.get("min_abs_delta", 10.0)),
        neighbor_factor=float(cfg.get("neighbor_factor", 0.4)),
        max_passes=max(1, int(cfg.get("max_passes", 3))),
        online=bool(cfg.get("online", True)),
    )


def is_spike(
    prev: float,
    cur: float,
    nxt: float,
    *,
    min_abs_delta: float = 10.0,
    neighbor_factor: float = 0.4,
) -> bool:
    """单点尖峰或骤降：两侧跳变大，前后点彼此接近。"""
    jump_in = abs(cur - prev)
    jump_out = abs(cur - nxt)
    span = abs(nxt - prev)
    if jump_in < min_abs_delta or jump_out < min_abs_delta:
        return False
    return span <= neighbor_factor * min(jump_in, jump_out)


def _mid(a: float, b: float) -> float:
    return (a + b) / 2.0


def clean_values(
    values: Sequence[float],
    *,
    min_abs_delta: float = 10.0,
    neighbor_factor: float = 0.4,
    max_passes: int = 3,
) -> tuple[list[float], list[int]]:
    """清洗数值序列，返回 (新序列, 被修改的下标列表)。"""
    out = [float(v) for v in values]
    changed: set[int] = set()
    n = len(out)
    if n < 3:
        return out, []

    for _ in range(max_passes):
        pass_hits: list[int] = []
        # 基于本轮开始时的快照判定，避免同轮相互干扰
        snapshot = list(out)
        for i in range(1, n - 1):
            if is_spike(
                snapshot[i - 1],
                snapshot[i],
                snapshot[i + 1],
                min_abs_delta=min_abs_delta,
                neighbor_factor=neighbor_factor,
            ):
                pass_hits.append(i)
        if not pass_hits:
            break
        for i in pass_hits:
            out[i] = _mid(snapshot[i - 1], snapshot[i + 1])
            changed.add(i)

    return out, sorted(changed)


def correct_middle_reading(
    prev: SensorReading,
    cur: SensorReading,
    nxt: SensorReading,
    *,
    min_abs_delta: float = 10.0,
    neighbor_factor: float = 0.4,
) -> SensorReading:
    """若 cur 相对 prev/nxt 为尖峰，返回前后中点替代后的 reading。"""
    if prev.sensor != cur.sensor or cur.sensor != nxt.sensor:
        return cur
    if not is_spike(
        prev.value,
        cur.value,
        nxt.value,
        min_abs_delta=min_abs_delta,
        neighbor_factor=neighbor_factor,
    ):
        return cur

    new_value = _mid(prev.value, nxt.value)
    new_voltage = cur.voltage
    if prev.voltage is not None and nxt.voltage is not None:
        new_voltage = _mid(prev.voltage, nxt.voltage)

    return SensorReading(
        sensor=cur.sensor,
        value=new_value,
        unit=cur.unit,
        voltage=new_voltage,
        extras=dict(cur.extras) if cur.extras else {},
    )


def correct_middle_frame(
    prev: ParsedFrame,
    cur: ParsedFrame,
    nxt: ParsedFrame,
    *,
    min_abs_delta: float = 10.0,
    neighbor_factor: float = 0.4,
) -> ParsedFrame:
    """按传感器名对齐，校正中间帧中的尖峰读数。"""
    prev_map = {r.sensor: r for r in prev.readings}
    nxt_map = {r.sensor: r for r in nxt.readings}
    fixed: list[SensorReading] = []
    for r in cur.readings:
        p = prev_map.get(r.sensor)
        n = nxt_map.get(r.sensor)
        if p is None or n is None:
            fixed.append(r)
            continue
        fixed.append(
            correct_middle_reading(
                p,
                r,
                n,
                min_abs_delta=min_abs_delta,
                neighbor_factor=neighbor_factor,
            )
        )
    return ParsedFrame(
        device_id=cur.device_id,
        timestamp=cur.timestamp,
        readings=fixed,
        raw=cur.raw,
    )


def clean_csv_file(
    path: Path,
    *,
    min_abs_delta: float = 10.0,
    neighbor_factor: float = 0.4,
    max_passes: int = 3,
) -> int:
    """原地清洗单个传感器 CSV，返回修改行数。文件须未被占用。"""
    if not path.exists() or path.stat().st_size == 0:
        return 0

    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "value" not in fieldnames or len(rows) < 3:
        return 0

    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row["value"]))
        except (TypeError, ValueError):
            values.append(float("nan"))

    # 含 NaN 时跳过整文件，避免破坏数据
    if any(v != v for v in values):  # NaN check
        logger.warning("跳过含非法 value 的 CSV: %s", path)
        return 0

    cleaned, changed = clean_values(
        values,
        min_abs_delta=min_abs_delta,
        neighbor_factor=neighbor_factor,
        max_passes=max_passes,
    )
    if not changed:
        return 0

    has_voltage = "voltage" in fieldnames
    for i in changed:
        rows[i]["value"] = cleaned[i]
        if not has_voltage:
            continue
        # 电压随 value 一并做前后中点（仅当两侧电压可解析）
        try:
            vp = float(rows[i - 1]["voltage"]) if rows[i - 1].get("voltage") not in ("", None) else None
            vn = float(rows[i + 1]["voltage"]) if rows[i + 1].get("voltage") not in ("", None) else None
        except (TypeError, ValueError):
            vp = vn = None
        if vp is not None and vn is not None:
            rows[i]["voltage"] = _mid(vp, vn)

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            delete=False,
            dir=str(path.parent),
            suffix=".csv.tmp",
        ) as tmp:
            tmp_path = Path(tmp.name)
            writer = csv.DictWriter(tmp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        shutil.move(str(tmp_path), str(path))
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return len(changed)


def clean_data_dirs(
    data_dirs: Sequence[Path],
    *,
    min_abs_delta: float = 10.0,
    neighbor_factor: float = 0.4,
    max_passes: int = 3,
) -> dict[str, int]:
    """清洗若干 data_dir 下全部 CSV，返回 {相对路径: 修改行数}。"""
    results: dict[str, int] = {}
    for data_dir in data_dirs:
        if not data_dir.is_dir():
            continue
        for csv_path in sorted(data_dir.glob("*.csv")):
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
                logger.info("已修正尖峰/骤降 %d 处: %s", n, csv_path)
    return results


class OnlineSpikeFilter:
    """按设备缓存最近两帧，新帧到达时校正中间帧后放出。"""

    def __init__(
        self,
        *,
        min_abs_delta: float = 10.0,
        neighbor_factor: float = 0.4,
    ) -> None:
        self.min_abs_delta = min_abs_delta
        self.neighbor_factor = neighbor_factor
        self._prev: ParsedFrame | None = None
        self._pending: ParsedFrame | None = None

    def push(self, frame: ParsedFrame) -> ParsedFrame | None:
        """喂入新帧；若有已校正的中间帧则返回，否则返回 None。"""
        if self._pending is None:
            self._pending = frame
            return None

        if self._prev is not None:
            out = correct_middle_frame(
                self._prev,
                self._pending,
                frame,
                min_abs_delta=self.min_abs_delta,
                neighbor_factor=self.neighbor_factor,
            )
        else:
            out = self._pending

        self._prev = out
        self._pending = frame
        return out

    def flush(self) -> list[ParsedFrame]:
        """结束时放出尚未写出的缓存帧（末帧无法再做三点判定）。"""
        leftover: list[ParsedFrame] = []
        if self._pending is not None:
            leftover.append(self._pending)
            self._pending = None
        self._prev = None
        return leftover
