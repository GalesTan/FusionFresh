"""传感器行协议解析：按正则从一行中提取一个或多个传感器读数。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class SensorReading:
    sensor: str
    value: float
    unit: str
    voltage: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedFrame:
    device_id: str
    timestamp: datetime
    readings: list[SensorReading]
    raw: str


# 统一协议默认 pattern（也可由 config.yaml 覆盖）:
#   SENSOR:value unit[|KEY:VAL]*
# 模拟量附 |V:电压；Modbus 附 |STATUS|LOW|HIGH|FS
# 示例:
#   H2S:12.34 ppm|V:0.246 NH3:56.78 ppm|V:0.567 VOC:3.45 ppm|V:0.014
#   C2H5OH:1.23 %LEL|STATUS:1 C2H4:0.450 ppm|V:0.123 DHT1_T:25.3 C
DEFAULT_UNIFIED_PATTERN = (
    r"(?P<sensor>[A-Za-z0-9_]+):(?P<value>-?\d+(?:\.\d+)?)\s+"
    r"(?P<unit>[^\s|]+)(?:\|(?P<attrs>[^\s]+))?"
)


class LineParser:
    """根据 config 中的 pattern 解析一行蓝牙文本。

    行首以 # 开头的视为注释/状态行，忽略。
    """

    def __init__(self, name: str, pattern: str, multi_sensor: bool = True) -> None:
        self.name = name
        self.multi_sensor = multi_sensor
        self._re = re.compile(pattern)

    def parse(self, device_id: str, line: str) -> ParsedFrame | None:
        text = line.strip()
        if not text or text.startswith("#"):
            return None

        matches = list(self._re.finditer(text))
        if not matches:
            return None
        if not self.multi_sensor:
            matches = matches[:1]

        readings: list[SensorReading] = []
        for m in matches:
            gd = m.groupdict()
            sensor = gd.get("sensor")
            value = gd.get("value")
            if sensor is None or value is None:
                continue
            unit = gd.get("unit") or ""
            extras = self._parse_attrs(gd.get("attrs"))
            voltage_raw = gd.get("voltage")
            if voltage_raw is None and "V" in extras:
                voltage_raw = extras.pop("V")
            voltage = None
            if voltage_raw is not None:
                try:
                    voltage = float(voltage_raw)
                except (TypeError, ValueError):
                    extras["V"] = voltage_raw

            # 命名组里除标准字段外的其它捕获也并入 extras
            for k, v in gd.items():
                if k in ("sensor", "value", "unit", "attrs", "voltage") or v is None:
                    continue
                extras[k] = v

            readings.append(
                SensorReading(
                    sensor=sensor.upper(),
                    value=float(value),
                    unit=unit,
                    voltage=voltage,
                    extras=extras,
                )
            )

        if not readings:
            return None

        return ParsedFrame(
            device_id=device_id,
            timestamp=datetime.now(timezone.utc).astimezone(),
            readings=readings,
            raw=text,
        )

    @staticmethod
    def _parse_attrs(attrs: str | None) -> dict[str, Any]:
        """解析 |KEY:VAL|KEY:VAL 段（传入时不含前导 |）。"""
        out: dict[str, Any] = {}
        if not attrs:
            return out
        for piece in attrs.split("|"):
            if not piece or ":" not in piece:
                continue
            key, val = piece.split(":", 1)
            key = key.strip()
            val = val.strip()
            if not key:
                continue
            try:
                if "." in val:
                    out[key] = float(val)
                else:
                    out[key] = int(val)
            except ValueError:
                out[key] = val
        return out


def build_parsers(parsers_cfg: dict[str, Any]) -> dict[str, LineParser]:
    out: dict[str, LineParser] = {}
    for name, cfg in parsers_cfg.items():
        out[name] = LineParser(
            name=name,
            pattern=cfg.get("pattern", DEFAULT_UNIFIED_PATTERN),
            multi_sensor=bool(cfg.get("multi_sensor", True)),
        )
    return out
