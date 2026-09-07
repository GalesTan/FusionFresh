"""虾仁场景气体融合规则（仅当检测结果全是虾仁时生效）。

规则（窗口默认 180s）：
1. H2S 或 NH3 在窗口内持续检出（> present_ppm）且未消失 → 强制 spoiled
2. 否则 VOC 或 C2H5OH（乙醇）在窗口内持续 > mid_ppm → 强制 spoiling（中期）
3. 强制等级时，图像 score∈[0,1] 线性映射到对应虾仁阈值带：
   - spoiling → [early_mid, mid_late)
   - spoiled  → [mid_late, 1.0]
其余情况不改图像判定。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_DEFAULT_DATA_ROOT = _HERE.parent.parent / "collect" / "data"

# 传感器文件相对 data root 的位置
_SENSOR_FILES = {
    "VOC": ("esp32_gas", "VOC.csv"),
    "C2H5OH": ("esp32_env", "C2H5OH.csv"),
    "H2S": ("esp32_gas", "H2S.csv"),
    "NH3": ("esp32_gas", "NH3.csv"),
}

_GAS_DISPLAY = {
    "VOC": "VOC",
    "C2H5OH": "Ethanol",
    "H2S": "Hydrogen Sulfide",
    "NH3": "Ammonia",
}

DEFAULT_THRESHOLDS = {
    "early_mid": 0.3034142553806305,
    "mid_late": 0.6775257289409637,
}


@dataclass
class GasOverride:
    level: str  # spoiling | spoiled
    reason: str
    trigger_sensors: list[str]
    produced_gases: list[str]
    evidence: dict[str, Any]


def _parse_ts(text: str) -> datetime | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        # 采集端多为本地 +08；无时区时按本地墙钟理解
        dt = dt.astimezone()
    return dt


def _read_csv_tail_rows(path: Path, max_rows: int = 256) -> list[dict[str, str]]:
    """读 CSV 尾部若干行（避免大文件全量加载）。"""
    if not path.is_file():
        return []
    try:
        with path.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            block = 65536
            data = b""
            while size > 0 and data.count(b"\n") <= max_rows + 2:
                step = min(block, size)
                size -= step
                fh.seek(size)
                data = fh.read(step) + data
                if size == 0:
                    break
        text = data.decode("utf-8", errors="replace")
    except OSError:
        return []

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    # 保证有表头
    header_line = None
    with path.open("r", encoding="utf-8", newline="") as fh:
        header_line = fh.readline().strip()
    if not header_line:
        return []
    body = lines[-(max_rows):]
    if body and body[0].startswith("timestamp"):
        pass
    else:
        body = [header_line] + body
    reader = csv.DictReader(body)
    return list(reader)


def load_sensor_window(
    sensor: str,
    data_root: Path | None = None,
    *,
    window_sec: float = 180.0,
    now: datetime | None = None,
    max_rows: int = 256,
) -> list[tuple[datetime, float, str]]:
    """返回窗口内 (timestamp, value, unit) 升序列表。"""
    root = Path(data_root) if data_root else _DEFAULT_DATA_ROOT
    rel = _SENSOR_FILES.get(sensor.upper())
    if rel is None:
        return []
    path = root / rel[0] / rel[1]
    rows = _read_csv_tail_rows(path, max_rows=max_rows)
    if not rows:
        return []

    now_dt = now or datetime.now().astimezone()
    if now_dt.tzinfo is None:
        now_dt = now_dt.astimezone()
    start = now_dt - timedelta(seconds=window_sec)

    out: list[tuple[datetime, float, str]] = []
    for row in rows:
        ts = _parse_ts(row.get("timestamp", ""))
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.astimezone()
        else:
            ts = ts.astimezone(now_dt.tzinfo)
        if ts < start or ts > now_dt + timedelta(seconds=5):
            continue
        try:
            val = float(row.get("value"))
        except (TypeError, ValueError):
            continue
        unit = str(row.get("unit") or "").strip()
        out.append((ts, val, unit))
    out.sort(key=lambda x: x[0])
    return out


def _window_ok(
    series: list[tuple[datetime, float, str]],
    *,
    window_sec: float,
    min_points: int,
    min_span_sec: float,
    stale_sec: float,
    now: datetime,
) -> bool:
    if len(series) < min_points:
        return False
    latest = series[-1][0]
    earliest = series[0][0]
    if (now - latest).total_seconds() > stale_sec:
        return False
    span = (latest - earliest).total_seconds()
    if span < min_span_sec:
        return False
    # 最早点应接近窗口起点（允许采样间隔空隙）
    if (now - earliest).total_seconds() < min_span_sec:
        return False
    return True


def continuous_above(
    series: list[tuple[datetime, float, str]],
    threshold: float,
    *,
    window_sec: float = 180.0,
    min_points: int = 3,
    min_span_sec: float = 150.0,
    stale_sec: float = 90.0,
    now: datetime | None = None,
) -> bool:
    """窗口内全部采样点均 > threshold，且覆盖足够时长。"""
    now_dt = now or datetime.now().astimezone()
    if not _window_ok(
        series,
        window_sec=window_sec,
        min_points=min_points,
        min_span_sec=min_span_sec,
        stale_sec=stale_sec,
        now=now_dt,
    ):
        return False
    return all(v > threshold for _, v, _ in series)


def scale_score_to_level(
    image_score: float | None,
    level: str,
    thresholds: dict[str, float] | None = None,
) -> float:
    """将图像 0-1 评分映射到强制等级对应的阈值带。"""
    thr = thresholds or DEFAULT_THRESHOLDS
    early_mid = float(thr.get("early_mid", thr.get("fresh_uncertain", DEFAULT_THRESHOLDS["early_mid"])))
    mid_late = float(thr.get("mid_late", thr.get("uncertain_spoiled", DEFAULT_THRESHOLDS["mid_late"])))
    s = 0.0 if image_score is None else float(image_score)
    s = min(max(s, 0.0), 1.0)
    if level == "spoiled":
        lo, hi = mid_late, 1.0
    elif level == "spoiling":
        lo, hi = early_mid, mid_late
    else:
        lo, hi = 0.0, early_mid
    if hi <= lo:
        return round(lo, 4)
    return round(lo + s * (hi - lo), 4)


def evaluate_shrimp_gas_rules(
    data_root: Path | None = None,
    *,
    window_sec: float = 180.0,
    mid_ppm: float = 1.0,
    present_ppm: float = 0.0,
    now: datetime | None = None,
    min_points: int = 3,
    min_span_sec: float = 150.0,
    stale_sec: float = 90.0,
) -> GasOverride | None:
    """评估虾仁气体强制规则；不触发则返回 None。"""
    root = Path(data_root) if data_root else _DEFAULT_DATA_ROOT
    now_dt = now or datetime.now().astimezone()

    series = {
        name: load_sensor_window(
            name, root, window_sec=window_sec, now=now_dt,
        )
        for name in ("H2S", "NH3", "VOC", "C2H5OH")
    }

    kwargs = dict(
        window_sec=window_sec,
        min_points=min_points,
        min_span_sec=min_span_sec,
        stale_sec=stale_sec,
        now=now_dt,
    )

    # 规则 1：H2S / NH3 持续存在 → 腐败
    toxic_hits = []
    for name in ("H2S", "NH3"):
        if continuous_above(series[name], present_ppm, **kwargs):
            toxic_hits.append(name)
    if toxic_hits:
        return GasOverride(
            level="spoiled",
            reason=(
                f"仅虾仁场景：{ '/'.join(toxic_hits) } "
                f"已持续 ≥{window_sec:.0f}s 未消失，强制判定腐败"
            ),
            trigger_sensors=toxic_hits,
            produced_gases=[_GAS_DISPLAY[n] for n in toxic_hits],
            evidence={
                name: {
                    "n": len(series[name]),
                    "min": min(v for _, v, _ in series[name]),
                    "max": max(v for _, v, _ in series[name]),
                    "unit": series[name][-1][2] if series[name] else "",
                }
                for name in toxic_hits
            },
        )

    # 规则 2：VOC 或乙醇持续 > mid_ppm → 中期
    mid_hits = []
    for name in ("VOC", "C2H5OH"):
        if continuous_above(series[name], mid_ppm, **kwargs):
            mid_hits.append(name)
    if mid_hits:
        return GasOverride(
            level="spoiling",
            reason=(
                f"仅虾仁场景：{ '/'.join(mid_hits) } "
                f"持续 ≥{window_sec:.0f}s 均 >{mid_ppm:g}，强制判定中期"
            ),
            trigger_sensors=mid_hits,
            produced_gases=[_GAS_DISPLAY[n] for n in mid_hits],
            evidence={
                name: {
                    "n": len(series[name]),
                    "min": min(v for _, v, _ in series[name]),
                    "max": max(v for _, v, _ in series[name]),
                    "unit": series[name][-1][2] if series[name] else "",
                }
                for name in mid_hits
            },
        )

    return None


def apply_gas_override_to_food(
    food: dict[str, Any],
    override: GasOverride,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """就地改写单项食物的等级/分数/气体/文案。"""
    image_score = food.get("spoilageScore")
    new_score = scale_score_to_level(image_score, override.level, thresholds)
    food["spoilageLevel"] = override.level
    food["spoilageScore"] = new_score
    # 关联气体按食物种类固定，不只显示触发传感器
    food["producedGases"] = ["VOC", "Ethanol", "Hydrogen Sulfide", "Ammonia"]
    if override.level == "spoiled":
        food["message"] = "气体检测异常，虾仁已不宜食用。"
    else:
        food["message"] = "气体略有升高，虾仁处于中期，建议尽快食用。"
    food["gasOverride"] = {
        "level": override.level,
        "reason": override.reason,
        "trigger_sensors": override.trigger_sensors,
        "imageScore": image_score,
        "scaledScore": new_score,
        "evidence": override.evidence,
    }
    return food
