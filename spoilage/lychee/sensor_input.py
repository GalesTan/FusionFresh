"""解析大系统传感器输入 JSON（气体 / 环境），暂不参与评分。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_sensor_payload(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """
    加载 example.json 风格的传感器输入。

    支持：
    - 文件路径
    - 已解析的 dict
    """
    if isinstance(source, dict):
        payload = source
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"传感器 JSON 不存在: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("传感器输入必须是 JSON 对象")
    return payload


def extract_sensors_for_output(payload: dict[str, Any]) -> dict[str, Any]:
    """
    从输入中提取供后续融合的传感器字段。

    注意：不读取 / 不使用 camera 字段作为图像路径。
    """
    sensors = payload.get("sensors")
    if sensors is not None and not isinstance(sensors, dict):
        raise ValueError("sensors 字段必须是对象")

    return {
        "updated_at": payload.get("updated_at"),
        "sensor_count": payload.get("sensor_count"),
        "sensors": sensors if sensors is not None else {},
        # camera 仅透传元数据，明确不作为图像输入
        "camera_ignored": True,
        "camera_meta": {
            "device": (payload.get("camera") or {}).get("device"),
            "device_id": (payload.get("camera") or {}).get("device_id"),
            "timestamp": (payload.get("camera") or {}).get("timestamp"),
            "width": (payload.get("camera") or {}).get("width"),
            "height": (payload.get("camera") or {}).get("height"),
            # path / history_file 故意不作为推理输入
        }
        if isinstance(payload.get("camera"), dict)
        else None,
        "gas_used_in_score": False,
    }
