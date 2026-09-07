"""
食物腐败检测 Release 包 — 独立推理入口。

输入:
  - 单果图片路径（独立指定，不使用传感器 JSON 中的 camera）
  - 食物类别
  - 可选：传感器 JSON（example.json 格式，气体信息暂透传、不参与评分）

输出: JSON（0-1 腐败评分 + 三阶段评级 + 传感器透传字段）

用法:
    python detect.py --image path/to/fruit.jpg --category lychee
    python detect.py --image fruit.jpg --category 荔枝 --sensors example.json --output result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

RELEASE_DIR = Path(__file__).resolve().parent
if str(RELEASE_DIR) not in sys.path:
    sys.path.insert(0, str(RELEASE_DIR))

from sensor_input import extract_sensors_for_output, load_sensor_payload

SUPPORTED_CATEGORIES = {
    "lychee": "lychee",
    "荔枝": "lychee",
    "litchi": "lychee",
}

UNSUPPORTED_CATEGORIES = {
    "shrimp": "虾仁检测尚未接入 release 包",
    "虾仁": "虾仁检测尚未接入 release 包",
}


def normalize_category(category: str) -> str:
    key = category.strip().lower()
    if key in UNSUPPORTED_CATEGORIES:
        raise ValueError(UNSUPPORTED_CATEGORIES[key])
    if key not in SUPPORTED_CATEGORIES and category.strip() not in SUPPORTED_CATEGORIES:
        supported = ", ".join(sorted({"lychee", "荔枝"}))
        raise ValueError(f"不支持的食物类别: {category}，当前可用: {supported}")
    return SUPPORTED_CATEGORIES.get(key, SUPPORTED_CATEGORIES[category.strip()])


def detect(
    image_path: str | Path,
    category: str,
    *,
    sensors: str | Path | dict[str, Any] | None = None,
    calibrate: bool = False,
    skip_standardize: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    """
    对单果图片进行腐败检测。

    Parameters
    ----------
    image_path : 单果图片路径（jpg/png/jpeg）。不使用 sensors JSON 里的 camera.path。
    category : 食物类别，当前支持 lychee / 荔枝
    sensors : 可选，example.json 风格传感器输入（路径或 dict）。气体信息目前仅透传。
    calibrate : 是否做桌面白色标定（全图输入时可开启）
    skip_standardize : 跳过白底标准化（输入已是 512×512 白底单果图时可开启）
    device : cuda / cpu，默认自动选择
    """
    sensor_block: dict[str, Any] | None = None
    if sensors is not None:
        try:
            payload = load_sensor_payload(sensors)
            sensor_block = extract_sensors_for_output(payload)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            return {
                "success": False,
                "category": category,
                "error": f"传感器输入无效: {exc}",
            }

    try:
        normalized = normalize_category(category)
    except ValueError as exc:
        result = {
            "success": False,
            "category": category,
            "error": str(exc),
        }
        if sensor_block is not None:
            result["sensor_input"] = sensor_block
        return result

    if normalized != "lychee":
        result = {
            "success": False,
            "category": normalized,
            "error": f"未实现的类别: {normalized}",
        }
        if sensor_block is not None:
            result["sensor_input"] = sensor_block
        return result

    from lychee.predictor import predict_lychee

    try:
        result = predict_lychee(
            image_path,
            calibrate=calibrate,
            skip_standardize=skip_standardize,
            device=device,
        )
    except FileNotFoundError as exc:
        result = {
            "success": False,
            "category": normalized,
            "error": str(exc),
        }
    except ValueError as exc:
        result = {
            "success": False,
            "category": normalized,
            "error": str(exc),
        }
    except Exception as exc:
        result = {
            "success": False,
            "category": normalized,
            "error": f"推理失败: {exc}",
        }

    if sensor_block is not None:
        result["sensor_input"] = sensor_block
        # 便于后续气体融合：原样保留 sensors 字典
        result["sensors"] = sensor_block.get("sensors", {})

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单果食物腐败检测")
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="单果图片路径（独立指定，不使用 --sensors 中的 camera）",
    )
    parser.add_argument("--category", type=str, required=True, help="食物类别 (lychee / 荔枝)")
    parser.add_argument(
        "--sensors",
        type=Path,
        default=None,
        help="传感器 JSON 路径（example.json 格式）；气体暂透传，不参与评分",
    )
    parser.add_argument("--output", type=Path, default=None, help="输出 JSON 文件路径，默认打印到 stdout")
    parser.add_argument("--calibrate", action="store_true", help="启用桌面颜色标定")
    parser.add_argument(
        "--skip-standardize",
        action="store_true",
        help="跳过白底标准化（输入已是标准白底单果图）",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        normalize_category(args.category)
    except ValueError as exc:
        result: dict[str, Any] = {
            "success": False,
            "category": args.category,
            "error": str(exc),
        }
        payload = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.write_text(payload, encoding="utf-8")
        else:
            print(payload)
        sys.exit(1)

    result = detect(
        args.image,
        args.category,
        sensors=args.sensors,
        calibrate=args.calibrate,
        skip_standardize=args.skip_standardize,
        device=args.device,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"结果已写入: {args.output}")
    else:
        print(payload)

    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
