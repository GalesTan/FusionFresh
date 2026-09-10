#!/usr/bin/env python3
"""用与线上相同的 food_pipeline 对 Demo 图片做一次检测。"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="FusionFresh Demo：跑完整检测流水线")
    parser.add_argument(
        "--image",
        default=str(ROOT / "demo" / "demo6.jpg"),
        help="输入图片（默认 demo/demo6.jpg）",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录（默认 outputs/demo_<图名>）",
    )
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--tagger",
        choices=["auto", "vl", "ram"],
        default="auto",
        help="与 detect_server / food_pipeline 相同",
    )
    args = parser.parse_args()

    raw = Path(args.image).expanduser()
    candidates = [raw] if raw.is_absolute() else [
        Path.cwd() / raw,
        ROOT / raw,
        ROOT / "demo" / raw.name,
    ]
    image = next((p.resolve() for p in candidates if p.is_file()), None)
    if image is None:
        raise SystemExit(f"找不到图片: {args.image}")

    out_dir = Path(args.output_dir) if args.output_dir else (
        ROOT / "outputs" / f"demo_{image.stem}"
    )

    cmd = [
        sys.executable,
        str(ROOT / "food_pipeline.py"),
        "--image", str(image),
        "--ram-checkpoint", str(
            ROOT / "vendor" / "recognize-anything" / "pretrained"
            / "ram_plus_swin_large_14m.pth"
        ),
        "--gdino-checkpoint", str(
            ROOT / "vendor" / "GroundingDINO" / "weights"
            / "groundingdino_swint_ogc.pth"
        ),
        "--output-dir", str(out_dir),
        "--tagger", args.tagger,
    ]
    if args.cpu_only:
        cmd.append("--cpu-only")

    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
