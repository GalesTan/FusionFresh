#!/usr/bin/env python3
"""蓝牙传感器 + USB 摄像头数据采集入口。

板卡（见 config.yaml / firmware/）:
  ESP32-Gas  → esp32_gas  → H2S NH3 VOC(模拟量 0~500ppm) CH3SH
  ESP32-Env  → esp32_env  → C2H5OH C2H4(模拟量 0~20ppm) DHT1 DHT2

摄像头（见 config.yaml camera）:
  定时抓拍 → data/camera/history/ 历史影像
           → data/latest.jpg       最新影像（蓝牙传输用）

用法:
  1. 与两台 ESP32 蓝牙配对（名称: ESP32-Gas / ESP32-Env）
  2. 在 Windows「设备管理器 → 端口」确认 COM 口，写入 config.yaml
  3. 连接 Logitech 等 USB 摄像头
  4. pip install -r requirements.txt
  5. python main.py

可选关闭（覆盖 config.yaml 中对应 enabled）:
  python main.py --no-camera
  python main.py --no-esp32-gas --no-esp32-env
  python main.py --no-camera --no-esp32-gas   # 只跑环境板等

移动端实时：data/latest.json + data/latest.jpg（见 config.yaml）
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from bt_reader.camera import CameraCapture
from bt_reader.device import DeviceManager
from bt_reader.parser import ParsedFrame, build_parsers
from bt_reader.snapshot import LatestSnapshot
from bt_reader.spike_clean import spike_clean_config_from_dict


ROOT = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_spike_clean(manager: DeviceManager, spike_cfg) -> int:
    """在写锁内巡检 CSV，用前后邻域中点替代尖峰/骤降。"""
    results = manager.clean_spike_csvs(
        min_abs_delta=spike_cfg.min_abs_delta,
        neighbor_factor=spike_cfg.neighbor_factor,
        max_passes=spike_cfg.max_passes,
    )
    total = sum(results.values())
    if total:
        logging.info("尖峰清洗完成：共修正 %d 处（%d 个文件）", total, len(results))
    else:
        logging.debug("尖峰清洗：未发现需修正数据")
    return total


def print_frame(frame: ParsedFrame) -> None:
    parts = [
        f"{r.sensor}={r.value}{r.unit}"
        + (f"(V={r.voltage})" if r.voltage is not None else "")
        for r in frame.readings
    ]
    ts = frame.timestamp.strftime("%H:%M:%S")
    print(f"[{ts}] {frame.device_id}: " + " | ".join(parts), flush=True)


def build_snapshot(config: dict, project_root: Path) -> LatestSnapshot:
    cfg = config.get("snapshot") or {}
    path = project_root / cfg.get("path", "data/latest.json")
    return LatestSnapshot(
        path=path,
        interval=float(cfg.get("interval", 1.0)),
        enabled=bool(cfg.get("enabled", True)),
    )


def apply_cli_disables(config: dict, *, no_camera: bool, no_esp32_gas: bool, no_esp32_env: bool) -> None:
    """用命令行开关覆盖 config 中 camera / 设备的 enabled。"""
    if no_camera:
        config.setdefault("camera", {})["enabled"] = False
        logging.info("CLI: 已禁用 camera")

    disabled_ids: set[str] = set()
    if no_esp32_gas:
        disabled_ids.add("esp32_gas")
    if no_esp32_env:
        disabled_ids.add("esp32_env")
    if not disabled_ids:
        return

    for item in config.get("devices") or []:
        if item.get("id") in disabled_ids:
            item["enabled"] = False
            logging.info("CLI: 已禁用设备 %s", item.get("id"))


def main() -> int:
    parser = argparse.ArgumentParser(description="蓝牙多设备传感器采集")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=ROOT / "config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="输出调试日志",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="不启动 USB 摄像头（覆盖 config.yaml camera.enabled）",
    )
    parser.add_argument(
        "--no-esp32-gas",
        action="store_true",
        help="不连接 esp32_gas 气体板",
    )
    parser.add_argument(
        "--no-esp32-env",
        action="store_true",
        help="不连接 esp32_env 环境板",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.config.exists():
        logging.error("配置文件不存在: %s", args.config)
        return 1

    config = load_config(args.config)
    apply_cli_disables(
        config,
        no_camera=args.no_camera,
        no_esp32_gas=args.no_esp32_gas,
        no_esp32_env=args.no_esp32_env,
    )
    snapshot = build_snapshot(config, ROOT)
    spike_cfg = spike_clean_config_from_dict(config.get("spike_clean"))

    def on_frame(frame: ParsedFrame) -> None:
        snapshot.update(frame)
        print_frame(frame)

    def on_camera(meta: dict[str, Any]) -> None:
        snapshot.update_camera(meta)
        ts = str(meta.get("timestamp", ""))[-12:]
        print(
            f"[{ts}] camera: {meta.get('device')} "
            f"{meta.get('width')}x{meta.get('height')} "
            f"→ latest.jpg / {meta.get('history_file')}",
            flush=True,
        )

    parsers = build_parsers(config.get("parsers", {}))
    manager = DeviceManager.from_config(
        config,
        parsers,
        project_root=str(ROOT),
        on_frame=on_frame,
        spike_cfg=spike_cfg,
    )
    camera = CameraCapture.from_config(config, ROOT, on_capture=on_camera)

    if not manager.devices and camera is None:
        logging.error(
            "没有启用的设备或摄像头，请检查 config.yaml / "
            "--no-camera --no-esp32-gas --no-esp32-env"
        )
        return 1

    stop_flag = {"stop": False}

    def _handle_signal(*_args) -> None:
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    if manager.devices:
        manager.start_all()
        logging.info("已启动 %d 台蓝牙设备", len(manager.devices))
    if camera is not None:
        camera.start()
    logging.info("运行中，Ctrl+C 退出；数据目录: %s/data/", ROOT)
    if snapshot.enabled:
        logging.info(
            "移动端快照: %s (每 %.1fs 更新)",
            snapshot.path,
            snapshot.interval,
        )
    if spike_cfg.enabled and manager.devices:
        logging.info(
            "尖峰清洗: 每 %.0fs 巡检 CSV (min_abs=%.1f, online=%s)",
            spike_cfg.interval,
            spike_cfg.min_abs_delta,
            spike_cfg.online,
        )

    last_spike_clean = 0.0
    try:
        while not stop_flag["stop"]:
            snapshot.maybe_flush()
            now = time.monotonic()
            if (
                spike_cfg.enabled
                and manager.devices
                and now - last_spike_clean >= spike_cfg.interval
            ):
                try:
                    run_spike_clean(manager, spike_cfg)
                except Exception:
                    logging.exception("尖峰清洗异常")
                last_spike_clean = now
            time.sleep(0.2)
    finally:
        logging.info("正在停止...")
        if camera is not None:
            camera.stop()
        snapshot.flush()
        manager.stop_all()
        logging.info("已退出")

    return 0


if __name__ == "__main__":
    sys.exit(main())
