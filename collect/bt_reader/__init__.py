"""蓝牙传感器 + USB 摄像头数据实时读取与本地保存。

固定板卡（与 config.yaml / firmware 一致）:
  esp32_gas / ESP32-Gas — H2S NH3 VOC(模拟量 0~500ppm) CH3SH
  esp32_env / ESP32-Env — C2H5OH C2H4(模拟量 0~20ppm) DHT1 DHT2

摄像头: data/camera/history/ 历史影像；data/latest.jpg 最新影像
"""

from .parser import LineParser, ParsedFrame, SensorReading
from .storage import DataStore

__all__ = [
    "LineParser",
    "SensorReading",
    "ParsedFrame",
    "DataStore",
    "LatestSnapshot",
    "BluetoothDevice",
    "DeviceManager",
    "CameraCapture",
]


def __getattr__(name: str):
    if name in ("BluetoothDevice", "DeviceManager"):
        from .device import BluetoothDevice, DeviceManager

        return {"BluetoothDevice": BluetoothDevice, "DeviceManager": DeviceManager}[name]
    if name == "LatestSnapshot":
        from .snapshot import LatestSnapshot

        return LatestSnapshot
    if name == "CameraCapture":
        from .camera import CameraCapture

        return CameraCapture
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
