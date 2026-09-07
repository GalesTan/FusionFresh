# 硬件采集

对应原项目 `fusionfresh_main/`。协议、COM 口字段、清洗参数保持不变。

## 板卡

| 板 | 蓝牙名 | 传感器 |
|----|--------|--------|
| `esp32_gas` | ESP32-Gas | H2S、NH3、VOC、CH3SH |
| `esp32_env` | ESP32-Env | C2H5OH、C2H4、DHT1 温湿度、DHT2 温湿度 |

固件：`collect/firmware/esp32_gas/esp32_gas.ino`、`collect/firmware/esp32_env/esp32_env.ino`

统一行协议（两板共用）：

```text
SENSOR:value unit[|KEY:VAL]*
```

`#` 开头为注释。多传感器空格分隔。

## 配置

编辑 `collect/config.yaml`：

1. 与两台 ESP32 蓝牙配对  
2. 在设备管理器确认 COM 口，写入 `devices[].port`  
3. 摄像头：C920 使用 `backend: msmf`；内置摄像头常用 `dshow`

相对路径均相对于 `collect/`。

## 启动

在仓库根目录或 `collect/` 下均可：

```powershell
python collect/main.py
```

可选：`--no-camera`、`--no-esp32-gas`、`--no-esp32-env`

## 输出

| 路径 | 说明 |
|------|------|
| `collect/data/latest.jpg` | 最新帧，检测服务默认输入 |
| `collect/data/latest.json` | 全传感器最新值，供移动端拉取 |
| `collect/data/camera/history/` | 历史抓拍 |
| `collect/data/esp32_gas/*.csv` | 气体时序（虾仁气体融合读取 VOC/H2S/NH3） |
| `collect/data/esp32_env/*.csv` | 环境时序（融合读取 C2H5OH） |

本集成包**不包含**原项目里的历史 CSV/影像。接上硬件后会重新落盘。
