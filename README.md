# FusionFresh

食材新鲜度监测系统的**阶段性集成软件**：采集端 + 检测流水线 + App HTTP 服务。

本目录由 `FusionFreshSever` 按当前工作流整理而来。原项目未改动；这里去掉了历史抓拍、实验输出、旧模型与训练脚本，只保留线上正在使用的能力。

---

## 当前工作流（不要改细节）

```
ESP32 气体/环境板 + USB 摄像头
        │  collect/
        ▼
collect/data/latest.jpg  +  气体 CSV
        │
        ▼
VL / RAM++ 打食物标签  →  GroundingDINO 框选裁剪
        │
        ├─ 荔枝  → DINOv2 多任务评分
        ├─ 虾仁  → EfficientNet-B0 多任务评分
        │         （全是虾仁时，再按气体规则覆盖等级）
        └─ 其它  → Qwen VL 评估裁剪图
        │
        ▼
outputs/latest/detection_results.json
        │
        ▼
HTTP：/detect  /latest  /stream  /health  /history
```

接口字段、阈值、定时策略、气体规则与原项目一致。对接说明见 [docs/APP_API.md](docs/APP_API.md)。

---

## 目录

```
FusionFresh/
├── detect_server.py      # App 算法服务入口
├── food_pipeline.py      # 检测流水线（也可单独 CLI）
├── collect/              # 硬件采集（蓝牙传感器 + 摄像头）
├── spoilage/             # 品类腐败模型（荔枝 / 虾仁）
├── qwen/                 # 视觉大模型打标签与通用腐败评估
├── vendor/               # 第三方推理依赖（RAM++、GroundingDINO）
├── demo/                 # Demo 图片 + 可视化
├── outputs/              # 运行时检测结果（空目录，运行后生成）
└── docs/                 # 架构、接口、硬件、扩展
```

更细的模块说明：[docs/architecture.md](docs/architecture.md)

---

## 环境

建议 Python 3.10+。检测服务需要 PyTorch 环境（原项目常用 conda env `food_pipe`）。

```powershell
cd FusionFresh
pip install -r requirements.txt
```

视觉大模型（Qwen VL）需要 API Key：

```powershell
copy qwen\.env.example qwen\.env
# 把 DASHSCOPE_API_KEY 改成有效密钥
```

若原项目 `FusionFreshSever\qwen\.env` 已有密钥，直接复制过来即可。

DINOv2 荔枝骨干首次运行会从 `torch.hub` 加载（国内可设 `HF_ENDPOINT=https://hf-mirror.com`，流水线已默认该镜像）。

---

## 启动

### 1. 采集（有硬件时）

先在 `collect/config.yaml` 填好 COM 口，再：

```powershell
python collect/main.py
```

会持续写入：

- `collect/data/latest.jpg` — 最新相机帧（检测服务读这张图）
- `collect/data/latest.json` — 传感器快照
- `collect/data/esp32_gas/`、`collect/data/esp32_env/` — 分传感器 CSV/JSONL

详见 [docs/hardware.md](docs/hardware.md)。

### 2. 算法服务（给 App）

```powershell
python detect_server.py --cpu-only --host 0.0.0.0 --port 4100
```

常用参数与原来相同：

| 参数 | 含义 |
|------|------|
| `--interval 2` | 定时检测间隔（小时，默认 2） |
| `--no-scheduler` | 只响应手动 `/detect` |
| `--tagger auto` | `auto` / `vl` / `ram` |

### 3. Demo（无硬件也可跑）

```powershell
# 对一张 Demo 图跑完整流水线
python demo/run_demo.py --image demo/demo6.jpg --cpu-only

# 浏览检测结果卡片
python demo/view_results.py --outputs outputs
```

浏览器打开提示的地址。Demo 说明见 [demo/README.md](demo/README.md)。

---

## 扩展

加新品类、换评分模型、接新传感器：见 [docs/extending.md](docs/extending.md)。

原则：流水线 `food_pipeline.py` 与 HTTP 契约保持稳定；新品类作为 `spoilage/` 下的独立包接入。
