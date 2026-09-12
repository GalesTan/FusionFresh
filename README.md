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
        ├─ 虾仁  → EfficientNet-B0 多任务评分 + 气体融合
        └─ 其它  → Qwen VL 评估裁剪图
        │
        ▼
outputs/latest/detection_results.json
        │
        ▼
HTTP：/detect  /latest  /stream  /health  /history
```

接口字段、阈值、定时策略、气体规则与原项目一致。App 三个 URL 怎么填见下文；字段契约见 [docs/APP_API.md](docs/APP_API.md)。

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

---

## 下载权重

第三方检测权重体积大（合计约 3.5 GB），**不进 Git**。克隆仓库后放到下面两个固定路径即可。荔枝 / 虾仁评分头（`.pt`）已随仓库提交；DINOv2 荔枝骨干首次运行会由 `torch.hub` 自动下载（流水线默认 `HF_ENDPOINT=https://hf-mirror.com`）。

| 文件 | 约大小 | 放置路径 |
|------|--------|----------|
| RAM++ | 2.9 GB | `vendor/recognize-anything/pretrained/ram_plus_swin_large_14m.pth` |
| GroundingDINO SwinT | 662 MB | `vendor/GroundingDINO/weights/groundingdino_swint_ogc.pth` |

PowerShell（推荐 `curl.exe`，可断点续传）：

```powershell
New-Item -ItemType Directory -Force -Path `
  vendor\recognize-anything\pretrained, `
  vendor\GroundingDINO\weights | Out-Null

# RAM++（Hugging Face；国内走镜像）
curl.exe -L --retry 5 -C - `
  "https://hf-mirror.com/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth" `
  -o "vendor\recognize-anything\pretrained\ram_plus_swin_large_14m.pth"

# GroundingDINO（官方 GitHub Release）
curl.exe -L --retry 5 -C - `
  "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" `
  -o "vendor\GroundingDINO\weights\groundingdino_swint_ogc.pth"
```

GitHub 较慢时可改用 Hugging Face 镜像：

```powershell
curl.exe -L --retry 5 -C - `
  "https://hf-mirror.com/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth" `
  -o "vendor\GroundingDINO\weights\groundingdino_swint_ogc.pth"
```

若本机已有这两份文件（例如原 `FusionFreshSever`），复制到上表路径即可，不必重新下载。`tagger=vl` 且 VL 可用时可以暂时不放 RAM++，但 GroundingDINO 始终需要。

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

手机 App 三个空栏的填写见下一节。

### 3. Demo（无硬件也可跑）

```powershell
# 对一张 Demo 图跑完整流水线
python demo/run_demo.py --image demo/demo6.jpg --cpu-only

# 浏览检测结果卡片
python demo/view_results.py --outputs outputs
```

浏览器打开提示的地址。Demo 说明见 [demo/README.md](demo/README.md)。

---

## App 三个 URL 怎么填

手机 App 设置里有三栏：**算法服务器 url**、**自动检测 url**、**数据服务器 url**。手机和电脑须同一 Wi‑Fi；算法服务必须 `--host 0.0.0.0`。

### 1. 查电脑局域网 IP

```powershell
ipconfig
```

看「无线局域网适配器 WLAN」的 IPv4（下文写成 `<IP>`）。换网络后会变，要改 App 里三处地址。

本机模拟器可用 `127.0.0.1` 代替 `<IP>`。

### 2. 启动两台服务

算法服务（已在上一节）：

```powershell
python detect_server.py --cpu-only --host 0.0.0.0 --port 4100
```

数据服务（另开一个终端，给 App 拉传感器快照和最新照片）：

```powershell
cd collect\data
python -m http.server 8000 --bind 0.0.0.0
```

Windows 防火墙若拦截 4100 / 8000，需放行入站。

### 3. 填入 App

把 `<IP>` 换成上一步查到的地址：

| App 空栏 | 填写 | 对应接口 |
|----------|------|----------|
| **算法服务器 url** | `http://<IP>:4100/detect` | 「检测」按钮：立刻重跑流水线（数十秒～数分钟） |
| **自动检测 url** | `http://<IP>:4100/stream` | SSE 长连接，只推送**定时**检测结果 |
| **数据服务器 url** | `http://<IP>:8000` | 静态目录 `collect/data/`：`/latest.json`、`/latest.jpg` |

App 会把「算法服务器 url」整段拿去 `POST`，**不会**自动补 `/detect`。只填 `http://<IP>:4100` 时，服务端日志是 `POST / … 404`，检测不会跑。必须写成带 `/detect` 的完整地址。

若自动检测连不上，可改填轮询地址 `http://<IP>:4100/latest`（读缓存，毫秒级；与 SSE 二选一即可）。

### 4. 自检

浏览器或本机：

```text
GET  http://127.0.0.1:4100/health
GET  http://127.0.0.1:4100/latest
GET  http://127.0.0.1:8000/latest.json
```

`/health` 能返回 JSON 说明算法服务已起来。手机上再改成 `<IP>` 填进 App。点「检测」时终端应出现 `POST /detect … 200`，并卡住一段时间（在跑模型），不要立刻 404。

字段含义、错误码见 [docs/APP_API.md](docs/APP_API.md)。VL 打标签需先配置 `qwen/.env`（见上文「环境」）；改 `.env` 后要重启 `detect_server.py`。

---

## 扩展

加新品类、换评分模型、接新传感器：见 [docs/extending.md](docs/extending.md)。

原则：流水线 `food_pipeline.py` 与 HTTP 契约保持稳定；新品类作为 `spoilage/` 下的独立包接入。
