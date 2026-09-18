# FusionFresh

基于视觉与气体传感的食材新鲜度监测系统：采集端持续抓拍并记录传感器数据，检测流水线识别食物并给出腐败评分，HTTP 服务向手机 App 提供检测结果。

当前阶段已打通 **硬件采集 → 检测评分 → App 接口** 的完整链路。荔枝、虾仁走专用视觉模型；其它品类由视觉大模型评估。

---

## 功能

- **采集**：ESP32 气体/环境板（蓝牙）+ USB 摄像头，持续写入最新照片与传感器快照
- **识别与定位**：Qwen VL / RAM++ 打食物标签，GroundingDINO 框选并裁剪
- **新鲜度评分**
  - 荔枝 → DINOv2 多任务评分
  - 虾仁 → EfficientNet-B0 多任务评分，并融合近 3 分钟气体数据
  - 其它食物 → Qwen VL 评估裁剪图
- **服务**：定时检测、手动检测、结果缓存、SSE 推送、历史记录
- **Demo**：无硬件时可用示例图走同一条流水线，并在浏览器查看结果卡片

---

## 检测流水线

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

一次检测的步骤：

1. **Tagger**（默认 `auto`）：先用 Qwen VL 得到英文食物标签与期望数量；失败则回退到 RAM++。
2. **Detect**：标签拼成 GroundingDINO caption，框选并裁剪到 `outputs/.../crops/`。
3. **Score**：荔枝 / 虾仁走专用模型，其余走 VL；VL 失败时该条目 `spoilageScore` 为 `null`。
4. **Gas fusion**：画面中含虾仁时，读取最近约 180s 的 H2S / NH3 / VOC / C2H5OH。全是虾仁且 H2S 或 NH3 持续检出时强制为 `spoiled`；VOC 或乙醇偏高只加分、不改等级。画面中无虾仁时不改视觉分数。
5. **Serve**：写入 `detection_results.json`；HTTP 层再补 `generated_at` / `age_seconds` / `trigger`。

字段契约见 [docs/APP_API.md](docs/APP_API.md)。模块划分见 [docs/architecture.md](docs/architecture.md)。

---



## 仓库结构

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

---



## 环境

建议 Python 3.10+，检测服务需要 PyTorch（可用 conda 环境 `food_pipe`）。

```powershell
cd FusionFresh
pip install -r requirements.txt
```

视觉大模型（Qwen VL）需要 API Key：

```powershell
copy qwen\.env.example qwen\.env
```

编辑 `qwen/.env`，填入有效的 `DASHSCOPE_API_KEY`。修改后需重启 `detect_server.py` 才会生效。

---



## 下载权重

第三方检测权重体积较大（合计约 3.5 GB），**不纳入 Git**。克隆仓库后放到下列固定路径即可。荔枝 / 虾仁评分头（`.pt`）已随仓库提交；DINOv2 荔枝骨干首次运行会由 `torch.hub` 自动下载（流水线默认 `HF_ENDPOINT=https://hf-mirror.com`）。


| 文件                  | 约大小    | 放置路径                                                               |
| ------------------- | ------ | ------------------------------------------------------------------ |
| RAM++               | 2.9 GB | `vendor/recognize-anything/pretrained/ram_plus_swin_large_14m.pth` |
| GroundingDINO SwinT | 662 MB | `vendor/GroundingDINO/weights/groundingdino_swint_ogc.pth`         |


PowerShell（推荐 `curl.exe`，支持断点续传）：

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

GitHub 较慢时可改用 Hugging Face 镜像下载 GroundingDINO：

```powershell
curl.exe -L --retry 5 -C - `
  "https://hf-mirror.com/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth" `
  -o "vendor\GroundingDINO\weights\groundingdino_swint_ogc.pth"
```

本机若已有这两份 `.pth`，复制到上表路径即可，不必重新下载。`tagger=vl` 且 VL 可用时可以暂时不放 RAM++，但 GroundingDINO 始终需要。

---



## 快速开始



### Demo（无硬件）

```powershell
# 对一张 Demo 图跑完整流水线
python demo/run_demo.py --image demo/demo6.jpg --cpu-only

# 浏览检测结果卡片
python demo/view_results.py --outputs outputs
```

浏览器打开提示的地址即可。更多说明见 [demo/README.md](demo/README.md)。

### 采集（有硬件时）

先在 `collect/config.yaml` 填好 COM 口，再：

```powershell
python collect/main.py
```

会持续写入：

- `collect/data/latest.jpg` — 最新相机帧（检测服务读这张图）
- `collect/data/latest.json` — 传感器快照
- `collect/data/esp32_gas/`、`collect/data/esp32_env/` — 分传感器 CSV / JSONL

硬件协议与配置见 [docs/hardware.md](docs/hardware.md)。

### 算法服务（给 App）

```powershell
python detect_server.py --cpu-only --host 0.0.0.0 --port 4100
```


| 参数               | 含义                    |
| ---------------- | --------------------- |
| `--interval 2`   | 定时检测间隔（小时，默认 2）       |
| `--no-scheduler` | 只响应手动 `/detect`       |
| `--tagger auto`  | `auto` / `vl` / `ram` |


日常部署建议同时运行采集进程与算法服务：前者写 `latest.jpg` 与气体 CSV，后者默认每 2 小时检测一次，App 通过 `/latest` 或 `/stream` 取结果。

---



## 对接手机 App

手机 App 设置里有三栏：**算法服务器 url**、**自动检测 url**、**数据服务器 url**。手机和电脑须同一 Wi‑Fi；算法服务必须 `--host 0.0.0.0`。

### 1. 查电脑局域网 IP

```powershell
ipconfig
```

看「无线局域网适配器 WLAN」的 IPv4（下文写成 `<IP>`）。换网络后地址会变，需同步改 App 里三处。本机模拟器可用 `127.0.0.1` 代替 `<IP>`。

### 2. 启动两台服务

算法服务：

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


| App 空栏        | 填写                        | 对应接口                                              |
| ------------- | ------------------------- | ------------------------------------------------- |
| **算法服务器 url** | `http://<IP>:4100/detect` | 「检测」按钮：立刻重跑流水线（数十秒～数分钟）                           |
| **自动检测 url**  | `http://<IP>:4100/stream` | SSE 长连接，只推送**定时**检测结果                             |
| **数据服务器 url** | `http://<IP>:8000`        | 静态目录 `collect/data/`：`/latest.json`、`/latest.jpg` |


App 会把「算法服务器 url」整段拿去 `POST`，**不会**自动补 `/detect`。只填 `http://<IP>:4100` 时，服务端日志是 `POST / … 404`，检测不会跑。必须写成带 `/detect` 的完整地址。

若自动检测连不上，可改填轮询地址 `http://<IP>:4100/latest`（读缓存，毫秒级；与 SSE 二选一即可）。

### 4. 自检

```text
GET  http://127.0.0.1:4100/health
GET  http://127.0.0.1:4100/latest
GET  http://127.0.0.1:8000/latest.json
```

`/health` 能返回 JSON 说明算法服务已起来。手机上再改成 `<IP>` 填进 App。点「检测」时终端应出现 `POST /detect … 200`，并卡住一段时间（在跑模型），不要立刻 404。

---



## HTTP 接口


| 方法         | 路径         | 说明                                           | 耗时      |
| ---------- | ---------- | -------------------------------------------- | ------- |
| GET / POST | `/detect`  | 立即重跑完整流水线，更新 `/latest` 缓存；**不**推送到 `/stream` | 数十秒～数分钟 |
| GET        | `/latest`  | 返回缓存的最新检测结果                                  | 毫秒级     |
| GET        | `/stream`  | SSE：仅推送定时检测完成的结果                             | 长连接     |
| GET        | `/health`  | 服务状态、缓存新鲜度                                   | 毫秒级     |
| GET        | `/history` | 历史检测记录（可选 `?limit=` / `?since=`）             | 毫秒级     |


建议 App「检测」按钮调 `/detect`（需提示用户等待）；后台展示用 `/latest` 或订阅 `/stream`。字段含义与错误码见 [docs/APP_API.md](docs/APP_API.md)。

---



## 扩展

加新品类、换评分模型、接新传感器：见 [docs/extending.md](docs/extending.md)。

原则：流水线 `food_pipeline.py` 与 HTTP 契约保持稳定；新品类作为 `spoilage/` 下的独立包接入。

---



## 文档


| 文档                                           | 内容                   |
| -------------------------------------------- | -------------------- |
| [docs/architecture.md](docs/architecture.md) | 模块划分与运行时数据           |
| [docs/APP_API.md](docs/APP_API.md)           | App 接口字段与错误码         |
| [docs/hardware.md](docs/hardware.md)         | ESP32 协议、COM 口与摄像头配置 |
| [docs/extending.md](docs/extending.md)       | 新品类、气体融合、传感器扩展       |
| [demo/README.md](demo/README.md)             | Demo 图与可视化           |


