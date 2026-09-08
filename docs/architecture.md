# 架构说明

FusionFresh 按**采集 / 感知 / 评分 / 服务**四层划分。算法步骤、字段名、阈值与原 `FusionFreshSever` 当前工作流一致，只调整了目录边界。

## 模块

| 模块 | 路径 | 职责 | 对外依赖 |
|------|------|------|----------|
| 采集 | `collect/` | 蓝牙读 ESP32、摄像头抓拍、尖峰清洗、latest 快照 | pyserial, OpenCV |
| 流水线 | `food_pipeline.py` | 打标签 → 检测裁剪 → 按品类评分 → 组装 JSON | 下列全部 |
| HTTP 服务 | `detect_server.py` | 定时/手动检测、缓存、SSE、历史 | 流水线 |
| 荔枝评分 | `spoilage/lychee/` | DINOv2 + 多任务头 | torch, torchvision |
| 虾仁评分 | `spoilage/shrimp/` | EfficientNet-B0 + 多任务头；含虾仁场景气体融合 | torch, OpenCV |
| 视觉语言模型 | `qwen/` | 食物标签 + 非荔枝/虾仁裁剪图腐败评估 | OpenAI 兼容 API |
| RAM++ | `vendor/recognize-anything/` | VL 失败时的标签回退 | torch, transformers, timm |
| GroundingDINO | `vendor/GroundingDINO/` | 开放词汇框选 | torch, transformers |
| Demo | `demo/` | 示例图、一键跑流水线、结果可视化 | 流水线；可视化另需 Flask |

## 运行时数据（不入库）

```
collect/data/latest.jpg          检测输入图
collect/data/latest.json         传感器最新值
collect/data/esp32_gas/*.csv     气体历史（虾仁融合读取）
collect/data/esp32_env/*.csv
outputs/latest/                  最近一次检测（会被覆盖）
outputs/history.jsonl            历次检测追加日志
```

历史实验图、旧 `outputs/demo_*`、1 万+ 抓拍未拷贝；本包运行后会重新生成 `outputs/`。

## 检测一次做了什么

1. **Tagger**（`tagger=auto`）  
   先调 Qwen VL 得到英文食物标签与期望数量；失败则加载 RAM++。
2. **Detect**  
   标签拼成 GroundingDINO caption，框选并裁剪到 `outputs/.../crops/`。  
   若 VL 给了 `food_counts`，每种食物按该数量截取置信度最高的框。
3. **Score**  
   - 荔枝 → `LycheePredictor`  
   - 虾仁 → `ShrimpPredictor`  
   - 其它 → `qwen.assess_spoilage`；失败则占位（`spoilageScore=null`）
4. **Gas fusion（画面中含虾仁时）**  
   读最近约 180s 的 H2S/NH3/VOC/C2H5OH：  
   - 全是虾仁：H2S 或 NH3 持续检出 → 强制 `spoiled`；VOC 或乙醇偏高 → 只加分、不改等级  
   - 混合食品：上述气体均只加分、不强制改等级  
   画面中无虾仁时不改视觉分数。
5. **Serve**  
   写入 `detection_results.json`；HTTP 再补 `generated_at` / `age_seconds` / `trigger`。

## 进程怎么配合

日常部署是两个常驻进程：

1. `python collect/main.py` — 写 `latest.jpg` 与气体 CSV  
2. `python detect_server.py ...` — 读 `latest.jpg`，默认每 2 小时跑一次，App 拉 `/latest` 或订 `/stream`

没有采集进程时，可用 Demo 图走 `food_pipeline.py` / `demo/run_demo.py`，行为与线上同一条流水线。

## 为何第三方在 `vendor/`

RAM++ 与 GroundingDINO 只保留推理所需代码和权重，去掉训练、Docker、Gradio。通过 `sys.path` 注入，**不改它们内部实现**。

## 相对原项目裁掉了什么

未拷贝（不影响当前工作流）：

- `history/`、`sampled_*`、采集 CSV 与历史影像
- 旧实验 `outputs/`、`detect_server_old.py`、`food_pipeline_old.py`
- RAM++ / GroundingDINO 的训练、Docker、Gradio
- 虾仁旧 DINOv2+PCA/Ridge 与 patch 拼图训练代码
- 荔枝旧 joblib 头
- `.claude`、`graphify-out` 等过程文件

原目录 `FusionFreshSever` 未修改。
