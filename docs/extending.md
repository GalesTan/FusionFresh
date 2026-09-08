# 如何扩展

目标：新增能力时尽量**只加模块、不改流水线契约**。App 仍读 `docs/APP_API.md` 中的字段。

## 1. 给新品类加专用评分模型

当前分支在 `food_pipeline.py` 的 `_spoilage_for_detection`：荔枝 / 虾仁走专用模型，其余走 VL。

推荐步骤：

1. 在 `spoilage/<name>/` 新建独立包，导出 `XxxPredictor.predict(image_path) -> dict`  
   至少包含：`success`、`score`（0–1）、`rating_id`（0/1/2）或与现有 `spoilage_level` 可映射的字段。
2. 权重放在该包的 `assets/` 或 `models/`，用相对 `__file__` 的路径加载（参考虾仁、荔枝）。
3. 在 `food_pipeline.py` 增加 `_is_<name>()` 与 `_spoilage_from_<name>_score()`，并在 `_spoilage_for_detection` 里加一个分支。  
   **不要改**已有荔枝/虾仁判定与 JSON 字段名。
4. 若要在 `detect_server.py` 启动时预加载，仿照 `_MODELS['lychee']` / `['shrimp']` 加一项，避免每次请求重新加载。

`spoilage/` 下各包互不 import，避免循环依赖。

## 2. 气体融合扩到其它食物

虾仁规则在 `spoilage/shrimp/gas_fusion.py`，当检测结果**含有虾仁**时由 `build_detection_results` 调用（全虾仁才允许 H2S/NH3 强制改等级）。

新品类融合建议：

- 新文件 `spoilage/<name>/gas_fusion.py`，不要把规则塞进 HTTP 层  
- 在 `build_detection_results` 里按「是否仅该品类」决定是否调用  
- 传感器 CSV 仍从 `collect/data/` 读，路径约定见现有 `_SENSOR_FILES`

荔枝目前**不用**气体评分，保持这一行为除非产品明确要求。

## 3. 新传感器

1. 固件：`collect/firmware/` 按现有 `SENSOR:value unit` 行协议追加字段  
2. `collect/config.yaml` 的 `devices` 不必列传感器名，解析器按行拆  
3. 采集端会按传感器名拆 CSV；融合侧在 `_SENSOR_FILES` 登记文件名即可

## 4. 换 VL 或检测器

- 换 Qwen 模型：改 `qwen/.env` 的 `QWEN_VL_MODEL` / `DASHSCOPE_BASE_URL`，不必改流水线  
- 换 Tagger：`detect_server.py --tagger vl|ram|auto`  
- GroundingDINO / RAM++ 权重路径在 `detect_server.py` 的 `CONFIG` 与 CLI 覆盖参数里

第三方代码在 `vendor/`，升级时替换该目录并核对 `food_pipeline.py` 顶部的 `sys.path`。

## 5. 不要动的契约

App 依赖这些名字与语义（详见 `docs/APP_API.md`）：

- `detectedFoods[].spoilageScore` / `spoilageLevel` / `type_probability` / `boundingBox`
- `isSpoiled` 仅当存在 `spoilageLevel == "spoiled"`
- `/detect` 更新缓存但不推 `/stream`；定时任务才推 SSE
