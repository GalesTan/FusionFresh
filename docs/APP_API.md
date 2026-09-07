# FusionFresh 检测结果 — App 对接说明

算法服务：`detect_server.py`  
默认地址：`http://<host>:4100`  
编码：`UTF-8`，`Content-Type: application/json; charset=utf-8`

---

## 1. 接口一览

| 方法 | 路径 | 说明 | 耗时 |
|------|------|------|------|
| GET/POST | `/detect` | **立即重跑**完整 pipeline，返回结果并更新 `/latest` 缓存；**不**推送到 `/stream` | 数十秒～数分钟 |
| GET | `/latest` | 返回**缓存**的最新检测结果（定时或手动产生） | 毫秒级 |
| GET | `/stream` | SSE：仅推送**定时**检测完成的结果 | 长连接 |
| GET | `/health` | 服务状态、缓存新鲜度 | 毫秒级 |
| GET | `/history` | 历史检测记录（可选 `?limit=` / `?since=`） | 毫秒级 |

建议 App「检测」按钮调 `/detect`（需提示用户等待）；后台展示用 `/latest` 或订阅 `/stream`。

---

## 2. `/detect` 响应

### 2.1 顶层字段

| 字段 | 类型 | 必有 | 说明 |
|------|------|------|------|
| `isSpoiled` | boolean | ✓ | 是否存在 `spoilageLevel == "spoiled"` 的条目 |
| `message` | string | ✓ | 总览文案（给人看） |
| `detectedFoods` | array | ✓ | 检测到的食物列表；无食物时为 `[]` |
| `tagger` | string | 通常有 | 标签来源：`"vl"` 或 `"ram"` |
| `expectedCounts` | object | 可选 | VL 给出的期望数量，如 `{"peeled shrimp": 1}`；键为英文标签 |
| `generated_at` | string | ✓（HTTP） | 本次结果生成时间，ISO-8601，含时区 |
| `age_seconds` | number\|null | ✓（HTTP） | 结果距今秒数 |
| `trigger` | string\|null | ✓（HTTP） | `"scheduled"`（定时）或 `"manual"`（`/detect` 按钮）；便于 App 区分来源 |
| `next_run_in_seconds` | number | 可选 | 距下次定时检测的秒数 |
| `last_error` | string\|null | 可选 | 最近一次失败原因（有缓存时仍可能带此字段） |
| `error` | string | 仅失败 | 无可用结果时出现；HTTP 状态多为 `503` / `500` |

磁盘文件 `outputs/latest/detection_results.json` 与业务字段一致，**不含** `generated_at` / `age_seconds` / `trigger` 等 HTTP 元数据。

> **`/detect` vs `/stream`**  
> `/detect` 会写入 `/latest` 缓存与 `history.jsonl`，但**不会**向 `/stream` 推送；SSE 只推送定时任务结果，避免按钮检测与自动推送重复打扰。

### 2.2 `detectedFoods[]` 每项

| 字段 | 类型 | 必有 | 说明 |
|------|------|------|------|
| `source_image` | string | ✓ | 源图文件名，如 `"latest.jpg"` |
| `label` | string | ✓ | 食物中文名（展示用），如 `"荔枝"`、`"去壳虾仁"` |
| `type_probability` | number | ✓ | **检测置信度** 0～1（GroundingDINO），表示「框内是该类食物」的把握 |
| `spoilageScore` | number\|null | ✓ | **腐败/不可食用评分** 0～1，越大越不宜食用；**1 = 不可食用阈值**（不是「完全腐烂」，达到不可食用后不再升高）。荔枝为 DINOv2 多任务；虾仁为 EfficientNet-B0 多任务；其它食物为 VL 评估；仅失败占位时为 `null` |
| `spoilageLevel` | string | ✓ | 枚举：`"fresh"` \| `"spoiling"` \| `"spoiled"` |
| `boundingBox` | number[4] | ✓ | `[x, y, width, height]`，单位像素，原点在图像左上角 |
| `message` | string | ✓ | 该条目说明文案 |
| `producedGases` | string[] | ✓ | 关联气体名列表（目前多为模板占位，**未接传感器融合**） |

> **注意（破坏性变更）**  
> 旧字段名 `probability` 已更名为 `type_probability`。  
> 腐败程度请读 `spoilageScore` + `spoilageLevel`，不要用 `type_probability`。

### 2.3 `spoilageLevel` 与 `spoilageScore` 关系

| spoilageLevel | 含义 | 典型分数区间（参考） |
|---------------|------|----------------------|
| `fresh` | 新鲜 / 前期 | 较低（因品类阈值不同，以服务端为准） |
| `spoiling` | 转差 / 中期 | 中等 |
| `spoiled` | 不宜食用 / 后期 | 较高，可接近或等于 1 |

- **荔枝**：`spoilageScore` / `spoilageLevel` / `message` / `producedGases` 由 DINOv2 多任务模型给出。  
- **虾仁**：由 EfficientNet-B0 多任务模型给出（`rating_id` 0/1/2 → fresh/spoiling/spoiled）。  
- **其它食物**：对每个检测裁剪图调用 VL。  
- VL / 专用模型失败时回退占位：`spoilageScore` 为 `null`，其余字段随机模板。

`isSpoiled === true` 当且仅当存在至少一项 `spoilageLevel === "spoiled"`（`spoiling` 不算 spoiled）。

---

## 3. 完整示例

```json
{
  "isSpoiled": false,
  "message": "Detection Complete. All items analyzed are currently fresh.",
  "detectedFoods": [
    {
      "source_image": "demo6.jpg",
      "label": "去壳虾仁",
      "type_probability": 0.759,
      "spoilageScore": 0.3037,
      "spoilageLevel": "fresh",
      "boundingBox": [543, 399, 4146, 3127],
      "message": "Shrimp fresh (score 0.30).",
      "producedGases": ["Ethylene"]
    }
  ],
  "tagger": "vl",
  "expectedCounts": {
    "peeled shrimp": 1
  },
  "generated_at": "2026-07-21T15:10:00+08:00",
  "age_seconds": 42,
  "next_run_in_seconds": 7158
}
```

多目标时 `detectedFoods` 会有多项（例如一盘 9 颗荔枝各一条），各自独立 `boundingBox` / `spoilageScore`。

---

## 4. `/health` 响应示例

```json
{
  "status": "ok",
  "device": "cpu",
  "interval_seconds": 7200,
  "has_result": true,
  "generated_at": "2026-07-21T15:10:00+08:00",
  "age_seconds": 120,
  "next_run_in_seconds": 7080,
  "last_error": null,
  "tagger": "auto"
}
```

App 可用 `has_result`、`age_seconds` 提示「结果是否过旧」。

---

## 5. `/history` 说明

- `GET /history`：全部历史，时间升序  
- `GET /history?limit=10`：最近 10 条  
- `GET /history?since=2026-07-21T00:00:00+08:00`：该时间之后  

每条大致为：

```json
{
  "timestamp": "2026-07-21T15:10:00+08:00",
  "trigger": "scheduled",
  "isSpoiled": false,
  "message": "...",
  "detectedFoods": [ /* 与 /detect 中单项相同结构 */ ]
}
```

`trigger`：`"scheduled"`（定时）或 `"forced"`（`?force=1`）。

---

## 6. 错误处理建议

| HTTP | 场景 | 处理 |
|------|------|------|
| 200 | 正常 | 解析 JSON 展示 |
| 503 | 尚无缓存（冷启动首跑未完成） | 提示稍后重试，可轮询 `/health` |
| 500 | `force=1` 推理失败 | 展示 `error`；可继续用上次成功结果（若有） |

无食物时：`detectedFoods: []`，`isSpoiled: false`，`message` 含 “No food items detected”。

---

## 7. App 对接 checklist

1. 读取 `detectedFoods[].label` 做列表标题  
2. 用 `boundingBox` 画框（注意坐标系：左上原点，`[x,y,w,h]`）  
3. 用 `spoilageLevel` 上色 / 图标；用 `spoilageScore` 显示具体分数（注意判空）  
4. 用 `type_probability` 仅表示「识别准不准」，**不是**新鲜度  
5. 顶栏可用 `isSpoiled` + 顶层 `message`  
6. 展示 `generated_at` / `age_seconds`，避免用户误以为是实时帧  
7. 字段名已变更：`probability` → `type_probability`，新增 `spoilageScore`

---

## 8. 本地联调

```text
# 启动算法服务（示例）
python detect_server.py --cpu-only --host 0.0.0.0 --port 4100

# 拉结果
GET http://127.0.0.1:4100/detect
GET http://127.0.0.1:4100/health
```

离线文件对照：`outputs/latest/detection_results.json`（或 CLI 的 `--output-dir`）。
相机输入默认：`collect/data/latest.jpg`。
