# 虾仁腐败评分

从原项目 `虾仁_mix/` 同步的 **EfficientNet-B0 多任务** 推理包（score + 三分类）。

- 版本：`efficientnet_b0_mthead_shrimp_v2`（两批数据：`虾仁_new` + `虾仁0908`）
- 输出：`score ∈ [0,1]` + 三分类（新鲜 / 中期 / 腐败）
- App 映射：`rating_id` 0/1/2 → `fresh` / `spoiling` / `spoiled`（中期在模型里名为 `uncertain`）

```
shrimp/
├── predictor.py         # 主推理入口
├── patch_sampling.py    # imread_unicode
├── gas_fusion.py        # 含虾仁场景的气体规则（强制仅 H2S/NH3 且全虾仁）
└── assets/nn_spoilage_head.pt
```

服务端通过 `food_pipeline.get_shrimp_predictor()` 加载。改权重后需重启 `detect_server.py`。

气体默认读 `collect/data/` 下 VOC / C2H5OH / H2S / NH3 的 CSV。
