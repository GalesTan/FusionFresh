# 虾仁腐败评分

从原项目 `shrimp_infer/` 同步的 **EfficientNet-B0 多任务** 推理包（score + 三分类）。

- 版本：`efficientnet_b0_mthead_shrimp_v1`
- 输出：`score ∈ [0,1]` + 三分类（新鲜 / 中期 / 腐败）
- App 映射：`rating_id` 0/1/2 → `fresh` / `spoiling` / `spoiled`（中期在模型里名为 `uncertain`）

```
shrimp/
├── predictor.py         # 主推理入口
├── patch_sampling.py    # imread_unicode
├── gas_fusion.py        # 仅虾仁场景的气体覆盖规则
└── assets/nn_spoilage_head.pt
```

服务端通过 `food_pipeline.get_shrimp_predictor()` 加载。改权重后需重启 `detect_server.py`。

气体默认读 `collect/data/` 下 VOC / C2H5OH / H2S / NH3 的 CSV。
