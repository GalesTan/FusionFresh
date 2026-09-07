# 荔枝腐败评分

从原项目 `detect_release/` 同步，可独立调用，不依赖其它品类。

- 骨干：DINOv2 ViT-B/14（首次运行经 torch.hub 下载）
- 头网络：`models/lychee/nn_spoilage_head.pt`
- 评级：前期 / 中期 / 后期 → App 的 fresh / spoiling / spoiled

```powershell
python spoilage/lychee/detect.py --image path/to/lychee.jpg --category lychee
```

气体 JSON（`example.json`）目前只透传，不参与荔枝评分。

独立包用法与字段见原 `detect_release/README.md` 的约定；权重只保留当前 NN 头与白色标定元数据。
