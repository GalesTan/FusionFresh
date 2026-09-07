# vendor

第三方推理代码与权重，仅保留当前工作流需要的部分。

| 目录 | 用途 | 权重 |
|------|------|------|
| `recognize-anything/` | RAM++ 图像打标签（VL 回退） | `pretrained/ram_plus_swin_large_14m.pth` |
| `GroundingDINO/` | 开放词汇目标检测 | `weights/groundingdino_swint_ogc.pth` |

由 `food_pipeline.py` 通过 `sys.path` 导入，未改内部实现。

许可证文件保留在各自目录的 `LICENSE`。
