"""腐败评分模块。

当前工作流：
- lychee: DINOv2 + 多任务头（detect_release 同步）
- shrimp: EfficientNet-B0 + 多任务头，可选气体融合

新增品类时见 docs/extending.md。
"""
