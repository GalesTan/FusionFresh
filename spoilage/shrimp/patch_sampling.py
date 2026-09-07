"""推理用图片读取。

当前虾仁模型（EfficientNet-B0 多任务）只使用 ``imread_unicode``。
旧版 DINOv2 patch 采样逻辑已从本集成包中移除。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    return img
