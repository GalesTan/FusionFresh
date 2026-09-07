"""颜色标定（可选，用于含桌面背景的全图输入）。"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from lychee.image_utils import load_image


def sample_tabletop_white(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    margin = int(min(h, w) * 0.08)
    cx0, cx1 = int(w * 0.25), int(w * 0.75)
    cy0, cy1 = int(h * 0.2), int(h * 0.8)

    edge_patches = [
        img[0:margin, 0:cx0],
        img[0:margin, cx1:w],
        img[h - margin : h, 0:cx0],
        img[h - margin : h, cx1:w],
        img[0:cy0, 0:margin],
        img[cy1:h, 0:margin],
        img[0:cy0, w - margin : w],
        img[cy1:h, w - margin : w],
    ]

    collected: list[np.ndarray] = []
    for patch in edge_patches:
        if patch.size == 0:
            continue
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 150)
        if mask.any():
            collected.append(patch[mask])

    if not collected:
        raise ValueError("未能在图片边缘找到足够的白色桌面像素，请关闭标定或使用单果裁剪图")

    pixels = np.vstack(collected)
    return pixels.mean(axis=0)


def compute_channel_gains(reference_white: np.ndarray, source_white: np.ndarray) -> np.ndarray:
    return (reference_white / np.maximum(source_white, 1.0)).astype(np.float32)


def apply_white_balance(img: np.ndarray, gains: np.ndarray) -> np.ndarray:
    corrected = img.astype(np.float32) * gains.reshape(1, 1, 3)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def calibrate_image(img: np.ndarray, reference_white: np.ndarray) -> np.ndarray:
    source_white = sample_tabletop_white(img)
    gains = compute_channel_gains(reference_white, source_white)
    return apply_white_balance(img, gains)


def load_reference_white(meta_path: Path) -> np.ndarray:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return np.array(meta["reference_white_bgr"], dtype=np.float32)
