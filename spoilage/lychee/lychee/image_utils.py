"""图像读写与荔枝前景掩膜。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

JPEG_QUALITY = 95


def load_image(path: Path) -> np.ndarray:
    """读取图片并应用 EXIF 方向信息，返回 BGR 数组。"""
    pil_img = ImageOps.exif_transpose(Image.open(path))
    rgb = np.array(pil_img)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def imread_unicode(path: Path) -> np.ndarray:
    return load_image(path)


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError(f"无法写入图片: {path}")
    buf.tofile(str(path))


def build_lychee_mask(img: np.ndarray) -> np.ndarray:
    """构建荔枝前景掩膜，兼容红/绿/深色果实。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 25, 30]), np.array([25, 255, 255])),
        cv2.inRange(hsv, np.array([155, 25, 30]), np.array([180, 255, 255])),
    )
    green = cv2.inRange(hsv, np.array([25, 20, 30]), np.array([50, 255, 255]))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lightness, a_channel, _ = cv2.split(lab)
    non_white = ~((hsv[:, :, 1] < 55) & (hsv[:, :, 2] > 175))
    dark_fruit = (lightness < 135) & non_white & (a_channel > 130) & (hsv[:, :, 1] > 18)

    mask = ((red > 0) | (green > 0) | dark_fruit).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )
    return mask
