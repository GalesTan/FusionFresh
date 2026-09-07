"""单果白底标准化。"""

from __future__ import annotations

import cv2
import numpy as np

from lychee.image_utils import build_lychee_mask

OUTPUT_SIZE = 512
FRUIT_FRAC = 0.68
MASK_DILATE = 2


def largest_component(mask: np.ndarray) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return mask
    best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == best_label).astype(np.uint8)) * 255


def remove_background_pixels(mask: np.ndarray, img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    background_like = (hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 145)
    cleaned = mask.copy()
    cleaned[background_like] = 0
    return cleaned


def grabcut_refine(img: np.ndarray, init_mask: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    if cv2.countNonZero(init_mask) == 0:
        return init_mask

    dist = cv2.distanceTransform(init_mask, cv2.DIST_L2, 5)
    core = dist > max(12.0, float(dist.max()) * 0.35)
    gc_mask[init_mask > 0] = cv2.GC_PR_FGD
    gc_mask[core] = cv2.GC_FGD

    margin = max(8, min(h, w) // 40)
    gc_mask[:margin, :] = cv2.GC_BGD
    gc_mask[h - margin :, :] = cv2.GC_BGD
    gc_mask[:, :margin] = cv2.GC_BGD
    gc_mask[:, w - margin :] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, gc_mask, None, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)
    refined = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    return largest_component(refined)


def refine_fruit_mask(mask: np.ndarray, img: np.ndarray) -> np.ndarray:
    fruit = largest_component(remove_background_pixels(mask, img))
    if cv2.countNonZero(fruit) == 0:
        return fruit
    fruit = grabcut_refine(img, fruit)
    if cv2.countNonZero(fruit) == 0:
        return fruit
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fruit = cv2.morphologyEx(fruit, cv2.MORPH_CLOSE, kernel)
    if MASK_DILATE > 0:
        fruit = cv2.dilate(
            fruit,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=MASK_DILATE,
        )
    return fruit


def apply_white_background(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.full_like(img, 255)
    output[mask > 0] = img[mask > 0]
    return output


def center_and_scale_fruit(
    img: np.ndarray,
    mask: np.ndarray,
    canvas_size: int = OUTPUT_SIZE,
    fruit_frac: float = FRUIT_FRAC,
) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return img

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    roi = img[y0 : y1 + 1, x0 : x1 + 1].copy()
    roi_mask = mask[y0 : y1 + 1, x0 : x1 + 1]

    roi_white = np.full_like(roi, 255)
    roi_white[roi_mask > 0] = roi[roi_mask > 0]

    rh, rw = roi_white.shape[:2]
    target_long = max(1, int(canvas_size * fruit_frac))
    scale = target_long / max(rh, rw)
    new_w = max(1, int(rw * scale))
    new_h = max(1, int(rh * scale))
    resized = cv2.resize(roi_white, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    x_off = (canvas_size - new_w) // 2
    y_off = (canvas_size - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def process_image(img: np.ndarray) -> np.ndarray:
    mask = refine_fruit_mask(build_lychee_mask(img), img)
    if cv2.countNonZero(mask) == 0:
        raise ValueError("未检测到荔枝区域")
    white = apply_white_background(img, mask)
    size = max(img.shape[0], img.shape[1])
    if img.shape[0] != img.shape[1]:
        size = OUTPUT_SIZE
    return center_and_scale_fruit(white, mask, canvas_size=size)
