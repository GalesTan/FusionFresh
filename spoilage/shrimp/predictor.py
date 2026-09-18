"""可复用的虾仁腐败评分器（EfficientNet-B0 多任务：score + 三分类）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

from patch_sampling import imread_unicode

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
NN_CKPT = ASSETS_DIR / "nn_spoilage_head.pt"

MODEL_VERSION = "efficientnet_b0_mthead_shrimp_v2"
INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

STAGE_CN = {0: "新鲜", 1: "中期", 2: "腐败"}
# 兼容旧 App 映射：uncertain ≈ 中期 / spoiling
STAGE_EN = {0: "fresh", 1: "uncertain", 2: "spoiled"}
STAGE_MSG = {
    "fresh": "虾仁新鲜，可放心食用（评分 {score:.2f}）。",
    "uncertain": "虾仁处于中期，建议尽快食用（评分 {score:.2f}）。",
    "spoiled": "虾仁已腐败，不宜食用（评分 {score:.2f}）。",
}


class ShrimpMultiTaskNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        n_stages: int = 3,
    ) -> None:
        super().__init__()
        backbone = efficientnet_b0(weights=None)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.reg_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.cls_head = nn.Linear(hidden_dim, n_stages)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        h = self.shared(feat)
        score = self.reg_head(h).squeeze(-1)
        logits = self.cls_head(h)
        return score, logits


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def center_square_bgr(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(h, w)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    return img[y0 : y0 + side, x0 : x0 + side]


class ShrimpPredictor:
    """虾仁推理器：中心方裁 → EfficientNet-B0 → score + 三分类。"""

    def __init__(
        self,
        device: str | None = None,
        score_mode: str = "cls",  # 保留参数以兼容旧调用，已忽略
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.score_mode = score_mode
        self.transform = build_transform()

        if not NN_CKPT.exists():
            raise FileNotFoundError(f"缺少虾仁 NN 权重: {NN_CKPT}")
        ckpt = torch.load(NN_CKPT, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})
        self._thresholds = {
            "early_mid": float(ckpt["thresholds"]["early_mid"]),
            "mid_late": float(ckpt["thresholds"]["mid_late"]),
        }
        self._model = ShrimpMultiTaskNet(
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            dropout=float(cfg.get("dropout", 0.0)),
        ).to(self.device)
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()
        self.model_version = str(cfg.get("model_version", MODEL_VERSION))

    @torch.inference_mode()
    def predict(
        self,
        image_path: str | Path,
        *,
        patch_groups: int | None = None,  # 兼容旧参数
        score_jitter: float | None = None,  # 兼容旧参数
    ) -> dict[str, Any]:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"图片不存在: {path}")

        img = imread_unicode(path)
        if img is None:
            raise ValueError(f"无法读取图片: {path}")

        square = center_square_bgr(img)
        rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        score_t, logits = self._model(tensor)
        score = float(score_t.item())
        rating_id = int(logits.argmax(dim=1).item())
        level = STAGE_EN[rating_id]
        rating = STAGE_CN[rating_id]
        message = STAGE_MSG[level].format(score=score)

        return {
            "success": True,
            "category": "shrimp",
            "score": round(score, 4),
            "rating": rating,
            "rating_id": rating_id,
            "spoilage_level": level,  # fresh | uncertain | spoiled（兼容 food_pipeline）
            "spoilage_stage": rating,
            "message": message,
            "group_scores": [round(score, 4)],
            "score_std": 0.0,
            "stage_thresholds": {
                "early_mid": self._thresholds["early_mid"],
                "mid_late": self._thresholds["mid_late"],
                # 旧字段名兼容
                "fresh_uncertain": self._thresholds["early_mid"],
                "uncertain_spoiled": self._thresholds["mid_late"],
            },
            "score_mode": "multitask_cls",
            "model_version": self.model_version,
        }


_default_predictor: ShrimpPredictor | None = None


def predict_shrimp(
    image_path: str | Path,
    *,
    device: str | None = None,
    score_mode: str = "cls",
) -> dict[str, Any]:
    """对单张虾仁图片进行腐败评分与三分类。"""
    global _default_predictor
    want = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if _default_predictor is None or str(_default_predictor.device) != str(want):
        _default_predictor = ShrimpPredictor(device=want, score_mode=score_mode)
    return _default_predictor.predict(image_path)
