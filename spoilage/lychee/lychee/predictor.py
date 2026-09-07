"""荔枝单果腐败评分推理（DINOv2 + 多任务头：score + 三分类）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from lychee.color_calibrate import calibrate_image, load_reference_white
from lychee.image_utils import load_image
from lychee.white_background import process_image

RELEASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = RELEASE_DIR / "models" / "lychee"
NN_CKPT = MODEL_DIR / "nn_spoilage_head.pt"
CALIBRATION_META = MODEL_DIR / "calibration_meta.json"

MODEL_NAME = "dinov2_vitb14"
MODEL_VERSION = "dinov2_vitb14_mthead_propagate_v1"
INPUT_SIZE = 518

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
STAGE_LABELS = {0: "前期", 1: "中期", 2: "后期"}


class SpoilageMultiTaskHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 512,
        dropout: float = 0.25,
        n_stages: int = 3,
    ) -> None:
        super().__init__()
        mid = max(hidden_dim // 2, 128)
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.reg_head = nn.Sequential(nn.Linear(mid, 1), nn.Sigmoid())
        self.cls_head = nn.Linear(mid, n_stages)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        score = self.reg_head(h).squeeze(-1)
        logits = self.cls_head(h)
        return score, logits


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (INPUT_SIZE, INPUT_SIZE),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class LycheePredictor:
    """可复用的荔枝推理器：DINOv2 特征 + 多任务头。"""

    def __init__(self, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.transform = build_transform()
        self._dino_model: nn.Module | None = None
        self._reference_white = load_reference_white(CALIBRATION_META)

        if not NN_CKPT.exists():
            raise FileNotFoundError(f"缺少 NN 头权重: {NN_CKPT}")
        ckpt = torch.load(NN_CKPT, map_location=self.device, weights_only=False)
        cfg = ckpt["config"]
        self._thresholds = {
            "early_mid": float(ckpt["thresholds"]["early_mid"]),
            "mid_late": float(ckpt["thresholds"]["mid_late"]),
        }
        self._head = SpoilageMultiTaskHead(
            in_dim=int(cfg["in_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
            dropout=float(cfg.get("dropout", 0.0)),
        ).to(self.device)
        self._head.load_state_dict(ckpt["model_state"])
        self._head.eval()
        self.model_version = str(cfg.get("model_version", MODEL_VERSION))

    def _load_dino(self) -> nn.Module:
        if self._dino_model is None:
            local_hub = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
            if local_hub.exists():
                model = torch.hub.load(
                    str(local_hub), MODEL_NAME, source="local", pretrained=True
                )
            else:
                model = torch.hub.load(
                    "facebookresearch/dinov2",
                    MODEL_NAME,
                    pretrained=True,
                    trust_repo=True,
                )
            model.eval()
            self._dino_model = model.to(self.device)
        return self._dino_model

    def preprocess(
        self,
        image_path: Path,
        *,
        calibrate: bool = False,
        skip_standardize: bool = False,
    ) -> np.ndarray:
        img = load_image(image_path)
        if calibrate:
            img = calibrate_image(img, self._reference_white)
        if skip_standardize:
            return img
        return process_image(img)

    @torch.inference_mode()
    def _extract_feature(self, bgr_image: np.ndarray) -> np.ndarray:
        rgb = bgr_image[:, :, ::-1]
        pil_img = Image.fromarray(rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        model = self._load_dino()
        feature = model(tensor).cpu().numpy().astype(np.float32)[0]
        return feature

    @torch.inference_mode()
    def _predict_from_feature(
        self, dino_feature: np.ndarray
    ) -> tuple[float, int, str, dict[str, float]]:
        x = torch.from_numpy(dino_feature.reshape(1, -1)).to(self.device)
        score_t, logits = self._head(x)
        score = float(score_t.item())
        stage_id = int(logits.argmax(dim=1).item())
        return score, stage_id, STAGE_LABELS[stage_id], dict(self._thresholds)

    def predict(
        self,
        image_path: str | Path,
        *,
        calibrate: bool = False,
        skip_standardize: bool = False,
    ) -> dict[str, Any]:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"图片不存在: {path}")

        processed = self.preprocess(
            path, calibrate=calibrate, skip_standardize=skip_standardize
        )
        dino_feature = self._extract_feature(processed)
        score, rating_id, rating, stage_thresholds = self._predict_from_feature(
            dino_feature
        )

        return {
            "success": True,
            "category": "lychee",
            "score": round(score, 4),
            "rating": rating,
            "rating_id": rating_id,
            "stage_thresholds": stage_thresholds,
            "model_version": self.model_version,
        }


_default_predictor: LycheePredictor | None = None


def predict_lychee(
    image_path: str | Path,
    *,
    calibrate: bool = False,
    skip_standardize: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    """对单张荔枝图片进行腐败评分与三分类。"""
    global _default_predictor
    if _default_predictor is None or (
        device is not None and str(_default_predictor.device) != device
    ):
        _default_predictor = LycheePredictor(device=device)
    return _default_predictor.predict(
        image_path,
        calibrate=calibrate,
        skip_standardize=skip_standardize,
    )
