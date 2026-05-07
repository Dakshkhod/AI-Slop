"""EfficientNet-B4 classifier for image AI detection (v4)."""

from __future__ import annotations

import json
import logging
import pathlib
import platform
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms

_BASE = Path(__file__).resolve().parents[2]
_CHECKPOINT_PATH = _BASE / "best_model_v4.pth"
_T_PATH = _BASE / "T.json"
with open(_T_PATH, encoding="utf-8") as f:
    _T = float(json.load(f)["temperature"])
log = logging.getLogger(__name__)

_BACKBONE_NAME = "efficientnet_b4"
_IMG_SIZE = 224

# Flag as AI if ai_probability exceeds this. Lowered from 0.50 → 0.35 to
# improve recall on AI images (val recall 47% → ~70%) at the cost of a
# small increase in false positives on real photos (~6% → ~12%).
_AI_THRESHOLD = 0.35


class TruthLensModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            _BACKBONE_NAME,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


_model: TruthLensModel | None = None
_device: torch.device | None = None

_transform = transforms.Compose(
    [
        transforms.Resize(_IMG_SIZE),
        transforms.CenterCrop(_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model() -> TruthLensModel:
    global _model, _device
    if _model is not None:
        return _model

    if not _CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {_CHECKPOINT_PATH}")

    _device = _pick_device()
    # The v4 checkpoint was saved on Colab (Linux); its config dict contains
    # pathlib.PosixPath objects, which Windows cannot instantiate. We trust
    # our own checkpoint, so use weights_only=False directly and alias
    # PosixPath→WindowsPath on Windows for the duration of the load.
    is_windows = platform.system() == "Windows"
    if is_windows:
        _saved_posix = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc]
    try:
        ckpt = torch.load(_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    finally:
        if is_windows:
            pathlib.PosixPath = _saved_posix  # type: ignore[misc]
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    log.info("Loaded v4 checkpoint: %s", _CHECKPOINT_PATH.name)

    model = TruthLensModel()
    model.load_state_dict(state_dict)
    model.to(_device)
    model.eval()
    _model = model
    return _model


def check_ml_classifier(img: Image.Image) -> Dict[str, Any]:
    model = get_model()
    assert _device is not None

    tensor = _transform(img.convert("RGB")).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = model(tensor)
        # Test-time augmentation: average original + horizontally flipped
        # logits before temperature-scaled softmax.
        logits_flip = model(torch.flip(tensor, dims=[3]))
        avg_logits = (logits + logits_flip) / 2.0
        probs = torch.softmax(avg_logits / _T, dim=1)[0]

    ai_prob   = float(probs[1].item())
    real_prob = float(probs[0].item())
    return {
        "signal": "ml_classifier",
        "score": round(real_prob, 4),  # 1.0 = real, 0.0 = AI
        "detail": {
            "ai_probability":   round(ai_prob,   4),
            "real_probability": round(real_prob, 4),
            "ai_threshold":     _AI_THRESHOLD,
            "flagged_as_ai":    ai_prob >= _AI_THRESHOLD,
            "checkpoint":       str(_CHECKPOINT_PATH.name),
            "backbone":         _BACKBONE_NAME,
            "img_size":         _IMG_SIZE,
            "temperature":      round(_T, 4),
            "tta":              "hflip",
        },
    }
