"""Fusion model inference. Combines the per-cycle feature dict into confidence + engagement."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from app.models.face.architecture import EMOTION_CLASSES
from app.models.voice.architecture import VOICE_CLASSES
from app.models.fusion.model import FUSION_INPUT_DIM, FusionNet


def build_feature_vector(
    face_scores: dict,
    voice_scores: dict,
    speaking_rate: float,
    energy_level: float,
    filler_probability: float,
    filler_rate: float,
    hedge_rate: float,
    language_confidence: float,
) -> np.ndarray:
    """Pack the 21-dim feature vector in the canonical order. All values clamped to [0, 1]."""
    face_vec = np.array(
        [face_scores.get(c, 0.0) for c in EMOTION_CLASSES], dtype=np.float32,
    )
    voice_vec = np.array(
        [voice_scores.get(c, 0.0) for c in VOICE_CLASSES], dtype=np.float32,
    )
    extras = np.array([
        np.clip(speaking_rate / 6.0, 0, 1),
        np.clip(energy_level / 0.2, 0, 1),
        np.clip(filler_probability, 0, 1),
        np.clip(filler_rate / 0.2, 0, 1),
        np.clip(hedge_rate / 0.2, 0, 1),
        np.clip(language_confidence / 100.0, 0, 1),
    ], dtype=np.float32)
    feat = np.concatenate([face_vec, voice_vec, extras])
    if feat.shape[0] != FUSION_INPUT_DIM:
        if feat.shape[0] < FUSION_INPUT_DIM:
            feat = np.pad(feat, (0, FUSION_INPUT_DIM - feat.shape[0]))
        else:
            feat = feat[:FUSION_INPUT_DIM]
    return feat


class FusionPredictor:
    """Loads the trained fusion model once. Call predict(features_dict) per cycle."""

    def __init__(self, model_path: str | Path, device: Optional[torch.device] = None):
        self.device = device or self._default_device()
        state = torch.load(str(model_path), map_location=self.device, weights_only=False)
        in_dim = state.get("in_dim", FUSION_INPUT_DIM)
        self.model = FusionNet(in_dim=in_dim).to(self.device)
        self.model.load_state_dict(state["model_state"])
        self.model.eval()

    @staticmethod
    def _default_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @torch.no_grad()
    def predict(self, **kwargs) -> tuple[float, float]:
        feat = build_feature_vector(**kwargs)
        x = torch.from_numpy(feat).unsqueeze(0).to(self.device)
        c, e = self.model(x)
        return float(c.item()), float(e.item())

    @torch.no_grad()
    def predict_vec(self, feat: np.ndarray) -> tuple[float, float]:
        x = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(self.device)
        c, e = self.model(x)
        return float(c.item()), float(e.item())
