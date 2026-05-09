"""Voice emotion prediction. Operates on a 3-second mono audio buffer."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from app.models.voice.architecture import (
    SAMPLE_RATE,
    STAT_FEATURES,
    VOICE_CLASSES,
    VOICE_SIGNAL_MAP,
    VoiceEmotionCNN,
)
from app.models.voice.voice_loader import extract_features


SILENCE_RMS = 0.005

FILLER_FORMANT_LOW = 200
FILLER_FORMANT_HIGH = 700


_QUIET_RESULT = {
    "emotion": "neutral",
    "confidence": 0.0,
    "signal": "confident",
    "scores": {c: 0.0 for c in VOICE_CLASSES},
    "speaking_rate": 0.0,
    "energy_level": 0.0,
    "filler_probability": 0.0,
    "speaking": False,
}


class VoicePredictor:
    """Loads the trained voice CNN once. Call predict_voice(audio, sr) per cycle."""

    def __init__(self, model_path: str | Path, device: Optional[torch.device] = None):
        self.device = device or self._default_device()
        state = torch.load(str(model_path), map_location=self.device, weights_only=False)
        stat_features = state.get("stat_features", STAT_FEATURES)
        self.model = VoiceEmotionCNN(num_classes=len(VOICE_CLASSES),
                                     stat_features=stat_features).to(self.device)
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
    def predict_voice(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
        if audio is None or audio.size == 0:
            return dict(_QUIET_RESULT)

        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-12))
        if rms < SILENCE_RMS:
            out = dict(_QUIET_RESULT)
            out["energy_level"] = rms
            return out

        mel, stats, descriptors = extract_features(audio, sr)
        mel_t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(self.device)
        stats_t = torch.from_numpy(stats).unsqueeze(0).to(self.device)
        logits = self.model(mel_t, stats_t)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        emotion = VOICE_CLASSES[idx]

        # Filler heuristic. Sustained low-energy mid-frequency centroid suggests "uh" or "um".
        # Real filler detection happens server-side from transcribed text. This is a soft prior.
        centroid_norm = stats[80] if len(stats) > 80 else 0
        filler_proxy = float(np.clip(1.0 - probs[VOICE_CLASSES.index("happy")] - probs[VOICE_CLASSES.index("calm")], 0, 1))
        filler_proxy *= float(np.clip(1.0 - descriptors["energy_level"] / 0.05, 0, 1))

        return {
            "emotion": emotion,
            "confidence": float(probs[idx]),
            "signal": VOICE_SIGNAL_MAP[emotion],
            "scores": {VOICE_CLASSES[i]: float(probs[i]) for i in range(len(VOICE_CLASSES))},
            "speaking_rate": descriptors["speaking_rate"],
            "energy_level": descriptors["energy_level"],
            "filler_probability": filler_proxy,
            "speaking": True,
        }
