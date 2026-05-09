"""Face emotion prediction. MediaPipe face detection plus our CNN."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.models.face.architecture import (
    EMOTION_CLASSES,
    INTERVIEW_SIGNAL_MAP,
    FaceEmotionCNN,
)


_NEUTRAL_RESULT = {
    "emotion": "neutral",
    "confidence": 0.0,
    "signal": "engaged",
    "scores": {c: 0.0 for c in EMOTION_CLASSES},
    "face_detected": False,
}


class FacePredictor:
    """Loads the trained face CNN once. Run detect_face then predict_emotion per frame."""

    def __init__(self, model_path: str | Path, device: Optional[torch.device] = None):
        self.device = device or self._default_device()
        self.model = FaceEmotionCNN(num_classes=len(EMOTION_CLASSES)).to(self.device)
        state = torch.load(str(model_path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state"])
        self.model.eval()
        self._init_face_detector()

    @staticmethod
    def _default_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _init_face_detector(self):
        try:
            import mediapipe as mp
            self._mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5,
            )
        except Exception:
            self._mp_face = None
            self._haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    def detect_face(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return a 48x48 grayscale face crop, or None if no face is found."""
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        h, w = frame_bgr.shape[:2]

        if self._mp_face is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self._mp_face.process(rgb)
            if not res.detections:
                return None
            best = max(res.detections, key=lambda d: d.score[0] if d.score else 0)
            box = best.location_data.relative_bounding_box
            x1 = max(0, int(box.xmin * w))
            y1 = max(0, int(box.ymin * h))
            x2 = min(w, int((box.xmin + box.width) * w))
            y2 = min(h, int((box.ymin + box.height) * h))
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self._haar.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
            if len(faces) == 0:
                return None
            x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            x1, y1, x2, y2 = x, y, x + fw, y + fh

        if x2 - x1 < 20 or y2 - y1 < 20:
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)

    def _to_tensor(self, face_48: np.ndarray) -> torch.Tensor:
        arr = face_48.astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        return t.to(self.device)

    @torch.no_grad()
    def predict_emotion(self, face_48: Optional[np.ndarray]) -> dict:
        if face_48 is None:
            return dict(_NEUTRAL_RESULT)
        x = self._to_tensor(face_48)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        emotion = EMOTION_CLASSES[idx]
        return {
            "emotion": emotion,
            "confidence": float(probs[idx]),
            "signal": INTERVIEW_SIGNAL_MAP[emotion],
            "scores": {EMOTION_CLASSES[i]: float(probs[i]) for i in range(len(EMOTION_CLASSES))},
            "face_detected": True,
        }

    def predict_frame(self, frame_bgr: np.ndarray) -> dict:
        """Convenience: detect a face in a BGR frame and predict in one call."""
        face = self.detect_face(frame_bgr)
        return self.predict_emotion(face)
