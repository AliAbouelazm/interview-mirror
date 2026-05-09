"""
Real-time per-cycle pipeline.

Each cycle takes the latest video frame and audio buffer, runs face + voice + fusion,
and (every Nth cycle) runs Whisper transcription on a longer audio window. Latency is
measured for each component and emitted alongside the output. The cycle time is adaptive:
if total latency exceeds the target, the orchestrator slows the cycle so it does not
fall further behind.
"""
from __future__ import annotations

import base64
import io
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import cv2
import numpy as np
from PIL import Image

from app.models.face.predict import FacePredictor
from app.models.voice.architecture import SAMPLE_RATE as VOICE_SR
from app.models.voice.predict import VoicePredictor
from app.models.fusion.predict import FusionPredictor, build_feature_vector
from app.pipeline.transcription import Transcriber, analyze_text
from app.utils.logging import get_logger


_log = get_logger(__name__)


def decode_jpeg_b64(data: str) -> Optional[np.ndarray]:
    if not data:
        return None
    if "," in data:
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        arr = np.array(img)[:, :, ::-1].copy()
        return arr
    except Exception:
        return None


def decode_pcm_b64(data: str) -> Optional[np.ndarray]:
    """Base64 of float32 little-endian samples. Returns 1-D float32 array."""
    if not data:
        return None
    try:
        raw = base64.b64decode(data)
        arr = np.frombuffer(raw, dtype=np.float32).copy()
        return arr
    except Exception:
        return None


@dataclass
class AudioRingBuffer:
    """Stores enough audio for both the per-cycle voice classifier and Whisper."""
    sample_rate: int
    seconds: float
    buf: Deque[np.ndarray] = field(default_factory=deque)
    total_samples: int = 0

    @property
    def capacity(self) -> int:
        return int(self.sample_rate * self.seconds)

    def push(self, chunk: np.ndarray) -> None:
        if chunk is None or chunk.size == 0:
            return
        self.buf.append(chunk.astype(np.float32))
        self.total_samples += chunk.shape[0]
        while self.total_samples - (self.buf[0].shape[0] if self.buf else 0) >= self.capacity and len(self.buf) > 1:
            removed = self.buf.popleft()
            self.total_samples -= removed.shape[0]

    def latest(self, seconds: float) -> np.ndarray:
        n = int(seconds * self.sample_rate)
        if not self.buf:
            return np.zeros(n, dtype=np.float32)
        merged = np.concatenate(list(self.buf))
        if merged.shape[0] >= n:
            return merged[-n:]
        out = np.zeros(n, dtype=np.float32)
        out[-merged.shape[0]:] = merged
        return out


def _dominant_driver(face_signal: str, voice_signal: str, language_confidence: float, filler_count: int) -> str:
    """Pick the signal that most explains this cycle's score."""
    if filler_count >= 2 or language_confidence < 45:
        return "language"
    if face_signal in ("nervous", "tense"):
        return "face"
    if voice_signal in ("nervous", "tense"):
        return "voice"
    if face_signal == "engaged" and voice_signal == "confident":
        return "face"
    return "voice"


class RealtimeOrchestrator:
    """One instance per session. Holds models and rolling state."""

    def __init__(
        self,
        face_predictor: FacePredictor,
        voice_predictor: VoicePredictor,
        fusion_predictor: FusionPredictor,
        transcriber: Transcriber,
        target_cycle_ms: int = 500,
        max_cycle_ms: int = 1000,
        transcription_interval_cycles: int = 2,
        whisper_window_seconds: float = 5.0,
    ):
        self.face = face_predictor
        self.voice = voice_predictor
        self.fusion = fusion_predictor
        self.transcriber = transcriber

        self.target_cycle_ms = target_cycle_ms
        self.max_cycle_ms = max_cycle_ms
        self.transcription_interval = transcription_interval_cycles
        self.whisper_window_seconds = whisper_window_seconds

        self.audio_ring = AudioRingBuffer(sample_rate=VOICE_SR, seconds=10.0)
        self.cycle_count = 0

        self.latest_transcript = ""
        self.session_filler_total = 0
        self.session_hedge_total = 0
        self.session_word_total = 0
        self.last_transcript_at = 0.0
        self.actual_cycle_ms = float(target_cycle_ms)
        self.in_flight_transcription = False
        self.latest_face_metrics: dict | None = None
        self.last_speech_at = time.time()
        self.session_start = time.time()

    def update_face_metrics(self, metrics: dict) -> None:
        self.latest_face_metrics = metrics

    def push_audio(self, pcm: np.ndarray) -> None:
        self.audio_ring.push(pcm)

    def _maybe_run_transcription(self, now: float) -> dict:
        """Return transcription stats. Skip if not at the right cycle or already running."""
        if self.cycle_count % self.transcription_interval != 0:
            return {}
        if self.in_flight_transcription:
            return {}
        audio = self.audio_ring.latest(self.whisper_window_seconds)
        if audio is None or float(np.sqrt(np.mean(audio ** 2) + 1e-12)) < 0.005:
            return {"text": "", "latency_ms": 0.0}
        try:
            self.in_flight_transcription = True
            tr = self.transcriber.transcribe_chunk(audio, sample_rate=VOICE_SR)
        except Exception as e:
            _log.warning(f"transcription failed: {e}")
            tr = {"text": "", "latency_ms": 0.0}
        finally:
            self.in_flight_transcription = False

        text = tr.get("text", "")
        if text:
            new_words = text.split()
            existing_tail = " ".join(self.latest_transcript.split()[-len(new_words):]) if self.latest_transcript else ""
            if not new_words or " ".join(new_words) != existing_tail:
                self.latest_transcript = (self.latest_transcript + " " + text).strip()
        analysis = analyze_text(text)
        self.session_filler_total += analysis["filler_count"]
        self.session_hedge_total += analysis["hedge_count"]
        self.session_word_total += analysis["word_count"]
        self.last_transcript_at = now
        return {
            "text": text,
            "latency_ms": tr.get("latency_ms", 0.0),
            "filler_count": analysis["filler_count"],
            "hedge_count": analysis["hedge_count"],
            "language_confidence": analysis["language_confidence"],
            "flagged_phrases": analysis["flagged_phrases"],
            "word_count": analysis["word_count"],
        }

    def cycle(self, frame_bgr: Optional[np.ndarray]) -> dict:
        """Run one pipeline cycle and return the WS payload as a plain dict."""
        cycle_start = time.perf_counter()
        self.cycle_count += 1

        t0 = time.perf_counter()
        face_result = self.face.predict_frame(frame_bgr) if frame_bgr is not None else self.face.predict_emotion(None)
        face_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        voice_audio = self.audio_ring.latest(3.0)
        voice_result = self.voice.predict_voice(voice_audio, sr=VOICE_SR)
        voice_ms = (time.perf_counter() - t0) * 1000.0

        transcription_info = self._maybe_run_transcription(cycle_start)
        transcription_ms = transcription_info.get("latency_ms", 0.0)

        session_words = max(self.session_word_total, 1)
        filler_rate = self.session_filler_total / session_words
        hedge_rate = self.session_hedge_total / session_words
        lang_conf_session = max(0.0, 100.0 - 200.0 * filler_rate - 150.0 * hedge_rate)

        t0 = time.perf_counter()
        feat = build_feature_vector(
            face_scores=face_result["scores"],
            voice_scores=voice_result["scores"],
            speaking_rate=voice_result.get("speaking_rate", 0.0),
            energy_level=voice_result.get("energy_level", 0.0),
            filler_probability=voice_result.get("filler_probability", 0.0),
            filler_rate=filler_rate,
            hedge_rate=hedge_rate,
            language_confidence=lang_conf_session,
        )
        confidence, engagement = self.fusion.predict_vec(feat)
        fusion_ms = (time.perf_counter() - t0) * 1000.0

        # Hard post-fusion penalties. These reflect objective bad behaviour the
        # fusion model alone is too soft about.
        if voice_result.get("speaking", False):
            self.last_speech_at = cycle_start
        silence_seconds = cycle_start - self.last_speech_at
        in_warmup = (cycle_start - self.session_start) < 5.0

        confidence_penalty = 0.0
        engagement_penalty = 0.0

        if face_result["signal"] in ("nervous", "tense"):
            confidence_penalty += 8.0
            engagement_penalty += 5.0
        if voice_result["signal"] in ("nervous", "tense"):
            confidence_penalty += 8.0
            engagement_penalty += 5.0
        if filler_rate > 0.10:
            confidence_penalty += min(15.0, (filler_rate - 0.10) * 100)
        if hedge_rate > 0.06:
            confidence_penalty += min(10.0, (hedge_rate - 0.06) * 100)
        if lang_conf_session < 50:
            confidence_penalty += (50 - lang_conf_session) * 0.4
        if not in_warmup and silence_seconds > 4.0:
            confidence_penalty += min(15.0, (silence_seconds - 4.0) * 2.0)
            engagement_penalty += min(20.0, (silence_seconds - 4.0) * 3.0)

        fm = self.latest_face_metrics or {}
        looking = fm.get("looking_at_camera")
        eye_open = fm.get("eye_openness")
        smile = fm.get("smile")
        head_yaw = abs(fm.get("head_yaw", 0.0))
        head_pitch = abs(fm.get("head_pitch", 0.0))
        face_detected = face_result.get("face_detected", False)

        if not face_detected:
            confidence_penalty += 12.0
            engagement_penalty += 18.0
        else:
            if looking is not None and looking < 0.6:
                confidence_penalty += (0.6 - looking) * 30
                engagement_penalty += (0.6 - looking) * 35
            if eye_open is not None and eye_open < 0.35:
                confidence_penalty += 6.0
            if head_yaw + head_pitch > 0.55:
                confidence_penalty += min(10.0, (head_yaw + head_pitch - 0.55) * 25)
            if smile is not None and smile < 0.02 and face_result["signal"] != "engaged":
                engagement_penalty += 4.0

        confidence = max(0.0, min(100.0, confidence - confidence_penalty))
        engagement = max(0.0, min(100.0, engagement - engagement_penalty))

        total_ms = (time.perf_counter() - cycle_start) * 1000.0
        if total_ms > self.target_cycle_ms:
            self.actual_cycle_ms = min(float(self.max_cycle_ms), 0.5 * self.actual_cycle_ms + 0.5 * total_ms)
        else:
            self.actual_cycle_ms = max(float(self.target_cycle_ms), 0.7 * self.actual_cycle_ms + 0.3 * total_ms)

        driver = _dominant_driver(
            face_result["signal"],
            voice_result["signal"],
            lang_conf_session,
            transcription_info.get("filler_count", 0),
        )

        return {
            "timestamp": time.time(),
            "cycle_count": self.cycle_count,
            "confidence_score": round(confidence, 1),
            "engagement_score": round(engagement, 1),
            "face_signal": face_result["signal"],
            "face_emotion": face_result["emotion"],
            "face_detected": face_result.get("face_detected", False),
            "voice_signal": voice_result["signal"],
            "voice_emotion": voice_result["emotion"],
            "speaking": voice_result.get("speaking", False),
            "dominant_driver": driver,
            "filler_count": self.session_filler_total,
            "hedge_count": self.session_hedge_total,
            "latest_transcript": self.latest_transcript[-1500:],
            "flagged_phrases": transcription_info.get("flagged_phrases", []),
            "transcript_chunk": transcription_info.get("text", ""),
            "acoustic": {
                "speaking_rate": round(voice_result.get("speaking_rate", 0.0), 2),
                "energy_level": round(voice_result.get("energy_level", 0.0), 4),
            },
            "language_confidence": round(lang_conf_session, 1),
            "actual_cycle_ms": round(self.actual_cycle_ms, 1),
            "latency_ms": {
                "face": round(face_ms, 1),
                "voice": round(voice_ms, 1),
                "transcription": round(transcription_ms, 1),
                "fusion": round(fusion_ms, 1),
                "total": round(total_ms, 1),
            },
        }
