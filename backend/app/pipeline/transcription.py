"""
Whisper transcription via faster-whisper, plus language-feature analysis.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from typing import Optional

import numpy as np

FILLER_WORDS = [
    "um", "uh", "like", "you know", "basically", "literally", "so", "right",
]
HEDGE_PHRASES = [
    "i think", "i believe", "maybe", "probably", "kind of", "sort of",
    "i guess", "perhaps", "i feel like",
]
ASSERTIVE_VERBS = {
    "build", "ship", "lead", "own", "deliver", "drive", "design",
    "implement", "launch", "execute", "scale", "improve", "increase",
    "reduce", "decide", "choose", "create", "solve", "manage",
}

# Tokenization keeps apostrophes inside words ("don't") but strips other punctuation.
_WORD_RE = re.compile(r"[A-Za-z']+")


class Transcriber:
    """Wrapper around faster-whisper with timing instrumentation."""

    def __init__(self, model_size: str = "base", device: str = "auto", compute_type: str = "int8"):
        from faster_whisper import WhisperModel
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.device = device
        self.model_size = model_size

    def transcribe_chunk(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        """Transcribe a mono float32 audio buffer. Returns dict with text and latency."""
        if audio is None or audio.size == 0:
            return {"text": "", "latency_ms": 0.0, "language": ""}
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
        if max_abs > 0:
            audio = audio / max(max_abs, 1.0)

        t0 = time.perf_counter()
        segments, info = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {"text": text, "latency_ms": latency_ms, "language": info.language}


def _normalize(text: str) -> str:
    return text.lower().strip()


def count_phrases(text: str, phrases: list[str]) -> tuple[int, list[str]]:
    """Case-insensitive count and ordered list of all matched phrases."""
    if not text:
        return 0, []
    norm = _normalize(text)
    found = []
    for phrase in phrases:
        pat = r"\b" + re.escape(phrase) + r"\b"
        for m in re.finditer(pat, norm):
            found.append(phrase)
    return len(found), found


def count_assertive_words(text: str) -> int:
    if not text:
        return 0
    return sum(1 for w in _WORD_RE.findall(text.lower()) if w in ASSERTIVE_VERBS)


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def language_confidence(filler_count: int, hedge_count: int, assertive_count: int, words: int) -> float:
    """0-100 score. Many fillers and hedges drop it; assertive verbs raise it. Smoothed for short text."""
    if words < 5:
        return 60.0
    filler_rate = filler_count / words
    hedge_rate = hedge_count / words
    assertive_rate = assertive_count / words

    base = 80.0
    base -= 220.0 * filler_rate
    base -= 180.0 * hedge_rate
    base += 120.0 * assertive_rate
    return float(max(0.0, min(100.0, base)))


def analyze_text(text: str) -> dict:
    """Run all language detectors on a single transcript chunk."""
    filler_count, filler_list = count_phrases(text, FILLER_WORDS)
    hedge_count, hedge_list = count_phrases(text, HEDGE_PHRASES)
    assertive_count = count_assertive_words(text)
    words = word_count(text)
    flagged = filler_list + hedge_list

    return {
        "filler_count": filler_count,
        "filler_list": filler_list,
        "hedge_count": hedge_count,
        "hedge_list": hedge_list,
        "assertive_count": assertive_count,
        "word_count": words,
        "language_confidence": language_confidence(filler_count, hedge_count, assertive_count, words),
        "flagged_phrases": flagged[:10],
    }


def aggregate_text_stats(per_chunk: list[dict]) -> dict:
    """Roll up per-chunk language stats over a session."""
    fillers = []
    hedges = []
    total_words = 0
    for c in per_chunk:
        fillers.extend(c.get("filler_list", []))
        hedges.extend(c.get("hedge_list", []))
        total_words += c.get("word_count", 0)
    return {
        "total_filler_count": len(fillers),
        "total_hedge_count": len(hedges),
        "total_word_count": total_words,
        "filler_breakdown": dict(Counter(fillers)),
        "hedge_breakdown": dict(Counter(hedges)),
    }
