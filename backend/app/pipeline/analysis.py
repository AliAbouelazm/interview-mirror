"""
Post-session analysis. Aggregates per-frame outputs into overall scores, segment moments,
filler/hedge breakdowns, voice analysis, face analysis, and 3-5 actionable insights.

All insight strings are derived from real session data, not from a fixed template list.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


WINDOW_SECONDS = 10.0
ENERGY_DROP_FRACTION = 0.30
TOP_K = 3


def _safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    arr = np.array([v for v in values if v is not None], dtype=np.float32)
    if arr.size == 0:
        return default
    return float(arr.mean())


def _trend_slope(values: list[float]) -> float:
    """Linear-regression slope of values vs index. Returns 0 for fewer than 5 points."""
    if len(values) < 5:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    y = np.array(values, dtype=np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom <= 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def _trend_label(slope: float) -> str:
    if slope > 0.05:
        return "improving"
    if slope < -0.05:
        return "declining"
    return "stable"


def _format_time(seconds: float) -> str:
    s = int(seconds)
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}"


def _segment_moments(frames: list[dict]) -> tuple[list[dict], list[dict]]:
    """Top-3 strongest and weakest 10s windows by mean confidence."""
    if not frames:
        return [], []
    starts = []
    last_ts = frames[-1].get("timestamp", 0)
    first_ts = frames[0].get("timestamp", 0)
    duration = max(1.0, last_ts - first_ts)
    n_windows = max(1, int(duration // WINDOW_SECONDS))

    windows = []
    for i in range(n_windows):
        w_start = first_ts + i * WINDOW_SECONDS
        w_end = w_start + WINDOW_SECONDS
        window_frames = [f for f in frames if w_start <= f.get("timestamp", 0) < w_end]
        if not window_frames:
            continue
        scores = [f.get("confidence_score", 0) for f in window_frames]
        avg_score = float(np.mean(scores))
        drivers = Counter(f.get("dominant_driver", "voice") for f in window_frames)
        excerpt = ""
        for f in window_frames:
            chunk = f.get("transcript_chunk")
            if chunk:
                excerpt += " " + chunk
        excerpt = excerpt.strip()[:300]
        windows.append({
            "start_seconds": round(w_start - first_ts, 2),
            "end_seconds": round(min(w_end, last_ts) - first_ts, 2),
            "avg_score": round(avg_score, 1),
            "dominant_driver": drivers.most_common(1)[0][0],
            "transcript_excerpt": excerpt,
        })

    if not windows:
        return [], []

    sorted_high = sorted(windows, key=lambda w: -w["avg_score"])[:TOP_K]
    sorted_low = sorted(windows, key=lambda w: w["avg_score"])[:TOP_K]
    return sorted_high, sorted_low


def _energy_drops(frames: list[dict]) -> list[dict]:
    energies = [f.get("acoustic", {}).get("energy_level", 0) for f in frames]
    if len(energies) < 5:
        return []
    avg = float(np.mean(energies))
    threshold = avg * (1 - ENERGY_DROP_FRACTION)
    drops = []
    in_drop = False
    drop_start = None
    base_ts = frames[0].get("timestamp", 0)
    for f, e in zip(frames, energies):
        if e < threshold and not in_drop:
            in_drop = True
            drop_start = f.get("timestamp", 0) - base_ts
        elif e >= threshold and in_drop:
            in_drop = False
            drops.append({
                "start_seconds": round(drop_start, 2),
                "end_seconds": round(f.get("timestamp", 0) - base_ts, 2),
            })
    if in_drop and drop_start is not None:
        drops.append({
            "start_seconds": round(drop_start, 2),
            "end_seconds": round(frames[-1].get("timestamp", 0) - base_ts, 2),
        })
    return drops


def _face_distribution(frames: list[dict]) -> dict:
    counter = Counter(f.get("face_signal", "engaged") for f in frames if f.get("face_detected"))
    total = sum(counter.values()) or 1
    distribution = {k: round(100.0 * v / total, 1) for k, v in counter.items()}
    dominant = counter.most_common(1)[0][0] if counter else "engaged"
    return {"dominant": dominant, "distribution": distribution, "frames_with_face": sum(counter.values())}


def _filler_analysis(frames: list[dict], duration_minutes: float) -> dict:
    all_phrases = []
    for f in frames:
        all_phrases.extend(f.get("flagged_phrases", []))
    counts = Counter(all_phrases)
    total = sum(counts.values())
    rate = total / max(duration_minutes, 1e-6)
    chart = [{"label": k, "count": v} for k, v in counts.most_common()]
    return {
        "total_count": total,
        "rate_per_minute": round(rate, 2),
        "breakdown": dict(counts),
        "chart_data": chart,
    }


def _hedging_analysis(frames: list[dict], duration_minutes: float) -> dict:
    counts = Counter()
    HEDGE_KEYS = {"i think", "i believe", "maybe", "probably", "kind of", "sort of",
                  "i guess", "perhaps", "i feel like"}
    for f in frames:
        for p in f.get("flagged_phrases", []):
            if p.lower() in HEDGE_KEYS:
                counts[p.lower()] += 1
    total = sum(counts.values())
    rate = total / max(duration_minutes, 1e-6)
    return {
        "total_count": total,
        "rate_per_minute": round(rate, 2),
        "breakdown": dict(counts),
    }


def _voice_analysis(frames: list[dict]) -> dict:
    rates = [f.get("acoustic", {}).get("speaking_rate", 0) for f in frames if f.get("speaking")]
    energies = [f.get("acoustic", {}).get("energy_level", 0) for f in frames if f.get("speaking")]
    avg_rate = _safe_mean(rates)
    avg_energy = _safe_mean(energies)
    energy_std = float(np.std(energies)) if energies else 0.0
    consistency = float(max(0.0, 1.0 - min(energy_std / max(avg_energy, 1e-3), 1.0)) * 100.0) if avg_energy else 0.0
    drops = _energy_drops(frames)
    return {
        "avg_speaking_rate": round(avg_rate, 2),
        "avg_energy_level": round(avg_energy, 4),
        "energy_consistency": round(consistency, 1),
        "energy_drops": drops,
        "frames_with_speech": len(rates),
    }


def _gen_insights(
    frames: list[dict],
    overall: dict,
    moments_high: list[dict],
    moments_low: list[dict],
    filler: dict,
    hedge: dict,
    voice: dict,
    face: dict,
    duration_minutes: float,
) -> list[dict]:
    """Each insight references a specific real metric or timestamp from this session."""
    insights: list[dict] = []

    if filler["total_count"] >= 3 and filler["chart_data"]:
        top_phrase, top_count = filler["chart_data"][0]["label"], filler["chart_data"][0]["count"]
        insights.append({
            "title": f"You used '{top_phrase}' {top_count} times",
            "body": (
                f"Across {duration_minutes:.1f} minutes you fell back on '{top_phrase}' "
                f"{top_count} times, contributing to a filler rate of {filler['rate_per_minute']} per minute. "
                "Replacing it with a short pause makes the answer sound more deliberate."
            ),
            "metric": "filler",
        })
    elif filler["rate_per_minute"] < 1.0 and duration_minutes > 0.5:
        insights.append({
            "title": "Filler use stayed under 1 per minute",
            "body": (
                f"Filler rate was {filler['rate_per_minute']} per minute, "
                "which is the band recruiters describe as polished. Hold the line on this in longer answers."
            ),
            "metric": "filler",
        })

    if moments_low:
        m = moments_low[0]
        insights.append({
            "title": f"Lowest stretch was {_format_time(m['start_seconds'])} to {_format_time(m['end_seconds'])}",
            "body": (
                f"Confidence dropped to {m['avg_score']} during this window, driven by your "
                f"{m['dominant_driver']} signal. Review the transcript here and rehearse this section."
            ),
            "metric": "low_moment",
        })

    if moments_high:
        m = moments_high[0]
        insights.append({
            "title": f"Strongest stretch was {_format_time(m['start_seconds'])} to {_format_time(m['end_seconds'])}",
            "body": (
                f"You averaged {m['avg_score']} confidence in this window, led by your "
                f"{m['dominant_driver']}. Use the same delivery pattern earlier in the answer."
            ),
            "metric": "high_moment",
        })

    if hedge["total_count"] >= 2:
        keys = list(hedge["breakdown"].keys())
        sample = keys[0] if keys else "i think"
        insights.append({
            "title": f"Hedging language appeared {hedge['total_count']} times",
            "body": (
                f"Phrases like '{sample}' weakened {hedge['total_count']} statements. "
                "Drop the qualifier and state the claim directly when you are sure."
            ),
            "metric": "hedge",
        })

    if voice["energy_drops"]:
        d = voice["energy_drops"][0]
        insights.append({
            "title": f"Voice trailed off near {_format_time(d['start_seconds'])}",
            "body": (
                f"Vocal energy fell more than {int(ENERGY_DROP_FRACTION * 100)}% below your average between "
                f"{_format_time(d['start_seconds'])} and {_format_time(d['end_seconds'])}. "
                "Practising the close of each answer can keep volume up through the final sentence."
            ),
            "metric": "energy_drop",
        })

    nervous_pct = face["distribution"].get("nervous", 0)
    if nervous_pct > 25:
        insights.append({
            "title": f"You looked nervous in {nervous_pct}% of frames",
            "body": (
                "When you noticed yourself stalling, your face shifted into a nervous pattern. "
                "Pausing to breathe before answering, even for one second, often resets this."
            ),
            "metric": "face",
        })

    if overall["trend"] == "declining" and overall["avg_confidence"] >= 50:
        insights.append({
            "title": "You started strong but trended down",
            "body": (
                f"Your confidence slope was negative across the session ({overall['trend_slope']:+.2f} per cycle). "
                "Expect the back half of an interview to be harder and pace yourself for it."
            ),
            "metric": "trend",
        })
    elif overall["trend"] == "improving" and overall["avg_confidence"] >= 50:
        insights.append({
            "title": "You warmed up as the session progressed",
            "body": (
                f"Confidence trended upward with slope {overall['trend_slope']:+.2f}. "
                "Use a one-minute warm-up out loud before the real interview to start at this level."
            ),
            "metric": "trend",
        })

    if not insights:
        insights.append({
            "title": "Not enough data for personalised insights",
            "body": "Run a session of at least one minute to generate detailed feedback.",
            "metric": "empty",
        })

    return insights[:5]


def analyse_session(frames: list[dict], started_at: float, ended_at: float) -> dict:
    """Build the full analysis payload from per-frame data."""
    duration_seconds = max(0.0, ended_at - started_at)
    duration_minutes = duration_seconds / 60.0

    confidences = [f.get("confidence_score", 0) for f in frames]
    engagements = [f.get("engagement_score", 0) for f in frames]

    avg_conf = _safe_mean(confidences)
    avg_eng = _safe_mean(engagements)
    slope = _trend_slope(confidences)

    overall = {
        "avg_confidence": round(avg_conf, 1),
        "avg_engagement": round(avg_eng, 1),
        "trend": _trend_label(slope),
        "trend_slope": round(slope, 4),
    }

    high, low = _segment_moments(frames)
    filler = _filler_analysis(frames, duration_minutes)
    hedge = _hedging_analysis(frames, duration_minutes)
    voice = _voice_analysis(frames)
    face = _face_distribution(frames)

    insights = _gen_insights(frames, overall, high, low, filler, hedge, voice, face, duration_minutes)

    return {
        "duration_seconds": round(duration_seconds, 1),
        "frame_count": len(frames),
        "overall": overall,
        "moments": {"strongest": high, "weakest": low},
        "filler": filler,
        "hedging": hedge,
        "voice": voice,
        "face": face,
        "insights": insights,
    }
