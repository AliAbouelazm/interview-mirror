"""Unit tests for app.pipeline.analysis."""
import pytest

from app.pipeline.analysis import (
    _energy_drops,
    _face_distribution,
    _filler_analysis,
    _gen_insights,
    _hedging_analysis,
    _segment_moments,
    _trend_label,
    _trend_slope,
    _voice_analysis,
    analyse_session,
)


def make_frame(t, conf=70.0, eng=70.0, signal="engaged", driver="voice", chunk="",
               flagged=None, energy=0.05, rate=3.0, speaking=True, face_detected=True):
    return {
        "timestamp": t,
        "confidence_score": conf,
        "engagement_score": eng,
        "face_signal": signal,
        "voice_signal": "confident",
        "dominant_driver": driver,
        "filler_count": 0,
        "hedge_count": 0,
        "transcript_chunk": chunk,
        "flagged_phrases": flagged or [],
        "acoustic": {"speaking_rate": rate, "energy_level": energy},
        "speaking": speaking,
        "face_detected": face_detected,
    }


def test_trend_slope_increasing():
    s = _trend_slope([10, 20, 30, 40, 50, 60])
    assert s > 0


def test_trend_slope_flat():
    assert _trend_slope([50.0] * 10) == 0


def test_trend_slope_too_few_points():
    assert _trend_slope([1, 2]) == 0


def test_trend_label_thresholds():
    assert _trend_label(0.5) == "improving"
    assert _trend_label(-0.5) == "declining"
    assert _trend_label(0.0) == "stable"
    assert _trend_label(0.04) == "stable"


def test_segment_moments_returns_high_and_low():
    base = 1000.0
    frames = []
    for i in range(60):
        score = 90 if i < 30 else 30
        frames.append(make_frame(base + i, conf=score, chunk=f"text-{i}"))
    high, low = _segment_moments(frames)
    assert len(high) > 0
    assert len(low) > 0
    assert high[0]["avg_score"] > low[0]["avg_score"]


def test_segment_moments_empty():
    assert _segment_moments([]) == ([], [])


def test_filler_analysis_counts_and_breakdown():
    base = 0.0
    frames = [
        make_frame(base, flagged=["um", "um", "like"]),
        make_frame(base + 5, flagged=["uh"]),
    ]
    analysis = _filler_analysis(frames, duration_minutes=1.0)
    assert analysis["total_count"] == 4
    assert analysis["breakdown"]["um"] == 2
    assert analysis["breakdown"]["uh"] == 1
    assert analysis["rate_per_minute"] == 4.0
    assert analysis["chart_data"][0]["label"] == "um"


def test_hedging_analysis_filters_to_known_hedges():
    frames = [
        make_frame(0, flagged=["i think", "um"]),
        make_frame(1, flagged=["maybe", "kind of"]),
    ]
    h = _hedging_analysis(frames, duration_minutes=2.0)
    assert h["total_count"] == 3
    assert "i think" in h["breakdown"]
    assert "um" not in h["breakdown"]


def test_energy_drops_detects_dip():
    frames = []
    for i in range(20):
        frames.append(make_frame(i, energy=0.1))
    for i in range(20, 30):
        frames.append(make_frame(i, energy=0.02))
    for i in range(30, 50):
        frames.append(make_frame(i, energy=0.1))
    drops = _energy_drops(frames)
    assert len(drops) == 1
    assert drops[0]["start_seconds"] >= 18 and drops[0]["start_seconds"] <= 20


def test_energy_drops_too_short():
    assert _energy_drops([make_frame(i) for i in range(3)]) == []


def test_voice_analysis_consistency_score():
    frames = [make_frame(i, energy=0.1) for i in range(20)]
    v = _voice_analysis(frames)
    assert v["frames_with_speech"] == 20
    assert v["energy_consistency"] >= 90.0


def test_face_distribution_dominant():
    frames = [make_frame(i, signal="engaged") for i in range(8)] + \
             [make_frame(i, signal="nervous") for i in range(2)]
    f = _face_distribution(frames)
    assert f["dominant"] == "engaged"
    assert f["distribution"]["engaged"] == 80.0


def test_face_distribution_empty():
    f = _face_distribution([])
    assert f["frames_with_face"] == 0


def test_gen_insights_at_most_five():
    frames = [make_frame(i, conf=80, flagged=["um"] * 3) for i in range(60)]
    overall = {"avg_confidence": 80, "avg_engagement": 80, "trend": "improving", "trend_slope": 0.1}
    high, low = _segment_moments(frames)
    filler = _filler_analysis(frames, 1.0)
    hedge = _hedging_analysis(frames, 1.0)
    voice = _voice_analysis(frames)
    face = _face_distribution(frames)
    insights = _gen_insights(frames, overall, high, low, filler, hedge, voice, face, 1.0)
    assert 1 <= len(insights) <= 5
    for ins in insights:
        assert ins["title"] and ins["body"]


def test_gen_insights_for_empty_session_returns_default():
    overall = {"avg_confidence": 0, "avg_engagement": 0, "trend": "stable", "trend_slope": 0.0}
    insights = _gen_insights(
        [],
        overall,
        [], [],
        {"total_count": 0, "rate_per_minute": 0.0, "breakdown": {}, "chart_data": []},
        {"total_count": 0, "rate_per_minute": 0.0, "breakdown": {}},
        {"avg_speaking_rate": 0, "avg_energy_level": 0, "energy_consistency": 0, "energy_drops": [], "frames_with_speech": 0},
        {"dominant": "engaged", "distribution": {}, "frames_with_face": 0},
        0.0,
    )
    assert len(insights) == 1
    assert insights[0]["metric"] == "empty"


def test_analyse_session_full():
    base = 1000.0
    frames = []
    for i in range(40):
        frames.append(make_frame(
            base + i,
            conf=80 if i < 20 else 40,
            chunk=f"chunk-{i}",
            flagged=["um"] if i % 5 == 0 else [],
        ))
    out = analyse_session(frames, started_at=base, ended_at=base + 40.0)
    assert out["frame_count"] == 40
    assert out["duration_seconds"] == 40.0
    assert "insights" in out and len(out["insights"]) >= 1
    assert out["overall"]["avg_confidence"] > 0
    assert out["filler"]["total_count"] >= 1


def test_analyse_session_single_frame():
    out = analyse_session([make_frame(1.0)], started_at=1.0, ended_at=1.0)
    assert out["frame_count"] == 1
    assert out["duration_seconds"] == 0.0
    assert "insights" in out


def test_analyse_session_no_speech():
    frames = [make_frame(i, energy=0.001, speaking=False, face_detected=False) for i in range(10)]
    out = analyse_session(frames, started_at=0.0, ended_at=10.0)
    assert out["voice"]["frames_with_speech"] == 0
    assert out["face"]["frames_with_face"] == 0
