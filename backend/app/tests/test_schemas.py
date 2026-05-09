"""Validate every Pydantic schema against valid and invalid inputs."""
import pytest
from pydantic import ValidationError

from app.api.schemas import (
    AcousticFeatures,
    AnalysisResponse,
    EndSessionResponse,
    FillerBlock,
    FrameSummary,
    HedgingBlock,
    InsightItem,
    MomentItem,
    MomentsBlock,
    OverallBlock,
    SessionListItem,
    StartSessionRequest,
    StartSessionResponse,
    TimelineResponse,
    FaceBlock,
    VoiceBlock,
)


def test_start_session_request_default():
    req = StartSessionRequest()
    assert req.context == ""


def test_start_session_request_too_long():
    with pytest.raises(ValidationError):
        StartSessionRequest(context="x" * 1000)


def test_start_session_response_required_fields():
    r = StartSessionResponse(session_id="abc", started_at=1.0)
    assert r.session_id == "abc"
    with pytest.raises(ValidationError):
        StartSessionResponse(started_at=1.0)


def test_end_session_response():
    r = EndSessionResponse(session_id="x", ended_at=10.0, duration_seconds=5.0)
    assert r.duration_seconds == 5.0


def test_frame_summary_defaults_and_required():
    f = FrameSummary(
        timestamp=1.0, confidence_score=80.0, engagement_score=70.0,
        face_signal="engaged", voice_signal="confident", dominant_driver="voice",
    )
    assert f.filler_count == 0
    assert isinstance(f.acoustic, AcousticFeatures)
    with pytest.raises(ValidationError):
        FrameSummary(timestamp=1.0)


def test_timeline_response_with_frames():
    frames = [
        FrameSummary(timestamp=1.0, confidence_score=80, engagement_score=70,
                     face_signal="engaged", voice_signal="confident", dominant_driver="voice"),
    ]
    r = TimelineResponse(session_id="abc", frames=frames)
    assert len(r.frames) == 1


def test_moments_block_default():
    m = MomentsBlock()
    assert m.strongest == [] and m.weakest == []


def test_moment_item_validation():
    item = MomentItem(start_seconds=0, end_seconds=10, avg_score=80, dominant_driver="face")
    assert item.transcript_excerpt == ""
    with pytest.raises(ValidationError):
        MomentItem(start_seconds="not-a-number", end_seconds=10, avg_score=80, dominant_driver="face")


def test_filler_block_default():
    f = FillerBlock()
    assert f.total_count == 0
    assert f.chart_data == []


def test_hedging_block_default():
    h = HedgingBlock()
    assert h.total_count == 0


def test_face_voice_block_defaults():
    f = FaceBlock()
    assert f.dominant == "engaged"
    v = VoiceBlock()
    assert v.frames_with_speech == 0


def test_insight_item():
    i = InsightItem(title="t", body="b", metric="filler")
    assert i.metric == "filler"


def test_overall_block_default():
    o = OverallBlock()
    assert o.trend == "stable"


def test_analysis_response_minimum_payload():
    payload = {
        "session_id": "abc",
        "duration_seconds": 60.0,
        "frame_count": 100,
        "overall": {"avg_confidence": 75, "avg_engagement": 70, "trend": "improving", "trend_slope": 0.1},
        "moments": {"strongest": [], "weakest": []},
        "filler": {"total_count": 0, "rate_per_minute": 0.0, "breakdown": {}, "chart_data": []},
        "hedging": {"total_count": 0, "rate_per_minute": 0.0, "breakdown": {}},
        "voice": {"avg_speaking_rate": 0, "avg_energy_level": 0, "energy_consistency": 0, "energy_drops": [], "frames_with_speech": 0},
        "face": {"dominant": "engaged", "distribution": {}, "frames_with_face": 0},
        "insights": [],
    }
    r = AnalysisResponse(**payload)
    assert r.session_id == "abc"
    assert r.frame_count == 100


def test_session_list_item():
    s = SessionListItem(id="a", started_at=1.0)
    assert s.duration_seconds == 0.0
