"""Pydantic schemas for the public HTTP API."""
from typing import Optional

from pydantic import BaseModel, Field


class StartSessionRequest(BaseModel):
    context: str = Field(default="", max_length=500, description="Optional notes about the session.")


class StartSessionResponse(BaseModel):
    session_id: str = Field(description="Server-generated session id.")
    started_at: float = Field(description="Unix timestamp when the session began.")


class EndSessionResponse(BaseModel):
    session_id: str
    ended_at: float
    duration_seconds: float


class AcousticFeatures(BaseModel):
    speaking_rate: float = Field(default=0.0)
    energy_level: float = Field(default=0.0)


class FrameSummary(BaseModel):
    timestamp: float
    confidence_score: float
    engagement_score: float
    face_signal: str
    voice_signal: str
    dominant_driver: str
    filler_count: int = 0
    hedge_count: int = 0
    transcript_chunk: str = ""
    flagged_phrases: list[str] = Field(default_factory=list)
    acoustic: AcousticFeatures = Field(default_factory=AcousticFeatures)


class TimelineResponse(BaseModel):
    session_id: str
    frames: list[FrameSummary]


class MomentItem(BaseModel):
    start_seconds: float
    end_seconds: float
    avg_score: float
    dominant_driver: str
    transcript_excerpt: str = ""


class MomentsBlock(BaseModel):
    strongest: list[MomentItem] = Field(default_factory=list)
    weakest: list[MomentItem] = Field(default_factory=list)


class FillerChartItem(BaseModel):
    label: str
    count: int


class FillerBlock(BaseModel):
    total_count: int = 0
    rate_per_minute: float = 0.0
    breakdown: dict[str, int] = Field(default_factory=dict)
    chart_data: list[FillerChartItem] = Field(default_factory=list)


class HedgingBlock(BaseModel):
    total_count: int = 0
    rate_per_minute: float = 0.0
    breakdown: dict[str, int] = Field(default_factory=dict)


class EnergyDrop(BaseModel):
    start_seconds: float
    end_seconds: float


class VoiceBlock(BaseModel):
    avg_speaking_rate: float = 0.0
    avg_energy_level: float = 0.0
    energy_consistency: float = 0.0
    energy_drops: list[EnergyDrop] = Field(default_factory=list)
    frames_with_speech: int = 0


class FaceBlock(BaseModel):
    dominant: str = "engaged"
    distribution: dict[str, float] = Field(default_factory=dict)
    frames_with_face: int = 0


class OverallBlock(BaseModel):
    avg_confidence: float = 0.0
    avg_engagement: float = 0.0
    trend: str = "stable"
    trend_slope: float = 0.0


class InsightItem(BaseModel):
    title: str
    body: str
    metric: str


class PerQuestionItem(BaseModel):
    question_id: str
    text: str
    category: str
    difficulty: str
    target_seconds: int
    answered_seconds: float
    avg_confidence: float
    avg_engagement: float
    word_count: int
    filler_count: int


class FaceDynamicsBlock(BaseModel):
    available: bool = False
    tick_count: int = 0
    head_yaw_std: float = 0.0
    head_pitch_std: float = 0.0
    avg_eye_openness: float = 1.0
    avg_smile: float = 0.0
    looking_pct: float = 100.0
    ticks: list[dict] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    session_id: str
    duration_seconds: float
    frame_count: int
    overall: OverallBlock
    moments: MomentsBlock
    filler: FillerBlock
    hedging: HedgingBlock
    voice: VoiceBlock
    face: FaceBlock
    insights: list[InsightItem]
    per_question: list[PerQuestionItem] = Field(default_factory=list)
    face_dynamics: FaceDynamicsBlock = Field(default_factory=FaceDynamicsBlock)


class SessionListItem(BaseModel):
    id: str
    context: str = ""
    started_at: float
    ended_at: Optional[float] = None
    duration_seconds: float = 0.0
    avg_confidence: float = 0.0
    avg_engagement: float = 0.0


class ModelMetrics(BaseModel):
    name: str
    dataset: str
    test_f1_macro: Optional[float] = None
    test_accuracy: Optional[float] = None
    classes: list[str] = Field(default_factory=list)


class ModelMetricsResponse(BaseModel):
    face: Optional[ModelMetrics] = None
    voice: Optional[ModelMetrics] = None
    fusion: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    models: dict[str, bool]


class QuestionItem(BaseModel):
    id: str
    text: str
    category: str
    difficulty: str
    target_seconds: int


class QuestionSetRequest(BaseModel):
    n: int = Field(default=5, ge=1, le=20)
    category: Optional[str] = None
    seed: Optional[int] = None


class QuestionEventRequest(BaseModel):
    question_id: str = Field(min_length=1, max_length=64)
    started_at: float
    ended_at: Optional[float] = None


class FaceMetricsTick(BaseModel):
    timestamp: float
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0
    eye_openness: float = 1.0
    smile: float = 0.0
    looking_at_camera: float = 1.0


class FaceMetricsBatch(BaseModel):
    ticks: list[FaceMetricsTick]
