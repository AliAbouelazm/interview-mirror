"""HTTP routes. The WebSocket lives in websocket.py."""
from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import (
    AnalysisResponse,
    EndSessionResponse,
    HealthResponse,
    ModelMetrics,
    ModelMetricsResponse,
    SessionListItem,
    StartSessionRequest,
    StartSessionResponse,
    TimelineResponse,
)


router = APIRouter()


@router.post("/session/start", response_model=StartSessionResponse)
async def start_session(req: StartSessionRequest, request: Request):
    store = request.app.state.session_store
    sess = await store.start(context=req.context)
    return StartSessionResponse(session_id=sess.id, started_at=sess.started_at)


@router.post("/session/{session_id}/end", response_model=EndSessionResponse)
async def end_session(session_id: str, request: Request):
    store = request.app.state.session_store
    if not store.is_live(session_id):
        meta = await store.get_session_meta(session_id)
        if not meta:
            raise HTTPException(status_code=404, detail="Session not found")
        return EndSessionResponse(
            session_id=session_id,
            ended_at=meta["ended_at"] or 0.0,
            duration_seconds=meta["duration_seconds"] or 0.0,
        )
    await store.end(session_id)
    meta = await store.get_session_meta(session_id)
    return EndSessionResponse(
        session_id=session_id,
        ended_at=meta["ended_at"] or time.time(),
        duration_seconds=meta["duration_seconds"] or 0.0,
    )


@router.get("/session/{session_id}/analysis", response_model=AnalysisResponse)
async def get_analysis(session_id: str, request: Request):
    store = request.app.state.session_store
    analysis = await store.get_analysis(session_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return AnalysisResponse(session_id=session_id, **analysis)


@router.get("/session/{session_id}/timeline", response_model=TimelineResponse)
async def get_timeline(session_id: str, request: Request):
    store = request.app.state.session_store
    frames = await store.get_timeline(session_id)
    if not frames:
        raise HTTPException(status_code=404, detail="Timeline not found")
    return TimelineResponse(session_id=session_id, frames=frames)


@router.get("/sessions", response_model=list[SessionListItem])
async def list_sessions(request: Request, limit: int = 50):
    store = request.app.state.session_store
    return await store.list_finished(limit=limit)


def _read_metrics_json(path: Path, name: str) -> ModelMetrics | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return ModelMetrics(
            name=name,
            dataset=data.get("dataset", ""),
            test_f1_macro=data.get("test_f1_macro"),
            test_accuracy=data.get("test_accuracy"),
            classes=data.get("classes", []),
        )
    except Exception:
        return None


@router.get("/model/metrics", response_model=ModelMetricsResponse)
async def model_metrics(request: Request):
    settings = request.app.state.settings
    face = _read_metrics_json(settings.resolve(settings.face_model_path).with_suffix(".metrics.json"), "face")
    voice = _read_metrics_json(settings.resolve(settings.voice_model_path).with_suffix(".metrics.json"), "voice")
    fusion_metrics_path = settings.resolve(settings.fusion_model_path).with_suffix(".metrics.json")
    fusion = None
    if fusion_metrics_path.exists():
        try:
            with open(fusion_metrics_path) as f:
                fusion = json.load(f)
        except Exception:
            fusion = None
    return ModelMetricsResponse(face=face, voice=voice, fusion=fusion)


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    state = request.app.state
    return HealthResponse(
        status="ok",
        uptime_seconds=time.time() - state.started_at,
        models={
            "face": getattr(state, "face_predictor", None) is not None,
            "voice": getattr(state, "voice_predictor", None) is not None,
            "fusion": getattr(state, "fusion_predictor", None) is not None,
            "transcriber": getattr(state, "transcriber", None) is not None,
        },
    )
