"""
FastAPI entry point. Loads all models on startup and wires HTTP + WS routes.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.routes import router as http_router
from app.api.websocket import router as ws_router
from app.pipeline.session import SessionStore
from app.pipeline.transcription import Transcriber
from app.utils.config import get_settings
from app.utils.logging import configure_logging, get_logger


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger("startup")
    app.state.settings = settings
    app.state.started_at = time.time()

    device = _device()
    log.info(f"loading models on {device}")

    face_path = settings.resolve(settings.face_model_path)
    voice_path = settings.resolve(settings.voice_model_path)
    fusion_path = settings.resolve(settings.fusion_model_path)

    missing = [p for p in (face_path, voice_path, fusion_path) if not p.exists()]
    if missing:
        msg = "Missing model files: " + ", ".join(str(m) for m in missing)
        log.error(msg)
        raise RuntimeError(msg)

    from app.models.face.predict import FacePredictor
    from app.models.voice.predict import VoicePredictor
    from app.models.fusion.predict import FusionPredictor

    app.state.face_predictor = FacePredictor(face_path, device=device)
    app.state.voice_predictor = VoicePredictor(voice_path, device=device)
    app.state.fusion_predictor = FusionPredictor(fusion_path, device=device)
    log.info("face/voice/fusion models loaded")

    app.state.transcriber = Transcriber(model_size=settings.whisper_model_size, device="cpu", compute_type="int8")
    log.info(f"whisper {settings.whisper_model_size} loaded")

    store = SessionStore(db_path=str(settings.resolve(settings.session_db_path)),
                         ttl_hours=settings.session_ttl_hours)
    await store.connect()
    app.state.session_store = store

    janitor_task = asyncio.create_task(_janitor_loop(store))
    yield
    janitor_task.cancel()
    try:
        await janitor_task
    except asyncio.CancelledError:
        pass
    await store.close()


async def _janitor_loop(store: SessionStore) -> None:
    while True:
        try:
            await store.janitor()
        except Exception as e:
            get_logger("janitor").warning(f"janitor error: {e}")
        await asyncio.sleep(3600)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Interview Mirror API", version="1.0.0", lifespan=lifespan)

    limiter = Limiter(key_func=get_remote_address, default_limits=[f"{settings.rate_limit_per_min}/minute"])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    cors_list = settings.cors_origins_list()
    allow_credentials = cors_list != ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_list,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        get_logger("api").exception("unhandled error")
        return JSONResponse(status_code=500, content={"error": "internal_error", "detail": str(exc)})

    app.include_router(http_router, prefix="/api")
    app.include_router(ws_router)
    return app


app = create_app()
