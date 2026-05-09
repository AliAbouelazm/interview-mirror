"""
WebSocket endpoint. The client sends JSON messages with type "video", "audio",
or "ping". The server runs the realtime orchestrator at the cycle cadence and
emits one realtime payload per cycle.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.pipeline.realtime import RealtimeOrchestrator, decode_jpeg_b64, decode_pcm_b64
from app.utils.logging import get_logger


_log = get_logger(__name__)

router = APIRouter()


class SessionConnection:
    """One websocket plus its orchestrator. Frames buffer in, payloads stream out."""

    def __init__(self, ws: WebSocket, session_id: str, orchestrator: RealtimeOrchestrator, store):
        self.ws = ws
        self.session_id = session_id
        self.orchestrator = orchestrator
        self.store = store
        self.latest_frame_bgr = None
        self.closed = False

    async def receive_loop(self) -> None:
        try:
            while not self.closed:
                msg = await self.ws.receive_text()
                try:
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")
                if msg_type == "video":
                    frame = decode_jpeg_b64(data.get("frame", ""))
                    if frame is not None:
                        self.latest_frame_bgr = frame
                elif msg_type == "audio":
                    pcm = decode_pcm_b64(data.get("samples", ""))
                    if pcm is not None and pcm.size > 0:
                        self.orchestrator.push_audio(pcm)
                elif msg_type == "ping":
                    await self.ws.send_text(json.dumps({"type": "pong", "ts": time.time()}))
        except WebSocketDisconnect:
            pass
        except Exception as e:
            _log.warning(f"ws receive loop error: {e}")
        finally:
            self.closed = True

    async def cycle_loop(self) -> None:
        try:
            while not self.closed:
                t0 = time.perf_counter()
                payload = await asyncio.to_thread(self.orchestrator.cycle, self.latest_frame_bgr)
                payload["session_id"] = self.session_id
                await self.store.record_frame(self.session_id, payload)
                await self.ws.send_text(json.dumps({"type": "realtime", **payload}))
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                target_ms = self.orchestrator.actual_cycle_ms
                sleep_ms = max(0.0, target_ms - elapsed_ms)
                await asyncio.sleep(sleep_ms / 1000.0)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            _log.warning(f"ws cycle loop error: {e}")
        finally:
            self.closed = True


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    state = websocket.app.state
    if not state.session_store.is_live(session_id):
        await websocket.close(code=4404, reason="Session not active")
        return

    if not all(getattr(state, k, None) for k in ("face_predictor", "voice_predictor", "fusion_predictor", "transcriber")):
        await websocket.close(code=4503, reason="Models not loaded")
        return

    settings = state.settings
    orchestrator = RealtimeOrchestrator(
        face_predictor=state.face_predictor,
        voice_predictor=state.voice_predictor,
        fusion_predictor=state.fusion_predictor,
        transcriber=state.transcriber,
        target_cycle_ms=settings.target_cycle_ms,
        max_cycle_ms=settings.adaptive_cycle_max_ms,
        transcription_interval_cycles=settings.transcription_interval_cycles,
    )

    await websocket.accept()
    conn = SessionConnection(websocket, session_id, orchestrator, state.session_store)

    receive_task = asyncio.create_task(conn.receive_loop())
    cycle_task = asyncio.create_task(conn.cycle_loop())
    try:
        await asyncio.wait({receive_task, cycle_task}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        conn.closed = True
        for task in (receive_task, cycle_task):
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
