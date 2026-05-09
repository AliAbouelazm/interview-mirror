"""
Session lifecycle and SQLite persistence.

Sessions hold per-frame outputs while live, then run analysis on end and write
the analysis JSON back to SQLite. Frames older than SESSION_TTL_HOURS are pruned
on a periodic janitor pass.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import aiosqlite

from app.pipeline.analysis import analyse_session
from app.utils.logging import get_logger


_log = get_logger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    context TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    duration_seconds REAL,
    avg_confidence REAL,
    avg_engagement REAL,
    analysis_json TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS frames (
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (session_id, seq),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS question_events (
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    question_id TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL,
    PRIMARY KEY (session_id, seq),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS face_ticks (
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (session_id, seq),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS bookmarks (
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    label TEXT NOT NULL,
    note TEXT,
    PRIMARY KEY (session_id, seq),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_frames_session ON frames(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_ended ON sessions(ended_at);
CREATE INDEX IF NOT EXISTS idx_qevents_session ON question_events(session_id);
CREATE INDEX IF NOT EXISTS idx_face_ticks_session ON face_ticks(session_id);
"""


@dataclass
class LiveSession:
    id: str
    context: str
    started_at: float
    frames: list[dict] = field(default_factory=list)
    seq: int = 0
    question_events: list[dict] = field(default_factory=list)
    face_ticks: list[dict] = field(default_factory=list)
    bookmarks: list[dict] = field(default_factory=list)


class SessionStore:
    """Async session manager. Live sessions live in memory, finished ones live in SQLite."""

    def __init__(self, db_path: str, ttl_hours: int = 24):
        self.db_path = db_path
        self.ttl_seconds = ttl_hours * 3600
        self._live: dict[str, LiveSession] = {}
        self._lock = asyncio.Lock()
        self._db: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        from pathlib import Path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.executescript(SCHEMA)
        await self._db.commit()
        _log.info(f"session store ready at {self.db_path}")

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def start(self, context: str = "") -> LiveSession:
        sid = uuid.uuid4().hex
        sess = LiveSession(id=sid, context=context or "", started_at=time.time())
        async with self._lock:
            self._live[sid] = sess
        if self._db:
            await self._db.execute(
                "INSERT INTO sessions(id, context, started_at, created_at) VALUES (?, ?, ?, ?)",
                (sid, sess.context, sess.started_at, sess.started_at),
            )
            await self._db.commit()
        return sess

    def is_live(self, session_id: str) -> bool:
        return session_id in self._live

    async def record_frame(self, session_id: str, payload: dict) -> None:
        sess = self._live.get(session_id)
        if not sess:
            return
        sess.seq += 1
        payload = dict(payload)
        payload["seq"] = sess.seq
        sess.frames.append(payload)

    async def record_question_event(self, session_id: str, event: dict) -> None:
        sess = self._live.get(session_id)
        if not sess:
            return
        sess.question_events.append(event)

    async def record_face_ticks(self, session_id: str, ticks: list[dict]) -> None:
        sess = self._live.get(session_id)
        if not sess:
            return
        sess.face_ticks.extend(ticks)

    async def add_bookmark(self, session_id: str, bookmark: dict) -> None:
        sess = self._live.get(session_id)
        if not sess:
            return
        sess.bookmarks.append(bookmark)

    async def end(self, session_id: str) -> Optional[dict]:
        sess = self._live.pop(session_id, None)
        if not sess:
            return None
        ended_at = time.time()
        analysis = analyse_session(sess.frames, sess.started_at, ended_at)
        analysis["per_question"] = self._compute_per_question(sess)
        analysis["face_dynamics"] = self._compute_face_dynamics(sess)
        analysis["bookmarks"] = list(sess.bookmarks)
        if self._db:
            await self._db.execute(
                "UPDATE sessions SET ended_at=?, duration_seconds=?, avg_confidence=?, "
                "avg_engagement=?, analysis_json=? WHERE id=?",
                (
                    ended_at,
                    analysis["duration_seconds"],
                    analysis["overall"]["avg_confidence"],
                    analysis["overall"]["avg_engagement"],
                    json.dumps(analysis),
                    session_id,
                ),
            )
            for f in sess.frames:
                await self._db.execute(
                    "INSERT OR REPLACE INTO frames(session_id, seq, timestamp, payload_json) VALUES (?, ?, ?, ?)",
                    (session_id, f["seq"], f.get("timestamp", time.time()), json.dumps(f)),
                )
            for i, ev in enumerate(sess.question_events):
                await self._db.execute(
                    "INSERT OR REPLACE INTO question_events(session_id, seq, question_id, started_at, ended_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (session_id, i, ev.get("question_id", ""), ev.get("started_at", 0), ev.get("ended_at")),
                )
            for i, t in enumerate(sess.face_ticks):
                await self._db.execute(
                    "INSERT OR REPLACE INTO face_ticks(session_id, seq, timestamp, payload_json) "
                    "VALUES (?, ?, ?, ?)",
                    (session_id, i, t.get("timestamp", 0), json.dumps(t)),
                )
            for i, b in enumerate(sess.bookmarks):
                await self._db.execute(
                    "INSERT OR REPLACE INTO bookmarks(session_id, seq, timestamp, label, note) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (session_id, i, b.get("timestamp", 0), b.get("label", "moment"), b.get("note", "")),
                )
            await self._db.commit()
        return analysis

    def _compute_per_question(self, sess: LiveSession) -> list[dict]:
        """Merge multiple events per question (open + close) into one entry per qid."""
        from app.data.questions import get_question
        from statistics import mean

        merged: dict[str, dict] = {}
        order: list[str] = []
        for ev in sess.question_events:
            qid = ev.get("question_id")
            if not qid:
                continue
            if qid not in merged:
                merged[qid] = {
                    "started_at": ev.get("started_at"),
                    "ended_at": ev.get("ended_at"),
                    "self_rating": ev.get("self_rating"),
                }
                order.append(qid)
                continue
            entry = merged[qid]
            if ev.get("started_at") is not None:
                entry["started_at"] = min(entry["started_at"] or ev["started_at"], ev["started_at"])
            if ev.get("ended_at") is not None:
                entry["ended_at"] = max(entry["ended_at"] or 0, ev["ended_at"])
            if ev.get("self_rating") is not None:
                entry["self_rating"] = ev["self_rating"]

        per_q: list[dict] = []
        last_ts = sess.frames[-1]["timestamp"] if sess.frames else 0
        for qid in order:
            q = get_question(qid)
            if q is None:
                continue
            ev = merged[qid]
            start = ev.get("started_at") or 0
            end = ev.get("ended_at") or last_ts or start
            window = [f for f in sess.frames if start <= f.get("timestamp", 0) < end]
            if not window:
                continue
            avg_conf = round(mean([f.get("confidence_score", 0) for f in window]), 1)
            avg_eng = round(mean([f.get("engagement_score", 0) for f in window]), 1)
            words = sum(len((f.get("transcript_chunk") or "").split()) for f in window)
            fillers = sum(len(f.get("flagged_phrases", [])) for f in window)
            per_q.append({
                "question_id": qid,
                "text": q["text"],
                "category": q["category"],
                "difficulty": q["difficulty"],
                "target_seconds": q["target_seconds"],
                "answered_seconds": round(end - start, 1),
                "avg_confidence": avg_conf,
                "avg_engagement": avg_eng,
                "word_count": words,
                "filler_count": fillers,
                "self_rating": ev.get("self_rating"),
            })
        return per_q

    def _compute_face_dynamics(self, sess: LiveSession) -> dict:
        if not sess.face_ticks:
            return {"available": False, "ticks": []}
        from statistics import mean, pstdev
        yaw = [t.get("head_yaw", 0.0) for t in sess.face_ticks]
        pitch = [t.get("head_pitch", 0.0) for t in sess.face_ticks]
        eye = [t.get("eye_openness", 1.0) for t in sess.face_ticks]
        smile = [t.get("smile", 0.0) for t in sess.face_ticks]
        looking = [t.get("looking_at_camera", 1.0) for t in sess.face_ticks]
        return {
            "available": True,
            "tick_count": len(sess.face_ticks),
            "head_yaw_std": round(pstdev(yaw) if len(yaw) > 1 else 0.0, 3),
            "head_pitch_std": round(pstdev(pitch) if len(pitch) > 1 else 0.0, 3),
            "avg_eye_openness": round(mean(eye), 3),
            "avg_smile": round(mean(smile), 3),
            "looking_pct": round(100.0 * mean(looking), 1),
            "ticks": sess.face_ticks[-300:],
        }

    async def get_analysis(self, session_id: str) -> Optional[dict]:
        if self._db is None:
            return None
        cursor = await self._db.execute(
            "SELECT analysis_json FROM sessions WHERE id=?", (session_id,),
        )
        row = await cursor.fetchone()
        if not row or not row[0]:
            return None
        return json.loads(row[0])

    async def get_session_meta(self, session_id: str) -> Optional[dict]:
        if self._db is None:
            return None
        cursor = await self._db.execute(
            "SELECT id, context, started_at, ended_at, duration_seconds, avg_confidence, avg_engagement "
            "FROM sessions WHERE id=?",
            (session_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "context": row[1] or "",
            "started_at": row[2],
            "ended_at": row[3],
            "duration_seconds": row[4],
            "avg_confidence": row[5],
            "avg_engagement": row[6],
        }

    async def get_timeline(self, session_id: str) -> list[dict]:
        live = self._live.get(session_id)
        if live:
            return list(live.frames)
        if self._db is None:
            return []
        cursor = await self._db.execute(
            "SELECT payload_json FROM frames WHERE session_id=? ORDER BY seq ASC", (session_id,),
        )
        rows = await cursor.fetchall()
        return [json.loads(r[0]) for r in rows]

    async def list_finished(self, limit: int = 50) -> list[dict]:
        if self._db is None:
            return []
        cursor = await self._db.execute(
            "SELECT id, context, started_at, ended_at, duration_seconds, avg_confidence, avg_engagement "
            "FROM sessions WHERE ended_at IS NOT NULL ORDER BY ended_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0],
                "context": r[1] or "",
                "started_at": r[2],
                "ended_at": r[3],
                "duration_seconds": r[4] or 0.0,
                "avg_confidence": r[5] or 0.0,
                "avg_engagement": r[6] or 0.0,
            }
            for r in rows
        ]

    async def janitor(self) -> int:
        """Drop sessions older than TTL. Returns number deleted."""
        if self._db is None:
            return 0
        cutoff = time.time() - self.ttl_seconds
        cursor = await self._db.execute(
            "DELETE FROM sessions WHERE created_at < ?", (cutoff,),
        )
        deleted = cursor.rowcount
        await self._db.commit()
        if deleted:
            _log.info(f"janitor pruned {deleted} sessions older than {self.ttl_seconds}s")
        return deleted
