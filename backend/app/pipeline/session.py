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

CREATE INDEX IF NOT EXISTS idx_frames_session ON frames(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_ended ON sessions(ended_at);
"""


@dataclass
class LiveSession:
    id: str
    context: str
    started_at: float
    frames: list[dict] = field(default_factory=list)
    seq: int = 0


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

    async def end(self, session_id: str) -> Optional[dict]:
        sess = self._live.pop(session_id, None)
        if not sess:
            return None
        ended_at = time.time()
        analysis = analyse_session(sess.frames, sess.started_at, ended_at)
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
            await self._db.commit()
        return analysis

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
