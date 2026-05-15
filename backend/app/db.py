"""SQLite transcript storage.

Every finalised translation segment is recorded so sessions can later be
exported for summaries or meeting minutes. Calls are synchronous; routers and
the websocket handler invoke them via `asyncio.to_thread`.
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from threading import Lock

from .config import DATA_DIR

DB_PATH = DATA_DIR / "transcripts.db"
_lock = Lock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _lock, _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id         TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                engine     TEXT NOT NULL,
                lang_a     TEXT NOT NULL,
                lang_b     TEXT NOT NULL,
                started_at INTEGER NOT NULL,
                ended_at   INTEGER
            );
            CREATE TABLE IF NOT EXISTS segments (
                id         TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                ts         INTEGER NOT NULL,
                lang_a     TEXT NOT NULL,
                lang_b     TEXT NOT NULL,
                text_a     TEXT NOT NULL,
                text_b     TEXT NOT NULL,
                spoken     TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_segments_session ON segments(session_id, ts);
            """
        )


def create_session(name: str, engine: str, lang_a: str, lang_b: str) -> str:
    session_id = f"sess-{uuid.uuid4().hex[:12]}"
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, name, engine, lang_a, lang_b, started_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, name, engine, lang_a, lang_b, int(time.time() * 1000)),
        )
    return session_id


def end_session(session_id: str) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
            (int(time.time() * 1000), session_id),
        )


def add_segment(session_id: str, seg: dict) -> None:
    """Insert or replace a finalised segment (segment_id stays stable across partials)."""
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO segments "
            "(id, session_id, ts, lang_a, lang_b, text_a, text_b, spoken) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                seg["segment_id"], session_id, seg["ts"],
                seg["lang_a"], seg["lang_b"],
                seg["text_a"], seg["text_b"], seg.get("spoken"),
            ),
        )


def list_sessions() -> list[dict]:
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT s.*, COUNT(seg.id) AS segment_count "
            "FROM sessions s LEFT JOIN segments seg ON seg.session_id = s.id "
            "GROUP BY s.id ORDER BY s.started_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> dict | None:
    with _lock, _connect() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        segs = conn.execute(
            "SELECT * FROM segments WHERE session_id = ? ORDER BY ts", (session_id,)
        ).fetchall()
    session = dict(row)
    session["segments"] = [dict(s) for s in segs]
    return session


def delete_session(session_id: str) -> None:
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM segments WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
