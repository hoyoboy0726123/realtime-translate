"""SQLite transcript storage.

Every finalised translation segment is recorded so sessions can later be
exported for summaries or meeting minutes. Calls are synchronous; routers and
the websocket handler invoke them via `asyncio.to_thread`.

A session may also be *analysed* after it ends: the recorded audio is
re-transcribed with speaker diarization into `diarized_segments`, and an LLM
summary is stored on the session.
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
    # timeout: the analysis subprocess and the backend share this DB file;
    # wait out a brief write lock instead of failing.
    conn = sqlite3.connect(DB_PATH, timeout=30)
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

            -- Post-processed, speaker-attributed transcript (one row per
            -- diarized utterance). Populated by the "analyze recording" job.
            CREATE TABLE IF NOT EXISTS diarized_segments (
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                idx        INTEGER NOT NULL,
                speaker    TEXT NOT NULL,
                start_ms   INTEGER NOT NULL,
                end_ms     INTEGER NOT NULL,
                text_a     TEXT NOT NULL,
                text_b     TEXT NOT NULL,
                PRIMARY KEY (session_id, idx)
            );
            """
        )
        # Migrate older databases: add the post-processing columns if absent.
        for col, decl in (
            ("audio_path", "TEXT"),
            ("process_status", "TEXT"),   # NULL | processing | done | failed
            ("processed_at", "INTEGER"),
            ("summary", "TEXT"),
        ):
            try:
                conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} {decl}")
            except sqlite3.OperationalError:
                pass  # column already exists


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


def set_audio_path(session_id: str, audio_path: str) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE sessions SET audio_path = ? WHERE id = ?",
            (audio_path, session_id),
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


# ---- post-processing (analyze recording) -------------------------------------

def set_process_status(session_id: str, status: str | None) -> None:
    """status: 'processing' | 'done' | 'failed' | None (not analysed)."""
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE sessions SET process_status = ? WHERE id = ?",
            (status, session_id),
        )


def save_diarized(session_id: str, segments: list[dict]) -> None:
    """Replace the diarized transcript for a session."""
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM diarized_segments WHERE session_id = ?", (session_id,))
        conn.executemany(
            "INSERT INTO diarized_segments "
            "(session_id, idx, speaker, start_ms, end_ms, text_a, text_b) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (session_id, i, s["speaker"], s["start_ms"], s["end_ms"],
                 s["text_a"], s["text_b"])
                for i, s in enumerate(segments)
            ],
        )


def save_summary(session_id: str, summary: str) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE sessions SET summary = ?, processed_at = ?, process_status = 'done' "
            "WHERE id = ?",
            (summary, int(time.time() * 1000), session_id),
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
        diar = conn.execute(
            "SELECT * FROM diarized_segments WHERE session_id = ? ORDER BY idx",
            (session_id,),
        ).fetchall()
    session = dict(row)
    session["segments"] = [dict(s) for s in segs]
    session["diarized"] = [dict(d) for d in diar]
    return session


def delete_session(session_id: str) -> None:
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM diarized_segments WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM segments WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


def reset_stuck_processing() -> None:
    """On startup, mark any analysis left mid-run (server restarted) as failed."""
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE sessions SET process_status = 'failed' "
            "WHERE process_status IN "
            "('processing', 'diarizing', 'translating', 'summarizing')"
        )
