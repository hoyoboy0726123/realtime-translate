"""Transcript history API: browse, analyse, play, export and delete sessions."""
from __future__ import annotations

import asyncio
import datetime as dt
import os
from concurrent.futures import ProcessPoolExecutor

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from .. import db
from ..postprocess.analyze import analyze_session

router = APIRouter(prefix="/api/transcripts", tags=["transcripts"])

# process_status values that mean analysis is still running.
_PROCESSING = {"processing", "diarizing", "translating", "summarizing"}

# The analysis pipeline is heavy (Whisper + NLLB + a 7B LLM). It runs in a
# separate *process* so its compute never starves the backend's event loop or
# blocks live translation. One worker — analyses run one at a time.
_analyze_pool: ProcessPoolExecutor | None = None


def _pool() -> ProcessPoolExecutor:
    global _analyze_pool
    if _analyze_pool is None:
        _analyze_pool = ProcessPoolExecutor(max_workers=1)
    return _analyze_pool


def _fmt_ts(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M:%S")


@router.get("")
async def list_transcripts() -> dict:
    sessions = await asyncio.to_thread(db.list_sessions)
    return {"sessions": sessions}


@router.get("/{session_id}")
async def get_transcript(session_id: str) -> dict:
    session = await asyncio.to_thread(db.get_session, session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    return session


@router.get("/{session_id}/audio")
async def get_audio(session_id: str):
    """Stream the session's recorded audio (for the in-browser player)."""
    session = await asyncio.to_thread(db.get_session, session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    audio_path = session.get("audio_path")
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(404, "No recording for this session")
    return FileResponse(audio_path, media_type="audio/wav")


@router.post("/{session_id}/analyze")
async def analyze(session_id: str) -> dict:
    """Start post-session analysis (diarization + translation + summary).

    Runs in the background; poll GET /{session_id} for `process_status` —
    `diarizing` -> `translating` -> `summarizing` -> `done` (or `failed`).
    """
    session = await asyncio.to_thread(db.get_session, session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    audio_path = session.get("audio_path")
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(400, "No recording available for this session")
    if session.get("process_status") in _PROCESSING:
        return {"status": session["process_status"]}

    await asyncio.to_thread(db.set_process_status, session_id, "diarizing")
    # Run the heavy pipeline in a separate process, fire-and-forget — the
    # backend stays responsive while it runs.
    asyncio.get_running_loop().run_in_executor(_pool(), analyze_session, session_id)
    return {"status": "diarizing"}


@router.get("/{session_id}/export.md", response_class=PlainTextResponse)
async def export_markdown(session_id: str) -> str:
    """Markdown export, ready to drop into meeting notes or feed to a summariser."""
    session = await asyncio.to_thread(db.get_session, session_id)
    if session is None:
        raise HTTPException(404, "Session not found")

    lines = [
        f"# {session['name']}",
        "",
        f"- Engine: `{session['engine']}`",
        f"- Languages: {session['lang_a']} / {session['lang_b']}",
        f"- Started: {_fmt_ts(session['started_at'])}",
        "",
        "## Transcript",
        "",
    ]
    for seg in session["segments"]:
        lines.append(f"**[{_fmt_ts(seg['ts'])}]**")
        lines.append(f"- ({seg['lang_a']}) {seg['text_a']}")
        lines.append(f"- ({seg['lang_b']}) {seg['text_b']}")
        lines.append("")
    return "\n".join(lines)


@router.get("/{session_id}/export.json")
async def export_json(session_id: str) -> dict:
    session = await asyncio.to_thread(db.get_session, session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    return session


@router.delete("/{session_id}")
async def delete_transcript(session_id: str) -> dict:
    """Delete a session, its segments, and its recording file."""
    session = await asyncio.to_thread(db.get_session, session_id)
    if session and session.get("audio_path"):
        try:
            os.remove(session["audio_path"])
        except OSError:
            pass  # already gone — fine
    await asyncio.to_thread(db.delete_session, session_id)
    return {"deleted": session_id}
