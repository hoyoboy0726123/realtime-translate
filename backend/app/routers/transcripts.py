"""Transcript history API: browse, analyse, export and delete recorded sessions."""
from __future__ import annotations

import asyncio
import datetime as dt
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from .. import db
from ..postprocess.analyze import analyze_session

router = APIRouter(prefix="/api/transcripts", tags=["transcripts"])


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


@router.post("/{session_id}/analyze")
async def analyze(session_id: str) -> dict:
    """Start post-session analysis (diarization + translation + summary).

    Runs in the background; poll GET /{session_id} for `process_status` and the
    resulting `diarized` transcript / `summary`.
    """
    session = await asyncio.to_thread(db.get_session, session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    audio_path = session.get("audio_path")
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(400, "No recording available for this session")
    if session.get("process_status") == "processing":
        return {"status": "processing"}

    await asyncio.to_thread(db.set_process_status, session_id, "processing")
    # Run the (long, compute-bound) pipeline in a worker thread, fire-and-forget.
    asyncio.create_task(asyncio.to_thread(analyze_session, session_id))
    return {"status": "processing"}


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
    await asyncio.to_thread(db.delete_session, session_id)
    return {"deleted": session_id}
