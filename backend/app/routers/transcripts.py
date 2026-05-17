"""Transcript history API: browse, analyse, play, export and delete sessions."""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from .. import db
from ..config import RECORDINGS_DIR, SAMPLE_RATE
from ..postprocess.analyze import analyze_session

router = APIRouter(prefix="/api/transcripts", tags=["transcripts"])

# process_status values that mean analysis is still running.
_PROCESSING = {"processing", "diarizing", "translating", "summarizing"}

# The analysis pipeline is heavy (Whisper + NLLB + a 7B LLM). It runs in a
# separate *process* so its compute never starves the backend's event loop or
# blocks live translation. One worker — analyses run one at a time.
#
# `max_tasks_per_child=1`: the worker is torn down after every analysis, so the
# Whisper/NLLB/LLM models it loaded are fully released. Without this the worker
# is reused and memory from each run accumulates until it is OOM-killed mid-run.
_analyze_pool: ProcessPoolExecutor | None = None


def _pool() -> ProcessPoolExecutor:
    global _analyze_pool
    if _analyze_pool is None:
        _analyze_pool = ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1)
    return _analyze_pool


def _on_analyze_done(session_id: str, future) -> None:
    """If the analysis worker crashed, the future carries the exception that
    `analyze_session`'s own try/except never saw — mark the session failed so
    it does not sit stuck on `diarizing` forever, and drop the broken pool."""
    global _analyze_pool
    try:
        exc = future.exception()
    except Exception:  # cancelled
        exc = None
    if exc is not None:
        logging.error(f"[analyze] {session_id}: worker crashed: {exc!r}")
        try:
            db.set_process_status(session_id, "failed")
        except Exception as db_exc:  # noqa: BLE001
            logging.error(f"[analyze] {session_id}: could not mark failed: {db_exc!r}")
        _analyze_pool = None  # a crashed worker breaks the whole pool


def _start_analysis(session_id: str, stage: str = "summary") -> None:
    """Kick off the analysis pipeline in the worker process (fire-and-forget).

    Must be called from within the running event loop."""
    future = asyncio.get_running_loop().run_in_executor(
        _pool(), analyze_session, session_id, stage,
    )
    future.add_done_callback(lambda f: _on_analyze_done(session_id, f))


def _fmt_ts(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M:%S")


def _srt_ts(ms: int) -> str:
    """Format milliseconds as an SRT timestamp: HH:MM:SS,mmm."""
    ms = max(0, int(ms))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _decode_to_wav(src_path: str, dst_path: str) -> tuple[bool, str]:
    """Decode any ffmpeg-readable media file to 16 kHz mono PCM16 WAV.

    Returns (ok, error_message). Blocking — call via asyncio.to_thread."""
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", src_path,
         "-vn", "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
         dst_path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        return False, (tail[-1] if tail else "ffmpeg failed")
    return True, ""


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
async def analyze(session_id: str, stage: str = "summary") -> dict:
    """Start post-session analysis. Each stage is usable on its own:

      stage=transcript — diarization + transcription + translation only.
      stage=summary    — also the LLM summary; reuses an existing transcript.

    Runs in the background; poll GET /{session_id} for `process_status` —
    `diarizing` -> `translating` -> `summarizing` -> `done` (or `failed`).
    """
    if stage not in ("transcript", "summary"):
        stage = "summary"
    session = await asyncio.to_thread(db.get_session, session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    audio_path = session.get("audio_path")
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(400, "No recording available for this session")
    if session.get("process_status") in _PROCESSING:
        return {"status": session["process_status"]}

    # If we only need the summary and a transcript already exists, the run
    # skips straight to the summary step — reflect that in the initial status.
    has_transcript = bool(session.get("diarized"))
    initial = "summarizing" if (stage == "summary" and has_transcript) else "diarizing"
    await asyncio.to_thread(db.set_process_status, session_id, initial)
    # Run the heavy pipeline in a separate process — the backend stays
    # responsive while it runs.
    _start_analysis(session_id, stage)
    return {"status": initial}


@router.post("/upload")
async def upload_media(file: UploadFile = File(...)) -> dict:
    """Upload an arbitrary audio/video file and decode it into a session.

    Any ffmpeg-readable format works (mp3, m4a, wav, flac, mp4, mov, mkv …).
    The file becomes a session but is *not* analysed yet — the caller chooses
    which stage to run via POST /{session_id}/analyze?stage=...
    """
    name = (file.filename or "uploaded-file").strip() or "uploaded-file"
    session_id = db.create_session(name, "upload", "zh", "en")
    db.end_session(session_id)
    out_path = RECORDINGS_DIR / f"{session_id}.wav"

    # Spool the upload to a temp file, then let ffmpeg decode it.
    suffix = os.path.splitext(name)[1] or ".bin"
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            await asyncio.to_thread(shutil.copyfileobj, file.file, tmp)
        ok, err = await asyncio.to_thread(_decode_to_wav, tmp_path, str(out_path))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not ok:
        await asyncio.to_thread(db.delete_session, session_id)
        if out_path.exists():
            out_path.unlink()
        raise HTTPException(400, f"無法解析此檔案（需為音訊或影片）：{err}")

    await asyncio.to_thread(db.set_audio_path, session_id, str(out_path))
    return {"session_id": session_id, "status": "ready"}


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


@router.get("/{session_id}/export.srt", response_class=PlainTextResponse)
async def export_srt(session_id: str, track: str = "both") -> PlainTextResponse:
    """SRT subtitle export of the analysed (diarized) transcript.

    `track`: `both` (lang_a then lang_b stacked), `a`, or `b`.
    Requires the session to have been analysed first.
    """
    session = await asyncio.to_thread(db.get_session, session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    diar = session.get("diarized") or []
    if not diar:
        raise HTTPException(400, "尚未分析此記錄，無法匯出 SRT")
    if track not in ("both", "a", "b"):
        track = "both"

    cues: list[str] = []
    for d in diar:
        if track == "a":
            lines = [d["text_a"]]
        elif track == "b":
            lines = [d["text_b"]]
        else:
            lines = [d["text_a"], d["text_b"]]
        lines = [ln.strip() for ln in lines if ln and ln.strip()]
        if not lines:
            continue
        start, end = d["start_ms"], d["end_ms"]
        if end <= start:
            end = start + 1500  # ensure a visible, non-zero cue duration
        cues.append(
            f"{len(cues) + 1}\n"
            f"{_srt_ts(start)} --> {_srt_ts(end)}\n"
            + "\n".join(lines)
        )

    body = "\n\n".join(cues) + "\n" if cues else ""
    return PlainTextResponse(
        body,
        media_type="application/x-subrip",
        headers={"Content-Disposition": f'attachment; filename="{session_id}.srt"'},
    )


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
