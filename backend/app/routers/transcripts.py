"""Transcript history API: browse, analyse, play, export and delete sessions."""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import multiprocessing
import os
import shutil
import subprocess
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from .. import db
from ..config import RECORDINGS_DIR, SAMPLE_RATE
from ..postprocess.analyze import analyze_session

router = APIRouter(prefix="/api/transcripts", tags=["transcripts"])

# process_status values that mean analysis is still running.
_PROCESSING = {"processing", "diarizing", "translating", "summarizing"}

# The analysis pipeline is heavy (Whisper + NLLB + a 7B LLM). Each run gets its
# own process, so its compute never starves the backend's event loop, its
# models are fully released when it exits, and — crucially — it can be
# force-stopped midway. Only one analysis runs at a time.
_mp = multiprocessing.get_context("spawn")
_current: dict | None = None    # the running analysis: {"session_id", "process"}
_cancelled: set[str] = set()    # sessions whose run was deliberately stopped


def _analysis_running() -> bool:
    return _current is not None and _current["process"].is_alive()


async def _watch(session_id: str, proc) -> None:
    """Wait for the analysis process; if it crashed, flag the session failed."""
    global _current
    await asyncio.get_running_loop().run_in_executor(None, proc.join)
    if _current is not None and _current["session_id"] == session_id:
        _current = None
    if session_id in _cancelled:
        _cancelled.discard(session_id)
        return  # the cancel handler already cleaned up after this run
    if proc.exitcode not in (0, None):
        # The process died without `analyze_session`'s own try/except running
        # (a hard crash) — don't let the session sit stuck on "diarizing".
        logging.error(
            f"[analyze] {session_id}: process crashed (exit {proc.exitcode})"
        )
        session = await asyncio.to_thread(db.get_session, session_id)
        if session and session.get("process_status") in _PROCESSING:
            await asyncio.to_thread(db.set_process_status, session_id, "failed")


def _start_analysis(
    session_id: str,
    stage: str = "summary",
    diarize: bool = True,
    *,
    prior_status: str | None = None,
    had_transcript: bool = False,
) -> None:
    """Spawn the analysis pipeline in its own process and watch it finish.

    `prior_status` / `had_transcript` snapshot the session's state before this
    run, so a cancel can restore it instead of discarding a valid transcript.
    Must be called from within the running event loop."""
    global _current
    proc = _mp.Process(
        target=analyze_session, args=(session_id, stage, diarize),
    )
    proc.start()
    _current = {
        "session_id": session_id,
        "process": proc,
        "prior_status": prior_status,
        "had_transcript": had_transcript,
    }
    asyncio.create_task(_watch(session_id, proc))


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
async def analyze(
    session_id: str, stage: str = "summary", diarize: bool = True,
) -> dict:
    """Start post-session analysis. Each stage is usable on its own:

      stage=transcript — diarization + transcription + translation only.
      stage=summary    — also the LLM summary; reuses an existing transcript.

    `diarize=false` skips speaker identification (faster, no speaker labels).

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
    if _analysis_running():
        raise HTTPException(409, "另一個分析正在進行中，請待其完成或先停止它")

    # If we only need the summary and a transcript already exists, the run
    # skips straight to the summary step — reflect that in the initial status.
    prior_status = session.get("process_status")
    has_transcript = bool(session.get("diarized"))
    initial = "summarizing" if (stage == "summary" and has_transcript) else "diarizing"
    await asyncio.to_thread(db.set_process_status, session_id, initial)
    # Run the heavy pipeline in a separate process — the backend stays
    # responsive while it runs.
    _start_analysis(
        session_id, stage, diarize,
        prior_status=prior_status, had_transcript=has_transcript,
    )
    return {"status": initial}


@router.post("/{session_id}/cancel")
async def cancel_analysis(session_id: str) -> dict:
    """Stop an in-progress analysis and discard its partial results.

    Force-kills the analysis process, removes any partial diarized transcript
    written so far, and resets the session to "not analysed".
    """
    global _current
    session = await asyncio.to_thread(db.get_session, session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    if session.get("process_status") not in _PROCESSING:
        return {"status": session.get("process_status")}

    # Mark it so `_watch` knows this exit was a deliberate stop, not a crash.
    _cancelled.add(session_id)

    run = _current if (_current and _current["session_id"] == session_id) else None
    if run is not None:
        proc = run["process"]
        if proc.is_alive():
            proc.terminate()
            await asyncio.to_thread(proc.join, 5)
            if proc.is_alive():  # didn't stop gracefully — force it
                proc.kill()
                await asyncio.to_thread(proc.join)
        _current = None

    # A run that started from an existing transcript (summary-only) kept that
    # transcript valid — restore it. Otherwise the transcript was being built
    # and is incomplete: drop it and reset the session to "not analysed".
    if run is not None and run["had_transcript"]:
        await asyncio.to_thread(
            db.set_process_status, session_id, run["prior_status"] or "done",
        )
    else:
        await asyncio.to_thread(db.save_diarized, session_id, [])
        await asyncio.to_thread(db.set_process_status, session_id, None)
    return {"status": "cancelled"}


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


@router.get("/{session_id}/export.txt", response_class=PlainTextResponse)
async def export_txt(session_id: str, track: str = "both") -> PlainTextResponse:
    """Plain-text transcript of the analysed session — one utterance per block,
    no timestamps. `track`: `both`, `a`, or `b`. Requires analysis first.
    """
    session = await asyncio.to_thread(db.get_session, session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    diar = session.get("diarized") or []
    if not diar:
        raise HTTPException(400, "尚未分析此記錄，無法匯出逐字稿")
    if track not in ("both", "a", "b"):
        track = "both"

    blocks: list[str] = []
    for d in diar:
        if track == "a":
            lines = [d["text_a"]]
        elif track == "b":
            lines = [d["text_b"]]
        else:
            lines = [d["text_a"], d["text_b"]]
        lines = [ln.strip() for ln in lines if ln and ln.strip()]
        if lines:
            blocks.append("\n".join(lines))

    body = "\n\n".join(blocks) + "\n" if blocks else ""
    return PlainTextResponse(
        body,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{session_id}.txt"'},
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
