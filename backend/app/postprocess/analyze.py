"""Orchestrate post-session analysis of a recording.

Pipeline (each step loads its own models, run sequentially to bound memory):
  1. diarize + transcribe   — sherpa-onnx + MLX-Whisper
  2. translate              — NLLB
  3. summarize              — local LLM (mlx-lm)

Blocking / compute-bound — call via asyncio.to_thread. Progress is reflected in
the session's `process_status` column (processing | done | failed).
"""
from __future__ import annotations

import logging

from .. import db
from ..config import load_settings
from .diarize import diarize_and_transcribe
from .summarize import summarize
from .translate import translate_utterances


def analyze_session(session_id: str) -> None:
    """Run the full analysis pipeline for one recorded session."""
    session = db.get_session(session_id)
    if session is None or not session.get("audio_path"):
        logging.error(f"[analyze] {session_id}: no recording to analyse")
        db.set_process_status(session_id, "failed")
        return

    settings = load_settings()
    lang_a, lang_b = session["lang_a"], session["lang_b"]
    local = settings.local

    try:
        # 1. diarize + transcribe (full large-v3 — offline, accuracy over speed)
        db.set_process_status(session_id, "diarizing")
        utterances = diarize_and_transcribe(
            session["audio_path"], local.analyze_whisper_model,
        )
        if not utterances:
            logging.error(f"[analyze] {session_id}: nothing transcribed")
            db.set_process_status(session_id, "failed")
            return

        # 2. translate into both locked languages
        db.set_process_status(session_id, "translating")
        diarized = translate_utterances(
            utterances, lang_a, lang_b,
            local.translate_model, settings.chinese_variant,
        )
        db.save_diarized(session_id, diarized)

        # 3. summarize — feed the lang_a column to the LLM
        db.set_process_status(session_id, "summarizing")
        summary = summarize(diarized, "text_a", local.summary_model)
        if settings.chinese_variant == "traditional" and "zh" in (lang_a, lang_b):
            import opencc
            summary = opencc.OpenCC("s2twp").convert(summary)

        db.save_summary(session_id, summary)  # also sets status = "done"
        logging.info(f"[analyze] {session_id}: done ({len(diarized)} utterances)")
    except Exception as exc:  # noqa: BLE001
        logging.error(f"[analyze] {session_id} failed: {exc}")
        db.set_process_status(session_id, "failed")
