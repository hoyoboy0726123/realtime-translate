"""Detect degenerate text — the repetition-loop hallucinations that both
Whisper (ASR) and NLLB (translation) can fall into on silence / odd input
("書書書…", "and a, and a, and a…").
"""
from __future__ import annotations


def is_degenerate(text: str) -> bool:
    """True if `text` looks like a repetition-loop hallucination."""
    t = text.strip()
    if len(t) < 40:
        return False  # too short to judge confidently
    compact = t.replace(" ", "")
    # Almost no distinct characters -> a single token repeated (e.g. 書書書…).
    if compact and len(set(compact)) / len(compact) < 0.12:
        return True
    # Almost no distinct words -> a short phrase looped (e.g. "and a, and a…").
    words = t.split()
    if len(words) >= 12 and len(set(words)) / len(words) < 0.25:
        return True
    return False
