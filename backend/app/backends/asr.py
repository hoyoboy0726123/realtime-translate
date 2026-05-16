"""Cross-platform speech recognition.

Apple Silicon  -> MLX-Whisper (fast, native).
Windows/Linux  -> faster-whisper (CTranslate2; CUDA or CPU).

The backend is auto-detected. Override with env var TRANSLATE_LOCAL_BACKEND
set to "mlx" or "ct2".
"""
from __future__ import annotations

import logging
import os

import numpy as np

# faster-whisper model names keyed by the MLX repo stored in settings, so the
# same settings.json works on both platforms.
_MLX_TO_FW = {
    "mlx-community/whisper-large-v3-turbo": "large-v3-turbo",
    "mlx-community/whisper-large-v3-mlx": "large-v3",
    "mlx-community/whisper-large-v3": "large-v3",
    "mlx-community/whisper-medium-mlx": "medium",
    "mlx-community/whisper-small-mlx": "small",
}

_fw_models: dict = {}   # cache of loaded faster-whisper models


def backend() -> str:
    """Return the active ASR backend: 'mlx' or 'ct2'."""
    forced = os.getenv("TRANSLATE_LOCAL_BACKEND", "").lower()
    if forced in ("mlx", "ct2"):
        return forced
    try:
        import mlx_whisper  # noqa: F401
        return "mlx"
    except ImportError:
        return "ct2"


def _fw_name(model: str) -> str:
    """Translate a settings model name to a faster-whisper model name."""
    if model in _MLX_TO_FW:
        return _MLX_TO_FW[model]
    # Otherwise assume it is already a faster-whisper name or a CT2 repo id.
    return model


def transcribe(samples: np.ndarray, model: str, language: str | None = None) -> dict:
    """Transcribe float32 16 kHz mono audio. Returns {text, language}.

    Blocking / compute-bound.
    """
    if backend() == "mlx":
        import mlx_whisper
        r = mlx_whisper.transcribe(
            samples, path_or_hf_repo=model, language=language, verbose=False,
            # Don't feed Whisper its own previous output back in — that is what
            # turns an occasional slip into an endless repetition loop.
            condition_on_previous_text=False,
        )
        text, lang = r.get("text", "").strip(), r.get("language", "")
    else:
        # faster-whisper (CTranslate2)
        from faster_whisper import WhisperModel
        name = _fw_name(model)
        if name not in _fw_models:
            logging.info(f"[asr] loading faster-whisper model: {name}")
            _fw_models[name] = WhisperModel(name, device="auto", compute_type="auto")
        segments, info = _fw_models[name].transcribe(
            samples, language=language, condition_on_previous_text=False,
        )
        text, lang = "".join(seg.text for seg in segments).strip(), info.language

    # Safety net: Whisper can still hallucinate a repetition loop ("書書書…",
    # "and a, and a, …") on silence or noise — drop a degenerate transcript.
    if _is_degenerate(text):
        return {"text": "", "language": lang}
    return {"text": text, "language": lang}


def _is_degenerate(text: str) -> bool:
    """True if `text` looks like a Whisper repetition-loop hallucination."""
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
