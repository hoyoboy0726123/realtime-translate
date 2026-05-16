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
        )
        return {"text": r.get("text", "").strip(), "language": r.get("language", "")}

    # faster-whisper (CTranslate2)
    from faster_whisper import WhisperModel
    name = _fw_name(model)
    if name not in _fw_models:
        logging.info(f"[asr] loading faster-whisper model: {name}")
        _fw_models[name] = WhisperModel(name, device="auto", compute_type="auto")
    segments, info = _fw_models[name].transcribe(samples, language=language)
    text = "".join(seg.text for seg in segments).strip()
    return {"text": text, "language": info.language}
