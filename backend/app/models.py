"""Locate and download the local models, so first-use downloads are not silent.

The analysis models are large (Whisper, NLLB, a 7B LLM, the diarization models)
and are fetched lazily on first use. To make that visible, the analysis
pipeline asks here what is missing and, if anything is, downloads it as an
explicit "downloading" phase the UI can show — instead of the operation just
appearing to hang.
"""
from __future__ import annotations

import logging
import os
import tarfile
import tempfile
import urllib.request

from .backends import asr, llm
from .backends.asr import _fw_name
from .backends.llm import _DEFAULT_GGUF, _MLX_TO_GGUF
from .config import DATA_DIR

_DIAR_DIR = DATA_DIR / "diarization"
_SEG_DIR = _DIAR_DIR / "sherpa-onnx-pyannote-segmentation-3-0"
_SEG_MODEL = _SEG_DIR / "model.int8.onnx"
_EMB_MODEL = _DIAR_DIR / "emb.onnx"

_SEG_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
_EMB_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/"
    "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
)


def _cached_hf_repos() -> set[str]:
    """Repo ids present in the local HuggingFace cache."""
    try:
        from huggingface_hub import scan_cache_dir
        return {r.repo_id for r in scan_cache_dir().repos}
    except Exception:  # noqa: BLE001 — treat an unreadable cache as "nothing cached"
        return set()


def _needed(settings, *, asr_: bool, nllb: bool, llm_: bool) -> list[tuple]:
    """Models a run needs, as (label, repo, filename | None).

    filename None → a whole HF repo; filename set → a single file in a repo.
    """
    local = settings.local
    items: list[tuple] = []
    if asr_:
        repo = (local.analyze_whisper_model if asr.backend() == "mlx"
                else f"Systran/faster-whisper-{_fw_name(local.analyze_whisper_model)}")
        items.append(("語音辨識模型 Whisper", repo, None))
    if nllb:
        items.append(("翻譯模型 NLLB", local.translate_model, None))
    if llm_:
        if llm.backend() == "mlx":
            items.append(("摘要模型 LLM", local.summary_model, None))
        else:
            repo, fname = _MLX_TO_GGUF.get(local.summary_model, _DEFAULT_GGUF)
            items.append(("摘要模型 LLM", repo, fname))
    return items


def _diarization_present() -> bool:
    return _EMB_MODEL.exists() and _SEG_MODEL.exists()


def missing(settings, *, asr_: bool, nllb: bool, llm_: bool, sherpa: bool) -> list[str]:
    """Human-readable names of the models a run needs that are not downloaded."""
    cached = _cached_hf_repos()
    names = [label for label, repo, _ in _needed(settings, asr_=asr_, nllb=nllb, llm_=llm_)
             if repo not in cached]
    if sherpa and not _diarization_present():
        names.append("講者辨識模型 sherpa-onnx")
    return names


def _download_diarization() -> None:
    """Fetch the sherpa-onnx speaker diarization models (segmentation + embedding)."""
    _DIAR_DIR.mkdir(parents=True, exist_ok=True)
    if not _EMB_MODEL.exists():
        logging.info("[models] downloading speaker embedding model")
        urllib.request.urlretrieve(_EMB_URL, _EMB_MODEL)
    if not _SEG_MODEL.exists():
        logging.info("[models] downloading speaker segmentation model")
        tmp = tempfile.NamedTemporaryFile(suffix=".tar.bz2", delete=False)
        tmp.close()
        try:
            urllib.request.urlretrieve(_SEG_URL, tmp.name)
            with tarfile.open(tmp.name, "r:bz2") as tar:
                tar.extractall(_DIAR_DIR)  # contains the segmentation dir
        finally:
            os.unlink(tmp.name)


def ensure(settings, *, asr_: bool, nllb: bool, llm_: bool, sherpa: bool) -> None:
    """Download whatever the run needs that is missing. Idempotent — anything
    already cached is skipped, so this is cheap to call every run."""
    from huggingface_hub import hf_hub_download, snapshot_download

    for label, repo, fname in _needed(settings, asr_=asr_, nllb=nllb, llm_=llm_):
        logging.info(f"[models] ensuring {label} ({repo})")
        if fname:  # a single file (e.g. one GGUF quant) — don't pull the whole repo
            hf_hub_download(repo, fname)
        else:
            snapshot_download(repo)
    if sherpa and not _diarization_present():
        _download_diarization()
