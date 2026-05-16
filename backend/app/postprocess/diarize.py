"""Offline speaker diarization + transcription of a recorded session.

sherpa-onnx (ONNX Runtime, no torch dependency) segments the recording into
speaker turns; each turn is then transcribed via the cross-platform ASR
backend in its own detected language, so a bilingual meeting is handled turn
by turn.

Returns one dict per utterance: {speaker, start_ms, end_ms, text, lang}.
"""
from __future__ import annotations

import logging
import wave

import numpy as np

from ..backends import asr
from ..config import DATA_DIR

_DIAR_DIR = DATA_DIR / "diarization"
_SEG_MODEL = _DIAR_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
_EMB_MODEL = _DIAR_DIR / "emb.onnx"

# Diarization turns shorter than this are skipped (too little to transcribe).
_MIN_SEG_SEC = 0.4
# Adjacent turns from the same speaker within this gap are merged, so a single
# utterance isn't chopped into fragments before transcription.
_MERGE_GAP_SEC = 1.2


def _merge_turns(turns) -> list[dict]:
    """Merge consecutive same-speaker turns separated by only a short gap."""
    merged: list[dict] = []
    for t in turns:
        if (merged and merged[-1]["speaker"] == t.speaker
                and t.start - merged[-1]["end"] <= _MERGE_GAP_SEC):
            merged[-1]["end"] = t.end
        else:
            merged.append({"speaker": t.speaker, "start": t.start, "end": t.end})
    return merged


def _build_diarizer():
    import sherpa_onnx

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(_SEG_MODEL),
            ),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(_EMB_MODEL)),
        # num_clusters=-1 → auto-detect the speaker count via the threshold.
        # Lower threshold = more speakers split apart; 0.4 separates a small
        # meeting better than the 0.5 default (which tends to merge speakers).
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.4),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError(
            "Diarization models missing — expected "
            f"{_SEG_MODEL} and {_EMB_MODEL}"
        )
    return sherpa_onnx.OfflineSpeakerDiarization(config)


def _read_wav(wav_path: str) -> tuple[np.ndarray, int]:
    """Read a mono PCM16 WAV as float32 samples in [-1, 1]."""
    with wave.open(str(wav_path), "rb") as w:
        sample_rate = w.getframerate()
        pcm = w.readframes(w.getnframes())
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate


def diarize_and_transcribe(wav_path: str, whisper_model: str) -> list[dict]:
    """Diarize a recording and transcribe each speaker turn.

    Blocking / compute-bound — run via asyncio.to_thread.
    """
    samples, sample_rate = _read_wav(wav_path)
    if len(samples) < sample_rate:  # under ~1 s of audio
        return []

    sd = _build_diarizer()
    turns = _merge_turns(sd.process(samples).sort_by_start_time())

    utterances: list[dict] = []
    for turn in turns:
        if turn["end"] - turn["start"] < _MIN_SEG_SEC:
            continue
        clip = samples[int(turn["start"] * sample_rate):int(turn["end"] * sample_rate)]
        try:
            tr = asr.transcribe(clip, whisper_model)
        except Exception as exc:  # noqa: BLE001
            logging.error(f"[diarize] transcribe failed for a turn: {exc}")
            continue
        text = tr["text"]
        if not text:
            continue
        utterances.append({
            "speaker": f"Speaker {turn['speaker'] + 1}",
            "start_ms": int(turn["start"] * 1000),
            "end_ms": int(turn["end"] * 1000),
            "text": text,
            "lang": tr["language"],
        })
    return utterances
