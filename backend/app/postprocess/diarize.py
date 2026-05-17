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
# Use the int8 segmentation model: the fp32 `model.onnx` triggers a SIGBUS
# inside sherpa-onnx's pyannote path on Apple Silicon for audio longer than
# ~10 s. The int8 model runs the full recording reliably (and faster), with
# negligible difference in segmentation quality.
_SEG_MODEL = _DIAR_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.int8.onnx"
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


def transcribe_only(wav_path: str, whisper_model: str) -> list[dict]:
    """Transcribe a recording without speaker diarization.

    A single Whisper pass over the whole file; utterances are its sentence
    segments, all with an empty speaker label. Faster than diarization and
    used when the caller opts out of identifying speakers.

    Blocking / compute-bound — run via asyncio.to_thread.
    """
    samples, sample_rate = _read_wav(wav_path)
    if len(samples) < sample_rate:  # under ~1 s of audio
        return []

    tr = asr.transcribe(samples, whisper_model)
    if not tr["text"]:
        return []

    lang = tr["language"]
    segs = tr.get("segments") or []
    if not segs:  # no sentence breaks — keep the whole thing as one utterance
        return [{
            "speaker": "", "start_ms": 0,
            "end_ms": int(len(samples) / sample_rate * 1000),
            "text": tr["text"], "lang": lang,
        }]

    utterances: list[dict] = []
    for s in segs:
        text = s["text"].strip()
        if not text:
            continue
        start, end = s["start"], s["end"]
        if end <= start:
            end = start + _MIN_SEG_SEC
        utterances.append({
            "speaker": "",
            "start_ms": int(start * 1000),
            "end_ms": int(end * 1000),
            "text": text,
            "lang": lang,
        })
    return utterances


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
        if not tr["text"]:
            continue

        speaker = f"Speaker {turn['speaker'] + 1}"
        lang = tr["language"]
        # Emit one utterance per Whisper sentence segment so a long turn
        # becomes several timestamped lines instead of one wall of text.
        # Segment times are relative to the clip — offset by the turn start.
        segs = tr.get("segments") or []
        if segs:
            for s in segs:
                text = s["text"].strip()
                if not text:
                    continue
                start = turn["start"] + s["start"]
                end = min(turn["start"] + s["end"], turn["end"])
                if end <= start:
                    end = start + _MIN_SEG_SEC
                utterances.append({
                    "speaker": speaker,
                    "start_ms": int(start * 1000),
                    "end_ms": int(end * 1000),
                    "text": text,
                    "lang": lang,
                })
        else:
            utterances.append({
                "speaker": speaker,
                "start_ms": int(turn["start"] * 1000),
                "end_ms": int(turn["end"] * 1000),
                "text": tr["text"],
                "lang": lang,
            })
    return utterances
