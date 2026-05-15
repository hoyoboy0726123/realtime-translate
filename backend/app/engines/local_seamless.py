"""Local engine backed by Meta's Seamless models, running on-device.

Default implementation uses `SeamlessM4Tv2` (speech-to-text translation) from
HuggingFace transformers with energy-based chunking — it runs comfortably on an
Apple Silicon MacBook (MPS). For true word-level simultaneous output, swap in
the SeamlessStreaming agent (see `_run_inference` notes); the engine interface
below stays identical.

Heavy dependencies (torch / torchaudio / transformers / sentencepiece) are
imported lazily so the cloud and mock engines work without them installed.
See backend/requirements-local.txt.
"""
from __future__ import annotations

import asyncio

import numpy as np

from .. import languages
from ..config import SAMPLE_RATE
from .base import TranslationEngine, TranslationEvent, new_segment_id

# Chunking parameters.
_SILENCE_RMS = 380          # int16 RMS below this counts as silence
_SILENCE_HANG = 0.6         # seconds of trailing silence that closes an utterance
_MAX_UTTERANCE = 6.0        # hard cap so a long monologue still produces output
_MIN_UTTERANCE = 0.4        # ignore blips shorter than this


class LocalEngine(TranslationEngine):
    name = "local"

    def __init__(self, lang_a, lang_b, out_queue, settings):
        super().__init__(lang_a, lang_b, out_queue, settings)
        self._model = None
        self._processor = None
        self._device = settings.local.device
        self._buf = bytearray()
        self._silence_run = 0.0
        self._speech_seen = False
        self._busy = False

    async def open(self) -> None:
        await asyncio.to_thread(self._load_model)

    def _load_model(self) -> None:
        import torch
        from transformers import AutoProcessor, SeamlessM4Tv2Model

        model_id = self.settings.local.model
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = SeamlessM4Tv2Model.from_pretrained(model_id)
        if self._device == "mps" and not torch.backends.mps.is_available():
            self._device = "cpu"
        self._model = self._model.to(self._device)

    async def send_audio(self, pcm16: bytes) -> None:
        self._buf.extend(pcm16)

        samples = np.frombuffer(pcm16, dtype=np.int16)
        if samples.size:
            rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
            duration = samples.size / SAMPLE_RATE
            if rms < _SILENCE_RMS:
                self._silence_run += duration
            else:
                self._silence_run = 0.0
                self._speech_seen = True

        buffered = len(self._buf) / 2 / SAMPLE_RATE
        closed_by_silence = self._speech_seen and self._silence_run >= _SILENCE_HANG
        if (closed_by_silence or buffered >= _MAX_UTTERANCE) and not self._busy:
            await self._flush()

    async def close(self) -> None:
        if not self._busy and self._speech_seen:
            await self._flush()

    async def _flush(self) -> None:
        if len(self._buf) / 2 / SAMPLE_RATE < _MIN_UTTERANCE:
            self._reset_buffer()
            return

        audio = np.frombuffer(bytes(self._buf), dtype=np.int16).astype(np.float32) / 32768.0
        self._reset_buffer()
        self._busy = True
        seg = new_segment_id()
        try:
            text_a, text_b = await asyncio.to_thread(self._run_inference, audio)
        finally:
            self._busy = False

        if text_a or text_b:
            await self.emit(TranslationEvent(
                kind="final", segment_id=seg,
                lang_a=self.lang_a, lang_b=self.lang_b,
                text_a=text_a, text_b=text_b,
            ))

    def _reset_buffer(self) -> None:
        self._buf.clear()
        self._silence_run = 0.0
        self._speech_seen = False

    def _run_inference(self, audio: np.ndarray) -> tuple[str, str]:
        """Translate one utterance into both locked languages.

        SeamlessM4Tv2 auto-detects the spoken language from the audio, so
        decoding once per target language yields the transcription (target ==
        spoken language) and the translation in a single pass each.
        """
        return (
            self._decode(audio, languages.get(self.lang_a)["seamless"]),
            self._decode(audio, languages.get(self.lang_b)["seamless"]),
        )

    def _decode(self, audio: np.ndarray, tgt_lang: str) -> str:
        inputs = self._processor(
            audios=audio, sampling_rate=SAMPLE_RATE, return_tensors="pt",
        ).to(self._device)
        tokens = self._model.generate(**inputs, tgt_lang=tgt_lang, generate_speech=False)
        ids = tokens[0].tolist() if hasattr(tokens, "__getitem__") else tokens
        return self._processor.decode(ids[0], skip_special_tokens=True).strip()
