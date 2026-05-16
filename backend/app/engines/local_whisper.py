"""On-device engine: Whisper speech recognition + **NLLB** translation.

Speech recognition runs through the cross-platform ASR backend — MLX-Whisper on
Apple Silicon, faster-whisper (CTranslate2) on Windows/Linux. (SeamlessStreaming
was dropped: numerically broken on Apple MPS, ~26x too slow on CPU.)

Pipeline, per utterance:
  1. Buffer 16 kHz PCM16 audio; an utterance ends after trailing silence.
  2. Whisper transcribes the buffer -> source text + detected language.
  3. NLLB translates the source text into the other locked language.
  4. Emit `text_a` / `text_b` so the spoken language and its translation each
     land in the correct (language-fixed) pane.

While the speaker is still talking we re-transcribe the growing buffer every
~2 s and emit a `partial`; on silence we emit the `final`.

Transcription and NLLB's `model.generate` are blocking, compute-bound calls, so
all model work runs on a dedicated worker thread; events are handed back to the
asyncio loop with `loop.call_soon_threadsafe`.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time

import numpy as np

from .. import languages
from ..backends import asr
from ..config import SAMPLE_RATE
from ..nllb import NllbTranslator
from .base import TranslationEngine, TranslationEvent, new_segment_id

# Utterance segmentation.
_BLOCK_MS = 320
_SILENCE_RMS = 380.0        # int16 RMS below this counts as silence
_SILENCE_HANG = 0.8         # trailing silence (s) that closes an utterance
_PARTIAL_EVERY = 2.0        # re-transcribe + emit a partial this often (s)
_MIN_UTTERANCE = 0.8        # don't transcribe buffers shorter than this (s)
_MAX_UTTERANCE = 15.0       # force-finalize an utterance this long (keeps cost bounded)
# If this many audio blocks are already queued, the worker has fallen behind —
# skip the (optional) partial transcription so it can catch up. Each block is
# _BLOCK_MS, so 9 blocks ~= 2.9 s of backlog.
_MAX_BACKLOG_BLOCKS = 9

_FLUSH = object()           # sentinel pushed onto the audio queue on close


class WhisperEngine(TranslationEngine):
    name = "local"

    def __init__(self, lang_a, lang_b, out_queue, settings):
        super().__init__(lang_a, lang_b, out_queue, settings)
        self._whisper_model = settings.local.whisper_model
        self._translator = NllbTranslator(settings.local.translate_model)
        # FLORES-200 codes for the two locked languages.
        self._nllb_a = languages.get(lang_a)["nllb"]
        self._nllb_b = languages.get(lang_b)["nllb"]
        self._block_bytes = int(_BLOCK_MS * SAMPLE_RATE / 1000) * 2
        self._block_dur = self._block_bytes / 2 / SAMPLE_RATE
        self._audio_q: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # Whisper/NLLB emit Simplified Chinese; convert to Traditional when
        # requested. `_cc` is the OpenCC converter, built in _build().
        self._zh_variant = settings.chinese_variant
        self._cc = None

    async def open(self) -> None:
        self._loop = asyncio.get_running_loop()
        # Loading NLLB + warming up Whisper is heavy; keep the loop responsive.
        await asyncio.to_thread(self._build)
        self._worker = threading.Thread(target=self._run, name="whisper-stream", daemon=True)
        self._worker.start()

    def _build(self) -> None:
        self._translator.build()
        # OpenCC converter for Simplified -> Traditional (Taiwan, with phrase
        # conversion), only if a Chinese pane needs it.
        if self._zh_variant == "traditional" and "zh" in (self.lang_a, self.lang_b):
            import opencc
            self._cc = opencc.OpenCC("s2twp")
        # Warm up Whisper: first call downloads + loads the model. Feed 1 s of
        # silence so the first real utterance isn't penalised by the load.
        t0 = time.time()
        asr.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), self._whisper_model)
        logging.info(f"[whisper] models ready in {time.time()-t0:.1f}s")

    def _zh(self, text: str, lang: str) -> str:
        """Convert a Chinese pane's text to the configured script variant."""
        if lang == "zh" and self._cc is not None and text:
            return self._cc.convert(text)
        return text

    async def send_audio(self, pcm16: bytes) -> None:
        self._audio_q.put(pcm16)

    async def close(self) -> None:
        self._audio_q.put(_FLUSH)
        if self._worker:
            await asyncio.to_thread(self._worker.join, 20)

    # ---- worker thread ------------------------------------------------------

    def _transcribe(self, pcm: bytes, lang_hint: str | None) -> tuple[str, str]:
        """Return (text, detected_lang_code) for a PCM16 buffer."""
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        result = asr.transcribe(samples, self._whisper_model, language=lang_hint)
        return result["text"], result["language"]

    def _resolve_spoken(self, detected: str) -> str:
        """Map Whisper's detected language onto 'a' or 'b' of the locked pair."""
        if detected == self.lang_b:
            return "b"
        # Default to lang_a — one of the pair must be it, and lang_a is the
        # most likely when detection is ambiguous.
        return "a"

    def _emit(self, kind: str, seg_id: str, text: str, detected: str) -> None:
        """Transcript is `text` in language `detected`; translate to the other."""
        spoken = self._resolve_spoken(detected)
        if spoken == "a":
            text_a = text
            text_b = self._translator.translate(text, self._nllb_a, self._nllb_b)
        else:
            text_b = text
            text_a = self._translator.translate(text, self._nllb_b, self._nllb_a)
        if not text_a and not text_b:
            return
        # Apply the configured Chinese script variant to whichever pane is zh.
        text_a = self._zh(text_a, self.lang_a)
        text_b = self._zh(text_b, self.lang_b)
        event = TranslationEvent(
            kind=kind, segment_id=seg_id,
            lang_a=self.lang_a, lang_b=self.lang_b,
            text_a=text_a, text_b=text_b, spoken=spoken,
        )
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.out_queue.put_nowait, event)

    def _emit_error(self, message: str) -> None:
        event = TranslationEvent(
            kind="final", segment_id=new_segment_id(),
            lang_a=self.lang_a, lang_b=self.lang_b,
            text_a=f"[engine error] {message}", text_b="",
        )
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.out_queue.put_nowait, event)

    def _run(self) -> None:
        block = bytearray()        # accumulates raw audio until one 320ms block
        utt = bytearray()          # the current utterance's audio
        seg_id = new_segment_id()
        silence = 0.0
        utt_dur = 0.0              # seconds of audio in `utt`
        since_partial = 0.0        # seconds of audio since the last partial
        speech_seen = False
        detected: str | None = None  # locked once detected for this utterance

        def finalize() -> None:
            nonlocal utt, seg_id, silence, utt_dur, since_partial, speech_seen, detected
            if utt_dur >= _MIN_UTTERANCE:
                text, lang = self._transcribe(bytes(utt), detected)
                if text:
                    self._emit("final", seg_id, text, lang or detected or self.lang_a)
            utt = bytearray()
            seg_id = new_segment_id()
            silence = 0.0
            utt_dur = 0.0
            since_partial = 0.0
            speech_seen = False
            detected = None

        while True:
            item = self._audio_q.get()
            if item is _FLUSH:
                try:
                    if speech_seen:
                        finalize()
                except Exception as exc:  # noqa: BLE001
                    self._emit_error(str(exc))
                return

            block.extend(item)
            try:
                while len(block) >= self._block_bytes:
                    chunk = bytes(block[: self._block_bytes])
                    del block[: self._block_bytes]

                    ints = np.frombuffer(chunk, dtype=np.int16)
                    rms = float(np.sqrt(np.mean(ints.astype(np.float32) ** 2)))
                    is_speech = rms >= _SILENCE_RMS

                    # Drop leading silence so an utterance starts on speech.
                    if not speech_seen and not is_speech:
                        continue
                    speech_seen = True

                    utt.extend(chunk)
                    utt_dur += self._block_dur
                    since_partial += self._block_dur
                    silence = 0.0 if is_speech else silence + self._block_dur

                    if silence >= _SILENCE_HANG or utt_dur >= _MAX_UTTERANCE:
                        finalize()
                    elif since_partial >= _PARTIAL_EVERY and utt_dur >= _MIN_UTTERANCE:
                        since_partial = 0.0
                        # Partials are optional — skip them if the worker has
                        # fallen behind, so it never spirals on a long buffer.
                        if self._audio_q.qsize() <= _MAX_BACKLOG_BLOCKS:
                            text, lang = self._transcribe(bytes(utt), detected)
                            if detected is None and lang:
                                detected = lang
                            if text:
                                self._emit("partial", seg_id, text,
                                           lang or detected or self.lang_a)
            except Exception as exc:  # noqa: BLE001 - surface model failures
                self._emit_error(str(exc))
                return
