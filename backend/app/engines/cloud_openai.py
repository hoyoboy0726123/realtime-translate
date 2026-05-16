"""Cloud engine — OpenAI Realtime Translation API (into lang_a) + local NLLB.

A fixed, deterministic pipeline:

    speech (any language)  ->  LEFT pane  = lang_a text
    LEFT pane              ->  RIGHT pane = lang_b text (translated by NLLB)

The OpenAI session's output language is fixed to lang_a, so the **left pane is
always lang_a** — whatever is spoken it can never show the wrong language. The
right pane is the left pane translated by NLLB.

One OpenAI session gives two reliable transcripts:
  * `input_transcript`  — what was actually spoken (any language);
  * `output_transcript` — the lang_a translation (reliable when the spoken
    language differs from lang_a; a same-language passthrough is unreliable, so
    we never depend on it).

So the lang_a left pane is taken from `input_transcript` when lang_a is spoken,
and from `output_transcript` when lang_b is spoken — both reliable paths.

The translation API emits no turn events, so an utterance is finalised after a
short delta-silence, which also bounds the rolling buffer. If NLLB is not
installed the right pane is left blank (graceful).

Audio must be 24 kHz PCM16 mono.  The frontend sends 16 kHz, so we resample.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time

import numpy as np
import websockets

from .. import languages
from ..config import SAMPLE_RATE, openai_api_key
from ..detect import guess_spoken
from .base import TranslationEngine, TranslationEvent, new_segment_id

TRANSLATION_URL = "wss://api.openai.com/v1/realtime/translations?model={model}"
OPENAI_SAMPLE_RATE = 24_000
TRANSCRIBE_MODEL = "gpt-realtime-whisper"

_NLLB_TICK = 1.2        # how often the right pane (NLLB) refreshes, seconds
# Delta-silence before the current line is finalised. The translation API
# streams in bursts, so this is generous — it should outlast a within-sentence
# gap and only fire at a real pause / speaker change.
_IDLE_FINALIZE = 3.0
_FINALIZER_TICK = 0.3


def _resample_16k_to_24k(pcm16_bytes: bytes) -> bytes:
    """Resample 16 kHz PCM16 mono to 24 kHz using linear interpolation."""
    samples_16 = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32)
    n_out = int(len(samples_16) * OPENAI_SAMPLE_RATE / SAMPLE_RATE)
    if n_out == 0:
        return b""
    indices = np.linspace(0, len(samples_16) - 1, n_out)
    idx_floor = np.floor(indices).astype(int)
    idx_ceil = np.minimum(idx_floor + 1, len(samples_16) - 1)
    frac = indices - idx_floor
    samples_24 = samples_16[idx_floor] * (1 - frac) + samples_16[idx_ceil] * frac
    return samples_24.astype(np.int16).tobytes()


async def _connect(url: str, headers: dict):
    """Open a websocket, tolerating the websockets v13/v14 header kwarg rename."""
    try:
        return await websockets.connect(url, additional_headers=headers, max_size=None)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers, max_size=None)


class CloudEngine(TranslationEngine):
    name = "cloud"

    def __init__(self, lang_a, lang_b, out_queue, settings):
        super().__init__(lang_a, lang_b, out_queue, settings)
        # OpenAI translates *into* lang_a, so its output is always lang_a.
        self._target = languages.get(lang_a)["code"]
        self._nllb_a = languages.get(lang_a)["nllb"]
        self._nllb_b = languages.get(lang_b)["nllb"]

        self._ws = None
        self._reader: asyncio.Task | None = None
        self._nllb_task: asyncio.Task | None = None
        self._finalizer: asyncio.Task | None = None
        self._build_task: asyncio.Task | None = None

        self._seg_id = new_segment_id()
        self._input_buf = ""    # input_transcript — the spoken language
        self._output_buf = ""   # output_transcript — the lang_a translation
        self._rev_src = ""      # left-pane snapshot last handed to NLLB
        self._rev_out = ""      # NLLB result — the lang_b text
        self._last_delta: float | None = None

        self._translator = None     # NllbTranslator once loaded, else None
        self._zh_variant = settings.chinese_variant
        self._cc = None

    # ---- lifecycle ----------------------------------------------------------

    async def open(self) -> None:
        key = openai_api_key()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot use the cloud engine.")

        if self._zh_variant == "traditional" and "zh" in (self.lang_a, self.lang_b):
            import opencc
            self._cc = opencc.OpenCC("s2twp")

        url = TRANSLATION_URL.format(model=self.settings.cloud.model)
        headers = {"Authorization": f"Bearer {key}"}
        self._ws = await _connect(url, headers)
        await self._ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "audio": {
                    "input": {"transcription": {"model": TRANSCRIBE_MODEL}},
                    "output": {"language": self._target},
                },
            },
        }))
        self._reader = asyncio.create_task(self._read_loop())
        self._nllb_task = asyncio.create_task(self._nllb_loop())
        self._finalizer = asyncio.create_task(self._finalizer_loop())
        # Load NLLB in the background so the session starts without waiting.
        self._build_task = asyncio.create_task(self._build_nllb())

    async def _build_nllb(self) -> None:
        """Load the NLLB translator for the right pane (best-effort)."""
        try:
            from ..nllb import NllbTranslator
            translator = NllbTranslator(self.settings.local.translate_model)
            await asyncio.to_thread(translator.build)
            self._translator = translator
            logging.info("[cloud] NLLB ready — right pane enabled")
        except Exception as exc:  # noqa: BLE001
            logging.warning(f"[cloud] NLLB unavailable — right pane disabled: {exc}")

    async def send_audio(self, pcm16: bytes) -> None:
        if not self._ws:
            return
        pcm24 = _resample_16k_to_24k(pcm16)
        if not pcm24:
            return
        try:
            await self._ws.send(json.dumps({
                "type": "session.input_audio_buffer.append",
                "audio": base64.b64encode(pcm24).decode("ascii"),
            }))
        except Exception:  # noqa: BLE001
            pass

    async def close(self) -> None:
        for task in (self._build_task, self._finalizer, self._nllb_task, self._reader):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self._ws:
            await self._ws.close()
            self._ws = None

    # ---- OpenAI session reader ---------------------------------------------

    async def _read_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await self._handle(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logging.error(f"[cloud] read loop error: {exc}")

    async def _handle(self, event: dict) -> None:
        etype = event.get("type", "")

        if etype == "session.input_transcript.delta":
            self._input_buf += event.get("delta", "")
            self._last_delta = time.monotonic()
            await self._emit_partial()

        elif etype == "session.output_transcript.delta":
            self._output_buf += event.get("delta", "")
            self._last_delta = time.monotonic()
            await self._emit_partial()

        elif etype == "error":
            detail = event.get("error", {}).get("message", "unknown error")
            logging.error(f"[cloud] error: {detail}")
            await self.out_queue.put(TranslationEvent(
                kind="final", segment_id=new_segment_id(),
                lang_a=self.lang_a, lang_b=self.lang_b,
                text_a=f"[engine error] {detail}", text_b="",
            ))

    # ---- left pane (always lang_a) -----------------------------------------

    def _left(self) -> str:
        """The lang_a (left-pane) text.

        lang_a spoken -> the input transcript is already lang_a.
        lang_b spoken -> the output transcript is the lang_a translation.
        (We never use the same-language passthrough, which is unreliable.)
        """
        if guess_spoken(self._input_buf, self.lang_a, self.lang_b) == "a":
            return self._input_buf.strip()
        return self._output_buf.strip()

    # ---- right pane (NLLB) + segmentation ----------------------------------

    async def _nllb_loop(self) -> None:
        """Keep the right pane in sync — translate the left (lang_a) to lang_b."""
        try:
            while True:
                await asyncio.sleep(_NLLB_TICK)
                if self._translator is None:
                    continue
                src = self._left()
                if not src or src == self._rev_src:
                    continue
                self._rev_src = src
                self._rev_out = await asyncio.to_thread(
                    self._translator.translate, src, self._nllb_a, self._nllb_b,
                )
                await self._emit_partial()
        except asyncio.CancelledError:
            pass

    async def _finalizer_loop(self) -> None:
        """Finalise the current line once OpenAI's deltas have gone quiet."""
        try:
            while True:
                await asyncio.sleep(_FINALIZER_TICK)
                if self._last_delta is None:
                    continue
                if time.monotonic() - self._last_delta >= _IDLE_FINALIZE:
                    self._last_delta = None
                    await self._emit_final()
        except asyncio.CancelledError:
            pass

    # ---- emit ---------------------------------------------------------------

    def _zh(self, text: str, lang: str) -> str:
        """Convert a Chinese pane's text to the configured script variant."""
        if lang == "zh" and self._cc is not None and text:
            return self._cc.convert(text)
        return text

    async def _emit_partial(self) -> None:
        text_a = self._left()
        text_b = self._rev_out.strip()
        if not text_a:  # the left (lang_a) pane drives the line
            return
        await self.emit(TranslationEvent(
            kind="partial", segment_id=self._seg_id,
            lang_a=self.lang_a, lang_b=self.lang_b,
            text_a=self._zh(text_a, self.lang_a), text_b=text_b,
        ))

    async def _emit_final(self) -> None:
        text_a = self._left()
        if not text_a:
            return
        seg_id = self._seg_id
        # Reset before the (awaited) translation so trailing deltas start a
        # fresh line instead of leaking into the one being finalised.
        self._seg_id = new_segment_id()
        self._input_buf = ""
        self._output_buf = ""
        self._rev_src = ""
        self._rev_out = ""

        text_b = ""
        if self._translator is not None:
            text_b = await asyncio.to_thread(
                self._translator.translate, text_a, self._nllb_a, self._nllb_b,
            )
        await self.emit(TranslationEvent(
            kind="final", segment_id=seg_id,
            lang_a=self.lang_a, lang_b=self.lang_b,
            text_a=self._zh(text_a, self.lang_a), text_b=text_b,
        ))
