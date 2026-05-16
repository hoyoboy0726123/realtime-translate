"""Cloud engine — OpenAI Realtime Translation API + local NLLB for the reverse
direction.

The translation API is one-directional per session, and running two sessions
desyncs badly. So instead we run **one** OpenAI session (target = lang_b) and
fill the other direction with the local NLLB translator:

  * spoken == lang_a  -> OpenAI returns the lang_a transcript *and* the lang_b
    translation; both panes come from the one session, perfectly in sync.
  * spoken == lang_b  -> OpenAI's output is a passthrough; the lang_a pane is
    produced by translating the lang_b transcript with NLLB.

This is a Chinese-primary setup: the main direction (lang_a -> lang_b) uses
OpenAI; the occasional reverse direction uses NLLB. If NLLB is not installed
the reverse direction is just left blank (graceful degradation).

Audio must be 24 kHz PCM16 mono.  The frontend sends 16 kHz, so we resample.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging

import numpy as np
import websockets

from .. import languages
from ..config import SAMPLE_RATE, openai_api_key
from ..detect import guess_spoken
from .base import TranslationEngine, TranslationEvent, new_segment_id

TRANSLATION_URL = "wss://api.openai.com/v1/realtime/translations?model={model}"
OPENAI_SAMPLE_RATE = 24_000
TRANSCRIBE_MODEL = "gpt-realtime-whisper"

# How often the reverse-direction (lang_b -> lang_a) NLLB translation refreshes.
_NLLB_TICK = 1.2


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
        self._code_a = languages.get(lang_a)["code"]
        self._code_b = languages.get(lang_b)["code"]
        self._target = self._code_b          # OpenAI translates into lang_b
        self._nllb_a = languages.get(lang_a)["nllb"]
        self._nllb_b = languages.get(lang_b)["nllb"]

        self._ws = None
        self._reader: asyncio.Task | None = None
        self._nllb_task: asyncio.Task | None = None
        self._build_task: asyncio.Task | None = None

        self._seg_id = new_segment_id()
        self._input_buf = ""    # transcript of whatever language was spoken
        self._output_buf = ""   # OpenAI translation, in lang_b

        # Reverse direction (lang_b spoken -> lang_a pane), via local NLLB.
        self._translator = None     # NllbTranslator once loaded, else None
        self._rev_src = ""          # input_buf snapshot last sent to NLLB
        self._rev_out = ""          # NLLB result, in lang_a

        # OpenCC converter for Simplified -> Traditional Chinese.
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
        # Load NLLB in the background so the session starts without waiting on it.
        self._build_task = asyncio.create_task(self._build_nllb())

    async def _build_nllb(self) -> None:
        """Load the NLLB translator for the reverse direction (best-effort)."""
        try:
            from ..nllb import NllbTranslator
            translator = NllbTranslator(self.settings.local.translate_model)
            await asyncio.to_thread(translator.build)
            self._translator = translator
            logging.info("[cloud] NLLB ready — reverse direction enabled")
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                f"[cloud] NLLB unavailable — reverse direction disabled: {exc}"
            )

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
        for task in (self._build_task, self._nllb_task, self._reader):
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
            await self._emit_partial()

        elif etype == "session.output_transcript.delta":
            self._output_buf += event.get("delta", "")
            await self._emit_partial()

        elif etype == "error":
            detail = event.get("error", {}).get("message", "unknown error")
            logging.error(f"[cloud] error: {detail}")
            await self.out_queue.put(TranslationEvent(
                kind="final", segment_id=new_segment_id(),
                lang_a=self.lang_a, lang_b=self.lang_b,
                text_a=f"[engine error] {detail}", text_b="",
            ))

    # ---- reverse-direction NLLB translation --------------------------------

    async def _nllb_loop(self) -> None:
        """Keep the lang_a pane updated (via NLLB) while lang_b is being spoken."""
        try:
            while True:
                await asyncio.sleep(_NLLB_TICK)
                if self._translator is None:
                    continue
                if guess_spoken(self._input_buf, self.lang_a, self.lang_b) != "b":
                    continue
                src = self._input_buf.strip()
                if not src or src == self._rev_src:
                    continue
                self._rev_src = src
                self._rev_out = await asyncio.to_thread(
                    self._translator.translate, src, self._nllb_b, self._nllb_a,
                )
                await self._emit_partial()
        except asyncio.CancelledError:
            pass

    # ---- emit ---------------------------------------------------------------

    def _zh(self, text: str, lang: str) -> str:
        """Convert a Chinese pane's text to the configured script variant."""
        if lang == "zh" and self._cc is not None and text:
            return self._cc.convert(text)
        return text

    def _panes(self) -> tuple[str, str, str | None]:
        """(text_a, text_b, spoken).

        spoken == lang_a : OpenAI translated lang_a -> lang_b (both panes synced).
        spoken == lang_b : lang_b pane is the transcript; lang_a pane is NLLB.
        """
        spoken = guess_spoken(self._input_buf, self.lang_a, self.lang_b)
        if spoken == "b":
            return self._rev_out.strip(), self._input_buf.strip(), "b"
        # lang_a spoken, or not yet known.
        return self._input_buf.strip(), self._output_buf.strip(), spoken

    async def _emit_partial(self) -> None:
        text_a, text_b, spoken = self._panes()
        if not text_a and not text_b:
            return
        await self.emit(TranslationEvent(
            kind="partial", segment_id=self._seg_id,
            lang_a=self.lang_a, lang_b=self.lang_b,
            text_a=self._zh(text_a, self.lang_a),
            text_b=self._zh(text_b, self.lang_b), spoken=spoken,
        ))
