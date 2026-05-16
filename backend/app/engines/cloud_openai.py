"""Cloud engine backed by the OpenAI Realtime Translation API (gpt-realtime-translate).

The translation API is one-directional per session: each session has a single
target output language.  To always show *both* languages regardless of which one
is spoken, we open **two parallel sessions**:

  Session A  →  target = lang_a  (e.g. Chinese)
  Session B  →  target = lang_b  (e.g. English)

The same microphone audio is sent to both.  When the speaker uses Chinese,
Session B produces the English translation while Session A passes through the
Chinese transcript.  When the speaker switches to English, Session A produces
the Chinese translation while Session B passes through.

Session A's output goes to the top pane (lang_a), Session B's to the bottom
pane (lang_b).  This works well as a Chinese-primary setup: speaking Chinese
gives live bilingual subtitles; speaking English shows the Chinese translation
up top while the English pane may pause.  (The translation API emits no turn
boundary events and offers no clean way to segment a single mic into both
directions — for fully symmetric bilingual subtitles use the on-device engine.)

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


class _Session:
    """One WebSocket session targeting a single output language."""

    def __init__(self, label: str):
        self.label = label          # "a" or "b", for logging
        self.ws = None
        self.reader: asyncio.Task | None = None
        self.input_buf: str = ""    # input_transcript (source language)
        self.output_buf: str = ""   # output_transcript (target language)
        self.turn_done: bool = False

    @property
    def text(self) -> str:
        """Best available text: translation if present, else passthrough transcript."""
        return (self.output_buf or self.input_buf).strip()

    async def open(self, url: str, headers: dict, tgt_code: str) -> None:
        self.ws = await _connect(url, headers)
        await self.ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "audio": {
                    "input": {
                        # gpt-realtime-whisper is the transcription model the
                        # translation endpoint expects; without a valid one the
                        # source-language `input_transcript` is never emitted.
                        "transcription": {"model": "gpt-realtime-whisper"},
                    },
                    "output": {
                        "language": tgt_code,
                    },
                },
            },
        }))

    async def send_audio(self, pcm24_b64: str) -> None:
        if not self.ws:
            return
        await self.ws.send(json.dumps({
            "type": "session.input_audio_buffer.append",
            "audio": pcm24_b64,
        }))

    async def close(self) -> None:
        if self.reader:
            self.reader.cancel()
            try:
                await self.reader
            except asyncio.CancelledError:
                pass
        if self.ws:
            await self.ws.close()
            self.ws = None


class CloudEngine(TranslationEngine):
    name = "cloud"

    def __init__(self, lang_a, lang_b, out_queue, settings):
        super().__init__(lang_a, lang_b, out_queue, settings)
        self._sess_a = _Session("a")  # target = lang_a
        self._sess_b = _Session("b")  # target = lang_b
        self._seg_id: str = new_segment_id()
        # OpenCC converter for Simplified -> Traditional, built in open().
        self._zh_variant = settings.chinese_variant
        self._cc = None

    def _zh(self, text: str, lang: str) -> str:
        """Convert a Chinese pane's text to the configured script variant."""
        if lang == "zh" and self._cc is not None and text:
            return self._cc.convert(text)
        return text

    async def open(self) -> None:
        key = openai_api_key()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot use the cloud engine.")

        # The translation API emits Simplified Chinese; convert to Traditional
        # (Taiwan, with phrase conversion) when requested.
        if self._zh_variant == "traditional" and "zh" in (self.lang_a, self.lang_b):
            import opencc
            self._cc = opencc.OpenCC("s2twp")

        model = self.settings.cloud.model
        url = TRANSLATION_URL.format(model=model)
        headers = {"Authorization": f"Bearer {key}"}

        code_a = languages.get(self.lang_a)["code"]
        code_b = languages.get(self.lang_b)["code"]

        # Open both sessions in parallel.
        await asyncio.gather(
            self._sess_a.open(url, headers, code_a),
            self._sess_b.open(url, headers, code_b),
        )

        self._sess_a.reader = asyncio.create_task(self._read_loop(self._sess_a))
        self._sess_b.reader = asyncio.create_task(self._read_loop(self._sess_b))

    async def send_audio(self, pcm16: bytes) -> None:
        pcm24 = _resample_16k_to_24k(pcm16)
        if not pcm24:
            return
        b64 = base64.b64encode(pcm24).decode("ascii")
        # Feed the same audio to both sessions.
        await asyncio.gather(
            self._sess_a.send_audio(b64),
            self._sess_b.send_audio(b64),
        )

    async def close(self) -> None:
        await asyncio.gather(
            self._sess_a.close(),
            self._sess_b.close(),
        )

    # ---- reader loops -------------------------------------------------------

    async def _read_loop(self, sess: _Session) -> None:
        assert sess.ws is not None
        try:
            async for raw in sess.ws:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await self._handle(sess, event)
        except Exception as exc:
            logging.error(f"[cloud-{sess.label}] _read_loop error: {exc}")

    async def _handle(self, sess: _Session, event: dict) -> None:
        etype = event.get("type", "")

        if etype == "session.input_transcript.delta":
            sess.input_buf += event.get("delta", "")
            await self._emit_partial()

        elif etype == "session.output_transcript.delta":
            sess.output_buf += event.get("delta", "")
            await self._emit_partial()

        elif etype == "session.turn.done":
            sess.turn_done = True
            # Only finalize when both sessions have finished the turn.
            if self._sess_a.turn_done and self._sess_b.turn_done:
                await self._emit_final()

        elif etype == "error":
            detail = event.get("error", {}).get("message", "unknown error")
            logging.error(f"[cloud-{sess.label}] error: {detail}")
            await self.out_queue.put(TranslationEvent(
                kind="final", segment_id=new_segment_id(),
                lang_a=self.lang_a, lang_b=self.lang_b,
                text_a=f"[engine error] {detail}", text_b="",
            ))

    def _panes(self) -> tuple[str, str]:
        """Return (text_a, text_b).

        Only the session that is actually *translating* (its target language
        differs from what was spoken) reliably emits transcripts — it carries
        both the source `input_transcript` and the translated `output_transcript`.
        The other session is doing a passthrough (target == spoken language) and
        the API emits it unreliably, so we never depend on it: both panes are
        taken from the one translating session.
        """
        a, b = self._sess_a, self._sess_b
        if b.output_buf.strip():
            # Session B translated lang_a -> lang_b (speaker used lang_a).
            text_a, text_b = b.input_buf, b.output_buf
        elif a.output_buf.strip():
            # Session A translated lang_b -> lang_a (speaker used lang_b).
            text_a, text_b = a.output_buf, a.input_buf
        else:
            # No translation output yet — show whatever source transcript exists.
            text_a = a.input_buf or b.input_buf
            text_b = b.input_buf or a.input_buf
        return text_a.strip(), text_b.strip()

    async def _emit_partial(self) -> None:
        text_a, text_b = self._panes()
        if not text_a and not text_b:
            return
        spoken = guess_spoken(text_a or text_b, self.lang_a, self.lang_b)
        await self.emit(TranslationEvent(
            kind="partial", segment_id=self._seg_id,
            lang_a=self.lang_a, lang_b=self.lang_b,
            text_a=self._zh(text_a, self.lang_a),
            text_b=self._zh(text_b, self.lang_b), spoken=spoken,
        ))

    async def _emit_final(self) -> None:
        text_a, text_b = self._panes()
        spoken = guess_spoken(text_a or text_b, self.lang_a, self.lang_b)
        await self.emit(TranslationEvent(
            kind="final", segment_id=self._seg_id,
            lang_a=self.lang_a, lang_b=self.lang_b,
            text_a=self._zh(text_a, self.lang_a),
            text_b=self._zh(text_b, self.lang_b), spoken=spoken,
        ))
        # Reset for next utterance.
        self._seg_id = new_segment_id()
        self._sess_a.input_buf = ""
        self._sess_a.output_buf = ""
        self._sess_b.input_buf = ""
        self._sess_b.output_buf = ""
        self._sess_a.turn_done = False
        self._sess_b.turn_done = False
