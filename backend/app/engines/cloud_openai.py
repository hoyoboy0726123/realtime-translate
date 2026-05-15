"""Cloud engine backed by the OpenAI Realtime API (gpt-realtime-translate).

The backend opens a server-to-server WebSocket to OpenAI, forwards the caller's
PCM16 audio, and instructs the model to return every utterance as a JSON object
with the text in both locked languages. Server VAD decides utterance boundaries.
"""
from __future__ import annotations

import asyncio
import base64
import json

import websockets

from .. import languages
from ..config import openai_api_key
from ..detect import guess_spoken
from .base import TranslationEngine, TranslationEvent, new_segment_id

REALTIME_URL = "wss://api.openai.com/v1/realtime?model={model}"


def _instructions(name_a: str, name_b: str) -> str:
    return (
        f"You are a simultaneous interpreter between {name_a} and {name_b}. "
        f"The speaker may talk in either language. For every utterance you hear, "
        f"reply with ONLY a compact JSON object and nothing else, of the form "
        f'{{"a": "<utterance in {name_a}>", "b": "<utterance in {name_b}>"}}. '
        f"One field is the faithful transcription of what was said, the other is "
        f"the translation. Do not add commentary, punctuation outside the JSON, "
        f"or markdown fences."
    )


async def _connect(url: str, headers: dict):
    """Open a websocket, tolerating the websockets v13/v14 header kwarg rename."""
    try:
        return await websockets.connect(url, additional_headers=headers, max_size=None)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers, max_size=None)


def _extract_pair(raw: str) -> tuple[str, str]:
    """Best-effort pull of the a/b strings out of a (possibly partial) JSON blob."""
    try:
        obj = json.loads(raw)
        return str(obj.get("a", "")), str(obj.get("b", ""))
    except json.JSONDecodeError:
        pass

    def grab(key: str) -> str:
        marker = f'"{key}"'
        i = raw.find(marker)
        if i == -1:
            return ""
        i = raw.find(":", i)
        if i == -1:
            return ""
        i = raw.find('"', i)
        if i == -1:
            return ""
        out, j = [], i + 1
        while j < len(raw):
            ch = raw[j]
            if ch == "\\" and j + 1 < len(raw):
                out.append(raw[j + 1])
                j += 2
                continue
            if ch == '"':
                break
            out.append(ch)
            j += 1
        return "".join(out)

    return grab("a"), grab("b")


class CloudEngine(TranslationEngine):
    name = "cloud"

    def __init__(self, lang_a, lang_b, out_queue, settings):
        super().__init__(lang_a, lang_b, out_queue, settings)
        self._ws = None
        self._reader: asyncio.Task | None = None
        self._segments: dict[str, str] = {}        # openai response_id -> our segment_id
        self._buffers: dict[str, str] = {}         # openai response_id -> accumulated text

    async def open(self) -> None:
        key = openai_api_key()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot use the cloud engine.")

        name_a = languages.get(self.lang_a)["openai"]
        name_b = languages.get(self.lang_b)["openai"]
        url = REALTIME_URL.format(model=self.settings.cloud.model)
        headers = {"Authorization": f"Bearer {key}", "OpenAI-Beta": "realtime=v1"}

        self._ws = await _connect(url, headers)
        await self._ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "instructions": _instructions(name_a, name_b),
                "input_audio_format": "pcm16",
                "input_audio_transcription": {"model": "gpt-4o-transcribe"},
                "turn_detection": {
                    "type": "server_vad",
                    "silence_duration_ms": 600,
                    "create_response": True,
                },
            },
        }))
        self._reader = asyncio.create_task(self._read_loop())

    async def send_audio(self, pcm16: bytes) -> None:
        if not self._ws:
            return
        await self._ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm16).decode("ascii"),
        }))

    async def close(self) -> None:
        if self._reader:
            self._reader.cancel()
            try:
                await self._reader
            except asyncio.CancelledError:
                pass
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def _read_loop(self) -> None:
        assert self._ws is not None
        async for raw in self._ws:
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            await self._handle(event)

    async def _handle(self, event: dict) -> None:
        etype = event.get("type", "")

        if etype == "response.text.delta":
            rid = event.get("response_id", "default")
            seg = self._segments.setdefault(rid, new_segment_id())
            buf = self._buffers.get(rid, "") + event.get("delta", "")
            self._buffers[rid] = buf
            text_a, text_b = _extract_pair(buf)
            await self._send(seg, "partial", text_a, text_b)

        elif etype in ("response.text.done", "response.done"):
            rid = event.get("response_id", "default")
            if rid not in self._buffers:
                return
            seg = self._segments.get(rid, new_segment_id())
            text_a, text_b = _extract_pair(self._buffers.pop(rid, ""))
            self._segments.pop(rid, None)
            if text_a or text_b:
                await self._send(seg, "final", text_a, text_b)

        elif etype == "error":
            detail = event.get("error", {}).get("message", "unknown error")
            await self.out_queue.put(TranslationEvent(
                kind="final", segment_id=new_segment_id(),
                lang_a=self.lang_a, lang_b=self.lang_b,
                text_a=f"[engine error] {detail}", text_b="",
            ))

    async def _send(self, seg: str, kind: str, text_a: str, text_b: str) -> None:
        spoken = guess_spoken(text_a or text_b, self.lang_a, self.lang_b)
        await self.emit(TranslationEvent(
            kind=kind, segment_id=seg,
            lang_a=self.lang_a, lang_b=self.lang_b,
            text_a=text_a, text_b=text_b, spoken=spoken,
        ))
