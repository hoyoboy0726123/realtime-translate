"""Engine interface shared by the cloud, local and mock translation backends.

Every engine consumes a stream of 16 kHz mono PCM16 audio and emits
`TranslationEvent`s onto an asyncio queue. Each event carries the utterance
rendered in BOTH locked languages (`text_a` / `text_b`) so the frontend can keep
the top and bottom panes language-fixed and aligned.
"""
from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from asyncio import Queue
from dataclasses import dataclass, field


@dataclass
class TranslationEvent:
    kind: str                       # "partial" while an utterance is forming, "final" when settled
    segment_id: str
    lang_a: str
    lang_b: str
    text_a: str
    text_b: str
    spoken: str | None = None       # "a" or "b": which language was actually spoken
    ts: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_message(self) -> dict:
        return {
            "type": "segment",
            "kind": self.kind,
            "segment_id": self.segment_id,
            "lang_a": self.lang_a,
            "lang_b": self.lang_b,
            "text_a": self.text_a,
            "text_b": self.text_b,
            "spoken": self.spoken,
            "ts": self.ts,
        }


def new_segment_id() -> str:
    return f"seg-{uuid.uuid4().hex[:12]}"


class TranslationEngine(ABC):
    """Base class. Subclasses push `TranslationEvent`s onto `out_queue`."""

    name = "base"

    def __init__(self, lang_a: str, lang_b: str, out_queue: "Queue[TranslationEvent]", settings):
        self.lang_a = lang_a
        self.lang_b = lang_b
        self.out_queue = out_queue
        self.settings = settings

    @abstractmethod
    async def open(self) -> None:
        """Establish any sessions/models needed before audio arrives."""

    @abstractmethod
    async def send_audio(self, pcm16: bytes) -> None:
        """Feed a chunk of 16 kHz mono PCM16 audio."""

    @abstractmethod
    async def close(self) -> None:
        """Flush and release resources."""

    async def emit(self, event: TranslationEvent) -> None:
        await self.out_queue.put(event)
