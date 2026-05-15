"""Keyless demo engine.

Ignores the incoming audio and replays a canned bilingual conversation on a
timer. Lets you verify the rolling subtitles, recording and history pages
before wiring up an API key or downloading the local model.
"""
from __future__ import annotations

import asyncio

from .base import TranslationEngine, TranslationEvent, new_segment_id

# (spoken side, Chinese text, English text)
_SCRIPT = [
    ("a", "大家好，歡迎來到今天的會議。", "Hello everyone, welcome to today's meeting."),
    ("b", "謝謝。我們先從專案進度開始。", "Thanks. Let's start with the project status."),
    ("a", "前端的即時字幕已經可以運作了。", "The real-time subtitles on the frontend are working now."),
    ("b", "太好了。後端的引擎呢？", "Great. How about the backend engines?"),
    ("a", "雲端跟地端兩種模型都可以切換。", "Both the cloud and local models can be switched."),
    ("b", "完美，我們把這段記錄下來做會議記錄。", "Perfect, let's record this for the meeting notes."),
]


def _truncate(text: str, step: int) -> str:
    """Return the first `step` words (space-joined) or characters of `text`."""
    if " " in text:
        return " ".join(text.split(" ")[:step])
    return text[:step]


def _units(text: str) -> int:
    return len(text.split(" ")) if " " in text else len(text)


class MockEngine(TranslationEngine):
    name = "mock"

    def __init__(self, lang_a, lang_b, out_queue, settings):
        super().__init__(lang_a, lang_b, out_queue, settings)
        self._task: asyncio.Task | None = None

    async def open(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def send_audio(self, pcm16: bytes) -> None:
        # Mock engine is timer-driven; audio is intentionally discarded.
        return

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        index = 0
        while True:
            spoken, zh, en = _SCRIPT[index % len(_SCRIPT)]
            index += 1
            seg_id = new_segment_id()
            text_a = zh if self.lang_a == "zh" else en
            text_b = en if self.lang_b == "en" else zh

            # Stream the line in unit-by-unit as partials, then finalise it.
            steps = max(_units(text_a), _units(text_b))
            for step in range(1, steps + 1):
                await asyncio.sleep(0.18)
                await self.emit(TranslationEvent(
                    kind="partial", segment_id=seg_id,
                    lang_a=self.lang_a, lang_b=self.lang_b,
                    text_a=_truncate(text_a, step), text_b=_truncate(text_b, step),
                    spoken=spoken,
                ))
            await self.emit(TranslationEvent(
                kind="final", segment_id=seg_id,
                lang_a=self.lang_a, lang_b=self.lang_b,
                text_a=text_a, text_b=text_b, spoken=spoken,
            ))
            await asyncio.sleep(1.2)
