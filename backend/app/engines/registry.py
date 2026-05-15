"""Engine factory: maps the configured engine name to an implementation."""
from __future__ import annotations

from asyncio import Queue

from .base import TranslationEngine, TranslationEvent
from .cloud_openai import CloudEngine
from .local_seamless import LocalEngine
from .mock import MockEngine

ENGINES: dict[str, type[TranslationEngine]] = {
    "cloud": CloudEngine,
    "local": LocalEngine,
    "mock": MockEngine,
}


def create_engine(settings, out_queue: "Queue[TranslationEvent]") -> TranslationEngine:
    engine_cls = ENGINES.get(settings.engine)
    if engine_cls is None:
        raise ValueError(f"Unknown engine: {settings.engine!r}")
    return engine_cls(settings.lang_a, settings.lang_b, out_queue, settings)
