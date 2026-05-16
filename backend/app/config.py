"""Application settings, persisted to a JSON file so the admin page survives restarts."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
SETTINGS_FILE = DATA_DIR / "settings.json"

# Raw session audio is recorded here for post-session analysis.
RECORDINGS_DIR = DATA_DIR / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

# Audio contract between frontend and engines: 16 kHz, mono, signed 16-bit PCM.
SAMPLE_RATE = 16_000


class CloudSettings(BaseModel):
    model: str = Field(default_factory=lambda: os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-translate"))


class LocalSettings(BaseModel):
    """On-device engine: MLX-Whisper (ASR) + NLLB (translation).

    Runs natively on Apple Silicon via Apple's MLX runtime. (SeamlessStreaming
    was dropped — it is numerically broken on the MPS backend and ~26x too slow
    on CPU.)
    """

    # MLX-Whisper model for LIVE subtitles — must keep up with real time, so
    # the speed-optimised turbo variant is used.
    whisper_model: str = "mlx-community/whisper-large-v3-turbo"
    # MLX-Whisper model for post-session ANALYSIS — runs offline, so the
    # slower full large-v3 is used for higher transcription accuracy.
    analyze_whisper_model: str = "mlx-community/whisper-large-v3-mlx"
    # NLLB translation model (HuggingFace transformers).
    translate_model: str = "facebook/nllb-200-distilled-600M"
    # Local LLM (mlx-lm) for post-session meeting summaries.
    # Qwen2.5 is the newest Qwen that works here — Qwen3/3.5 tokenizers need a
    # newer transformers/tokenizers than NLLB (pinned to 4.44.x) allows.
    summary_model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"


class Settings(BaseModel):
    engine: str = "mock"            # "cloud" | "local" | "mock"
    lang_a: str = "zh"              # shown in the TOP pane
    lang_b: str = "en"              # shown in the BOTTOM pane
    # Chinese script for any Chinese pane: "traditional" or "simplified".
    # Engines emit Simplified; "traditional" post-converts via OpenCC.
    chinese_variant: str = "traditional"
    cloud: CloudSettings = Field(default_factory=CloudSettings)
    local: LocalSettings = Field(default_factory=LocalSettings)


_lock = threading.Lock()


def load_settings() -> Settings:
    with _lock:
        if SETTINGS_FILE.exists():
            try:
                return Settings.model_validate_json(SETTINGS_FILE.read_text("utf-8"))
            except Exception:
                pass
        return Settings()


def save_settings(settings: Settings) -> Settings:
    with _lock:
        SETTINGS_FILE.write_text(settings.model_dump_json(indent=2), "utf-8")
    return settings


def cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    return [o.strip() for o in raw.split(",") if o.strip()]


def openai_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY")
