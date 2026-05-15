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

# Audio contract between frontend and engines: 16 kHz, mono, signed 16-bit PCM.
SAMPLE_RATE = 16_000


class CloudSettings(BaseModel):
    model: str = Field(default_factory=lambda: os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-translate"))


class LocalSettings(BaseModel):
    # transformers id for Seamless; the SeamlessStreaming agent uses its own checkpoint name.
    model: str = "facebook/seamless-m4t-v2-large"
    device: str = "mps"  # mps on Apple Silicon, else cpu / cuda


class Settings(BaseModel):
    engine: str = "mock"            # "cloud" | "local" | "mock"
    lang_a: str = "zh"              # shown in the TOP pane
    lang_b: str = "en"              # shown in the BOTTOM pane
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
    raw = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    return [o.strip() for o in raw.split(",") if o.strip()]


def openai_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY")
