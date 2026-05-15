"""Admin settings API: choose the engine and lock the language pair."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import languages
from ..config import Settings, load_settings, save_settings
from ..engines.registry import ENGINES

router = APIRouter(prefix="/api", tags=["settings"])


@router.get("/languages")
def get_languages() -> dict:
    return {"languages": languages.LANGUAGES}


@router.get("/engines")
def get_engines() -> dict:
    return {"engines": list(ENGINES.keys())}


@router.get("/settings")
def get_settings() -> Settings:
    return load_settings()


@router.put("/settings")
def update_settings(settings: Settings) -> Settings:
    if settings.engine not in ENGINES:
        raise HTTPException(400, f"Unknown engine: {settings.engine}")
    if not languages.is_supported(settings.lang_a) or not languages.is_supported(settings.lang_b):
        raise HTTPException(400, "Unsupported language code")
    if settings.lang_a == settings.lang_b:
        raise HTTPException(400, "The two locked languages must be different")
    return save_settings(settings)
