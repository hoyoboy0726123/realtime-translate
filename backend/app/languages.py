"""Supported languages and per-engine code mappings.

`code`     - canonical ISO-639-1 code used across the app / frontend.
`seamless` - 3-letter code expected by Meta Seamless models.
`openai`   - human-readable name used in OpenAI Realtime instructions.
"""

LANGUAGES = [
    {"code": "zh", "name": "Chinese (Mandarin)", "native": "中文", "seamless": "cmn", "openai": "Chinese"},
    {"code": "en", "name": "English", "native": "English", "seamless": "eng", "openai": "English"},
    {"code": "ja", "name": "Japanese", "native": "日本語", "seamless": "jpn", "openai": "Japanese"},
    {"code": "ko", "name": "Korean", "native": "한국어", "seamless": "kor", "openai": "Korean"},
    {"code": "es", "name": "Spanish", "native": "Español", "seamless": "spa", "openai": "Spanish"},
    {"code": "fr", "name": "French", "native": "Français", "seamless": "fra", "openai": "French"},
    {"code": "de", "name": "German", "native": "Deutsch", "seamless": "deu", "openai": "German"},
    {"code": "it", "name": "Italian", "native": "Italiano", "seamless": "ita", "openai": "Italian"},
    {"code": "pt", "name": "Portuguese", "native": "Português", "seamless": "por", "openai": "Portuguese"},
    {"code": "ru", "name": "Russian", "native": "Русский", "seamless": "rus", "openai": "Russian"},
    {"code": "vi", "name": "Vietnamese", "native": "Tiếng Việt", "seamless": "vie", "openai": "Vietnamese"},
    {"code": "th", "name": "Thai", "native": "ไทย", "seamless": "tha", "openai": "Thai"},
]

_BY_CODE = {lang["code"]: lang for lang in LANGUAGES}


def get(code: str) -> dict:
    """Return the language record for a canonical code, or raise KeyError."""
    return _BY_CODE[code]


def is_supported(code: str) -> bool:
    return code in _BY_CODE
