"""Supported languages and per-engine code mappings.

`code`     - canonical ISO-639-1 code used across the app / frontend; also the
             code Whisper reports for a detected language.
`seamless` - 3-letter code expected by Meta Seamless models.
`openai`   - human-readable name used in OpenAI Realtime instructions.
`nllb`     - FLORES-200 code expected by Meta's NLLB translation model.
"""

LANGUAGES = [
    {"code": "zh", "name": "Chinese (Mandarin)", "native": "中文", "seamless": "cmn", "openai": "Chinese", "nllb": "zho_Hans"},
    {"code": "en", "name": "English", "native": "English", "seamless": "eng", "openai": "English", "nllb": "eng_Latn"},
    {"code": "ja", "name": "Japanese", "native": "日本語", "seamless": "jpn", "openai": "Japanese", "nllb": "jpn_Jpan"},
    {"code": "ko", "name": "Korean", "native": "한국어", "seamless": "kor", "openai": "Korean", "nllb": "kor_Hang"},
    {"code": "es", "name": "Spanish", "native": "Español", "seamless": "spa", "openai": "Spanish", "nllb": "spa_Latn"},
    {"code": "fr", "name": "French", "native": "Français", "seamless": "fra", "openai": "French", "nllb": "fra_Latn"},
    {"code": "de", "name": "German", "native": "Deutsch", "seamless": "deu", "openai": "German", "nllb": "deu_Latn"},
    {"code": "it", "name": "Italian", "native": "Italiano", "seamless": "ita", "openai": "Italian", "nllb": "ita_Latn"},
    {"code": "pt", "name": "Portuguese", "native": "Português", "seamless": "por", "openai": "Portuguese", "nllb": "por_Latn"},
    {"code": "ru", "name": "Russian", "native": "Русский", "seamless": "rus", "openai": "Russian", "nllb": "rus_Cyrl"},
    {"code": "vi", "name": "Vietnamese", "native": "Tiếng Việt", "seamless": "vie", "openai": "Vietnamese", "nllb": "vie_Latn"},
    {"code": "th", "name": "Thai", "native": "ไทย", "seamless": "tha", "openai": "Thai", "nllb": "tha_Thai"},
]

_BY_CODE = {lang["code"]: lang for lang in LANGUAGES}


def get(code: str) -> dict:
    """Return the language record for a canonical code, or raise KeyError."""
    return _BY_CODE[code]


def is_supported(code: str) -> bool:
    return code in _BY_CODE
