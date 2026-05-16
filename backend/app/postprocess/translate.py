"""Translate diarized utterances into both locked languages.

Takes the output of `diarize_and_transcribe` — each utterance carries its text
in a single detected language — and fills in `text_a` / `text_b` for the two
locked languages, applying the Chinese script variant.
"""
from __future__ import annotations

from .. import languages
from ..nllb import NllbTranslator


def translate_utterances(
    utterances: list[dict],
    lang_a: str,
    lang_b: str,
    translate_model: str,
    chinese_variant: str,
) -> list[dict]:
    """Return utterances with `text_a` / `text_b` filled in.

    Blocking / compute-bound — run via asyncio.to_thread.
    """
    translator = NllbTranslator(translate_model)
    translator.build()

    cc = None
    if chinese_variant == "traditional" and "zh" in (lang_a, lang_b):
        import opencc
        cc = opencc.OpenCC("s2twp")

    def zh(text: str, lang: str) -> str:
        if lang == "zh" and cc is not None and text:
            return cc.convert(text)
        return text

    nllb_a = languages.get(lang_a)["nllb"]
    nllb_b = languages.get(lang_b)["nllb"]

    out: list[dict] = []
    for u in utterances:
        # Source language of this utterance; default to lang_a if Whisper
        # detected something outside the locked pair.
        src = u["lang"] if u["lang"] in (lang_a, lang_b) else lang_a
        text = u["text"]
        if src == lang_a:
            text_a = text
            text_b = translator.translate(text, nllb_a, nllb_b)
        else:
            text_b = text
            text_a = translator.translate(text, nllb_b, nllb_a)
        out.append({
            "speaker": u["speaker"],
            "start_ms": u["start_ms"],
            "end_ms": u["end_ms"],
            "text_a": zh(text_a, lang_a),
            "text_b": zh(text_b, lang_b),
        })
    return out
