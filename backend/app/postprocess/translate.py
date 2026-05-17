"""Translate diarized utterances into both locked languages.

Takes the output of `diarize_and_transcribe` — each utterance carries its text
in a single detected language — and fills in `text_a` / `text_b` for the two
locked languages, applying the Chinese script variant.
"""
from __future__ import annotations

import re

from .. import languages
from ..nllb import NllbTranslator

# A diarized turn can span a long stretch of speech. NLLB truncates anything
# past ~512 tokens and, on long greedy decodes, tends to fall into a repetition
# loop that the degenerate-text guard then discards — leaving an empty result.
# So translate long utterances chunk by chunk and stitch the pieces back.
_MAX_CHUNK_CHARS = 160
# Sentence-ending punctuation (CJK + Latin) — preferred split points.
_SENTENCE_END = re.compile(r"(?<=[。！？!?；;])")


def _chunk_text(text: str) -> list[str]:
    """Split text into translation-sized chunks, preferring sentence breaks."""
    text = text.strip()
    if len(text) <= _MAX_CHUNK_CHARS:
        return [text] if text else []

    chunks: list[str] = []
    for piece in _SENTENCE_END.split(text):
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) <= _MAX_CHUNK_CHARS:
            chunks.append(piece)
            continue
        # No sentence punctuation (Whisper sometimes emits spaces instead) —
        # fall back to splitting on whitespace, then hard-wrap if still long.
        words, buf = piece.split(), ""
        for w in words or [piece]:
            if buf and len(buf) + 1 + len(w) > _MAX_CHUNK_CHARS:
                chunks.append(buf)
                buf = ""
            buf = f"{buf} {w}".strip() if buf else w
            while len(buf) > _MAX_CHUNK_CHARS:
                chunks.append(buf[:_MAX_CHUNK_CHARS])
                buf = buf[_MAX_CHUNK_CHARS:]
        if buf:
            chunks.append(buf)
    return chunks


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

    def translate_long(text: str, src_nllb: str, tgt_nllb: str, tgt_lang: str) -> str:
        """Translate possibly-long text chunk by chunk and rejoin."""
        sep = "" if tgt_lang == "zh" else " "
        parts = [
            translator.translate(chunk, src_nllb, tgt_nllb)
            for chunk in _chunk_text(text)
        ]
        return sep.join(p for p in parts if p).strip()

    out: list[dict] = []
    for u in utterances:
        # Source language of this utterance; default to lang_a if Whisper
        # detected something outside the locked pair.
        src = u["lang"] if u["lang"] in (lang_a, lang_b) else lang_a
        text = u["text"]
        if src == lang_a:
            text_a = text
            text_b = translate_long(text, nllb_a, nllb_b, lang_b)
        else:
            text_b = text
            text_a = translate_long(text, nllb_b, nllb_a, lang_a)
        out.append({
            "speaker": u["speaker"],
            "start_ms": u["start_ms"],
            "end_ms": u["end_ms"],
            "text_a": zh(text_a, lang_a),
            "text_b": zh(text_b, lang_b),
        })
    return out
