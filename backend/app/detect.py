"""Lightweight script-based language guessing.

Used only to decide which locked language was *spoken* (for highlighting).
The actual translation always comes from the engine.
"""
import re

_CJK = re.compile(r"[一-鿿]")
_KANA = re.compile(r"[぀-ヿ]")
_HANGUL = re.compile(r"[가-힯]")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_THAI = re.compile(r"[฀-๿]")


def script_lang(text: str) -> str:
    """Return a coarse language code based on the dominant script."""
    if _HANGUL.search(text):
        return "ko"
    if _KANA.search(text):
        return "ja"
    if _CJK.search(text):
        return "zh"
    if _CYRILLIC.search(text):
        return "ru"
    if _THAI.search(text):
        return "th"
    return "latin"


# Languages that `script_lang` can identify by their own distinctive script.
# Anything else ("latin": en, es, fr, de, it, pt, vi) is reported as "latin".
_SPECIAL = {"ko", "ja", "zh", "ru", "th"}


def guess_spoken(text: str, lang_a: str, lang_b: str) -> str | None:
    """Return 'a' or 'b' for which locked language the text appears to be in."""
    detected = script_lang(text)
    if detected == lang_a:
        return "a"
    if detected == lang_b:
        return "b"
    # `detected` is "latin" — script_lang can't tell latin languages apart.
    # But the pair is locked: if exactly one side uses a distinctive non-latin
    # script, latin text must be the *other* (latin) side.
    a_special = lang_a in _SPECIAL
    b_special = lang_b in _SPECIAL
    if a_special != b_special:
        return "b" if a_special else "a"
    return None
