"""Meeting summary from a diarized transcript, via a local LLM.

Uses the cross-platform LLM backend (mlx-lm on Apple Silicon, llama-cpp-python
elsewhere). Blocking / compute-bound — run via asyncio.to_thread. The first
call downloads the model.
"""
from __future__ import annotations

from ..backends import llm

_PROMPT = """以下是一場會議的逐字稿，每行開頭標註說話者。請仔細閱讀後，用**繁體中文**整理一份會議記錄，包含以下段落：

## 會議摘要
（三到五句話概述整場會議）

## 重點討論
（條列主要討論的議題與內容）

## 決議事項
（條列達成的結論或決定；若無則寫「無」）

## 待辦事項
（條列後續行動，盡量標註負責的說話者；若無則寫「無」）

逐字稿：
{transcript}
"""


def summarize(diarized: list[dict], text_field: str, model_name: str) -> str:
    """Summarise a diarized transcript. `text_field` is "text_a" or "text_b" —
    which language column to feed the model."""
    lines = [f"{d['speaker']}: {d[text_field]}" for d in diarized if d.get(text_field)]
    if not lines:
        return ""
    messages = [{
        "role": "user",
        "content": _PROMPT.format(transcript="\n".join(lines)),
    }]
    return llm.chat(messages, model_name, max_tokens=2000)
