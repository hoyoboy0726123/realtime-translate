"""Meeting summary from a diarized transcript, via a local LLM.

Uses the cross-platform LLM backend (mlx-lm on Apple Silicon, llama-cpp-python
elsewhere). Blocking / compute-bound — run via asyncio.to_thread. The first
call downloads the model.
"""
from __future__ import annotations

from ..backends import llm

_PROMPT_SIMPLE = """以下是一場會議的逐字稿，每行開頭標註說話者。請仔細閱讀後，用**繁體中文**整理一份簡潔的會議記錄，包含以下段落：

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

_PROMPT_DETAILED = """以下是一場會議的逐字稿，每行開頭標註說話者。請仔細閱讀後，用**繁體中文**整理一份**詳盡**的會議記錄。

重要原則：
- 依逐字稿內容的豐富程度充分展開，內容多就寫得詳細，不要過度精簡。
- 忠實反映實際討論，不要杜撰逐字稿中沒有的內容。

請包含以下段落：

## 會議摘要
（完整說明會議的背景、進行過程與整體結論，至少一段，必要時分多段）

## 重點討論
（逐項條列主要議題。每一項都必須做到兩件事：
 1. 詳細說明該議題的討論內容、脈絡，以及不同說話者的觀點；
 2. 引用逐字稿中的關鍵原話——用「」把說話者的原始發言**原文複述**一次，並標明是誰說的。）

## 決議事項
（條列達成的結論或決定，並說明其理由與背景；若無則寫「無」）

## 待辦事項
（條列後續行動，標註負責的說話者與相關細節；若無則寫「無」）

逐字稿：
{transcript}
"""


def summarize(
    diarized: list[dict],
    text_field: str,
    model_name: str,
    detail: str = "detailed",
) -> str:
    """Summarise a diarized transcript.

    `text_field` is "text_a"/"text_b" — which language column to feed the model.
    `detail` is "simple" (concise) or "detailed" (rich, with verbatim quotes).
    """
    lines = [
        f"{d['speaker']}: {d[text_field]}" if d.get("speaker") else d[text_field]
        for d in diarized if d.get(text_field)
    ]
    if not lines:
        return ""
    detailed = detail == "detailed"
    prompt = _PROMPT_DETAILED if detailed else _PROMPT_SIMPLE
    messages = [{
        "role": "user",
        "content": prompt.format(transcript="\n".join(lines)),
    }]
    return llm.chat(messages, model_name, max_tokens=4000 if detailed else 2000)
