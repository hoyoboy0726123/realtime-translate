"""Meeting summary from a diarized transcript, via a local LLM.

Short transcripts are summarised in one LLM pass. Long ones (more than
`_SINGLE_PASS_CHAR_LIMIT` characters) are summarised map-reduce style: the
transcript is split into chunks, each chunk is summarised, then the chunk
summaries are combined into the final meeting record — so a recording of any
length is covered without overflowing the model's context window.

Uses the cross-platform LLM backend (mlx-lm on Apple Silicon, llama-cpp-python
elsewhere). Blocking / compute-bound — run via asyncio.to_thread. The first
call downloads the model.
"""
from __future__ import annotations

from ..backends import llm

# Transcripts (in characters of the fed language column) at or below this go
# through a single LLM pass; longer ones are chunked. ~6000 chars is roughly
# 25–30 minutes of speech — see the project notes.
_SINGLE_PASS_CHAR_LIMIT = 6000
# Map step: target characters per chunk, and the most chunks we will make
# (bigger chunks beyond that, to keep the reduce step's input bounded).
_CHUNK_CHAR_TARGET = 3500
_MAX_CHUNKS = 8

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

# Map step — summarise one chunk of a long transcript.
_MAP_PROMPT = """以下是一場會議逐字稿的其中一段（第 {i} 段，共 {n} 段），每行開頭可能標註說話者。請用**繁體中文**摘錄這一段的內容，保留：
- 這一段討論的主要事項與脈絡
- 關鍵的原話：用「」標出說話者的原始發言並標明是誰說的
- 這一段出現的任何結論、決定或待辦事項

只摘錄這一段實際出現的內容，不要杜撰，也不要加入其他段落的資訊。

逐字稿（第 {i}/{n} 段）：
{chunk}
"""

# Reduce step — combine the chunk summaries into the final meeting record.
_REDUCE_SIMPLE = """以下是一場會議的逐字稿，已依時間順序分段摘錄。請整合所有分段內容，用**繁體中文**整理成一份簡潔的會議記錄，包含以下段落：

## 會議摘要
（三到五句話概述整場會議）

## 重點討論
（條列主要討論的議題與內容）

## 決議事項
（條列達成的結論或決定；若無則寫「無」）

## 待辦事項
（條列後續行動，盡量標註負責的說話者；若無則寫「無」）

分段摘錄：
{sections}
"""

_REDUCE_DETAILED = """以下是一場會議的逐字稿，已依時間順序分段摘錄。請整合所有分段內容，用**繁體中文**整理成一份**詳盡**的會議記錄。

重要原則：
- 整合全部分段，依內容豐富程度充分展開，不要過度精簡。
- 保留各分段中以「」標示的關鍵原話與說話者。
- 忠實反映實際討論，不要杜撰。

請包含以下段落：

## 會議摘要
（完整說明會議的背景、進行過程與整體結論，至少一段，必要時分多段）

## 重點討論
（逐項條列主要議題。每一項都要：1. 詳細說明討論內容、脈絡與不同說話者的觀點；2. 引用關鍵原話，用「」原文複述並標明說話者。）

## 決議事項
（條列達成的結論或決定，並說明其理由；若無則寫「無」）

## 待辦事項
（條列後續行動，標註負責的說話者與相關細節；若無則寫「無」）

分段摘錄：
{sections}
"""


def _transcript_lines(diarized: list[dict], text_field: str) -> list[str]:
    return [
        f"{d['speaker']}: {d[text_field]}" if d.get("speaker") else d[text_field]
        for d in diarized if d.get(text_field)
    ]


def _chunk_lines(lines: list[str], limit: int) -> list[list[str]]:
    """Group consecutive lines into chunks of roughly `limit` characters."""
    chunks: list[list[str]] = []
    buf: list[str] = []
    size = 0
    for ln in lines:
        if buf and size + len(ln) > limit:
            chunks.append(buf)
            buf, size = [], 0
        buf.append(ln)
        size += len(ln) + 1
    if buf:
        chunks.append(buf)
    return chunks


def _summarize_single(lines: list[str], model_name: str, detailed: bool) -> str:
    prompt = _PROMPT_DETAILED if detailed else _PROMPT_SIMPLE
    messages = [{"role": "user", "content": prompt.format(transcript="\n".join(lines))}]
    return llm.chat(messages, model_name, max_tokens=4000 if detailed else 2000)


def _summarize_chunked(lines: list[str], model_name: str, detailed: bool) -> str:
    total = sum(len(ln) for ln in lines)
    # Keep the chunk count bounded so the reduce step's input stays in context.
    limit = max(_CHUNK_CHAR_TARGET, -(-total // _MAX_CHUNKS))
    chunks = _chunk_lines(lines, limit)
    n = len(chunks)

    sections: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        messages = [{"role": "user", "content": _MAP_PROMPT.format(
            i=i, n=n, chunk="\n".join(chunk))}]
        section = llm.chat(messages, model_name, max_tokens=1200).strip()
        if section:
            sections.append(f"【第 {i}/{n} 段】\n{section}")
    if not sections:
        return ""

    prompt = _REDUCE_DETAILED if detailed else _REDUCE_SIMPLE
    messages = [{"role": "user", "content": prompt.format(
        sections="\n\n".join(sections))}]
    return llm.chat(messages, model_name, max_tokens=4000 if detailed else 2000)


def summarize(
    diarized: list[dict],
    text_field: str,
    model_name: str,
    detail: str = "detailed",
) -> str:
    """Summarise a diarized transcript.

    `text_field` is "text_a"/"text_b" — which language column to feed the model.
    `detail` is "simple" (concise) or "detailed" (rich, with verbatim quotes).
    Long transcripts are summarised chunk by chunk and then combined.
    """
    lines = _transcript_lines(diarized, text_field)
    if not lines:
        return ""
    detailed = detail == "detailed"
    total = sum(len(ln) for ln in lines)
    if total <= _SINGLE_PASS_CHAR_LIMIT:
        return _summarize_single(lines, model_name, detailed)
    return _summarize_chunked(lines, model_name, detailed)
