"""Cross-platform local LLM chat, used for post-session meeting summaries.

Apple Silicon  -> mlx-lm.
Windows/Linux  -> llama-cpp-python (GGUF; CUDA or CPU).

The backend is auto-detected. Override with env var TRANSLATE_LOCAL_BACKEND
set to "mlx" or "ct2".
"""
from __future__ import annotations

import logging
import os
import re

# GGUF (repo_id, filename) keyed by the MLX repo stored in settings, so the
# same settings.json works on both platforms.
_MLX_TO_GGUF = {
    "mlx-community/Qwen2.5-7B-Instruct-4bit": (
        "bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    ),
    "mlx-community/Qwen2.5-14B-Instruct-4bit": (
        "bartowski/Qwen2.5-14B-Instruct-GGUF", "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
    ),
}
_DEFAULT_GGUF = (
    "bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
)

_mlx_models: dict = {}    # cache: model name -> (model, tokenizer)
_gguf_models: dict = {}   # cache: (repo, filename) -> Llama


def backend() -> str:
    """Return the active LLM backend: 'mlx' or 'ct2'."""
    forced = os.getenv("TRANSLATE_LOCAL_BACKEND", "").lower()
    if forced in ("mlx", "ct2"):
        return forced
    try:
        import mlx_lm  # noqa: F401
        return "mlx"
    except ImportError:
        return "ct2"


def chat(messages: list[dict], model: str, max_tokens: int = 2000) -> str:
    """Run a chat completion with a local LLM. Returns the assistant's text.

    Blocking / compute-bound.
    """
    if backend() == "mlx":
        from mlx_lm import generate, load
        if model not in _mlx_models:
            _mlx_models[model] = load(model)
        mdl, tok = _mlx_models[model]
        # Qwen3 models default to a "thinking" mode — disable it where supported.
        try:
            prompt = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
        text = generate(mdl, tok, prompt=prompt, max_tokens=max_tokens, verbose=False)
    else:
        from llama_cpp import Llama
        repo, fname = _MLX_TO_GGUF.get(model, _DEFAULT_GGUF)
        key = (repo, fname)
        if key not in _gguf_models:
            logging.info(f"[llm] loading GGUF model: {repo}/{fname}")
            _gguf_models[key] = Llama.from_pretrained(
                repo_id=repo, filename=fname, n_ctx=8192, verbose=False,
            )
        out = _gguf_models[key].create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=0.7,
        )
        text = out["choices"][0]["message"]["content"]

    # Strip any leftover <think>...</think> reasoning block.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
