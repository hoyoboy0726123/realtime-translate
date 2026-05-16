"""NLLB-200 text translation, shared by the local engine and post-processing.

Loaded once, run on CPU via HuggingFace transformers (pinned to 4.44.x — see
the requirements-local-*.txt files). `src`/`tgt` are FLORES-200 codes
(see languages.py).
"""
from __future__ import annotations

from .text_guard import is_degenerate


class NllbTranslator:
    """NLLB-200 translation model, loaded once and run on CPU."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._tok = None
        self._model = None
        self._torch = None

    def build(self) -> None:
        """Load the model — heavy, downloads on first use."""
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = (
            AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            .to("cpu")
            .eval()
        )
        self._torch = torch

    def translate(self, text: str, src_nllb: str, tgt_nllb: str) -> str:
        text = text.strip()
        if not text or self._model is None:
            return ""
        self._tok.src_lang = src_nllb
        inputs = self._tok(text, return_tensors="pt", truncation=True, max_length=512)
        bos = self._tok.convert_tokens_to_ids(tgt_nllb)
        with self._torch.inference_mode():
            out = self._model.generate(
                **inputs,
                forced_bos_token_id=bos,
                max_length=512,
                num_beams=1,         # greedy — fast enough for near-real-time
            )
        result = self._tok.batch_decode(out, skip_special_tokens=True)[0].strip()
        # NLLB can also fall into a repetition loop — drop a degenerate result
        # rather than show looping garbage in the pane.
        return "" if is_degenerate(result) else result
