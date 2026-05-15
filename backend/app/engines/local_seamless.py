"""On-device engine backed by Meta's **SeamlessStreaming** model.

SeamlessStreaming performs *simultaneous* speech translation: its EMMA
(Efficient Monotonic Multihead Attention) policy decides, while the speaker is
still talking, when enough audio has arrived to emit the next word — so target
text appears word-by-word rather than one chunk at a time.

The model runs on Meta's `fairseq2` + `simuleval` streaming stack (not plain
transformers). For a locked pair {A, B} we run two streaming pipelines, one per
target language, over the same audio:

  * the pipeline whose target equals the spoken language yields the live
    transcription;
  * the other yields the live translation.

So the top pane (language A) and bottom pane (language B) each receive a true
word-by-word stream regardless of which language is being spoken.

`simuleval.pushpop()` is a blocking, compute-bound call, so all model work runs
on a dedicated worker thread; events are handed back to the asyncio loop with
`loop.call_soon_threadsafe`.

Heavy dependencies are imported lazily — see backend/requirements-local.txt.
"""
from __future__ import annotations

import asyncio
import queue
import threading

import numpy as np

from .. import languages
from ..config import SAMPLE_RATE
from .base import TranslationEngine, TranslationEvent, new_segment_id

# Utterance segmentation (we drive boundaries ourselves; the streaming policy
# still emits words *within* an utterance as audio arrives).
_SILENCE_RMS = 380.0        # int16 RMS below this counts as silence
_SILENCE_HANG = 0.7         # seconds of trailing silence that closes an utterance

# Target languages that are not space-delimited, used when joining streamed
# word pieces back into a line.
_NO_SPACE_LANGS = {"zh", "ja", "th"}

_FLUSH = object()           # sentinel pushed onto the audio queue on close


def _join(pieces: list[str], lang_code: str) -> str:
    cleaned = [p.strip() for p in pieces if p and p.strip()]
    if lang_code in _NO_SPACE_LANGS:
        return "".join(cleaned)
    return " ".join(cleaned)


def _pieces_of(output) -> list[str]:
    """Pull text out of whatever `pushpop` returned (segment, list, or None)."""
    if output is None:
        return []
    segments = output if isinstance(output, list) else [output]
    pieces: list[str] = []
    for seg in segments:
        content = getattr(seg, "content", None)
        if isinstance(content, str):
            if content:
                pieces.append(content)
        elif isinstance(content, (list, tuple)):
            pieces.extend(str(c) for c in content if c)
    return pieces


class _Pipeline:
    """One SeamlessStreaming speech-to-text system fixed to a single target language."""

    def __init__(self, tgt_lang_seamless: str, settings):
        self.tgt_lang = tgt_lang_seamless
        self.device = settings.local.device
        self.segment_size_ms = settings.local.source_segment_size_ms
        self.decision_threshold = settings.local.decision_threshold
        self.unity_model = settings.local.unity_model
        self.monotonic_decoder_model = settings.local.monotonic_decoder_model
        self._system = None
        self._states = None

    def build(self) -> None:
        """Construct the streaming agent system. First call downloads checkpoints."""
        try:
            from simuleval.utils.agent import build_system_args
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError(
                "SeamlessStreaming requires 'simuleval' and 'seamless_communication'. "
                "Install backend/requirements-local.txt, then: "
                "pip install git+https://github.com/facebookresearch/seamless_communication.git"
            ) from exc

        dtype = "fp16" if self.device.startswith("cuda") else "fp32"
        config = {
            # Non-VAD speech-to-text streaming pipeline; we segment utterances
            # ourselves so both language pipelines share the same boundaries.
            "agent_class": (
                "seamless_communication.streaming.agents."
                "seamless_streaming_s2t.SeamlessStreamingS2TAgent"
            ),
            "task": "s2st",
            "unity_model_name": self.unity_model,
            "monotonic_decoder_model_name": self.monotonic_decoder_model,
            "sentencepiece_model": "spm_256k.model",
            "tgt_lang": self.tgt_lang,
            "device": self.device,
            "dtype": dtype,
            "source_segment_size": self.segment_size_ms,
            "decision_threshold": self.decision_threshold,
            "min_unit_chunk_size": 50,
            "no_early_stop": True,
            "detokenize_only": True,
        }
        self._system, _ = build_system_args(config)
        self._states = self._system.build_states()

    def reset(self) -> None:
        """Start a fresh utterance with clean decoder state."""
        self._states = self._system.build_states()

    def push(self, samples: np.ndarray, finished: bool) -> list[str]:
        """Feed one audio block; return any word pieces the policy chose to emit."""
        from simuleval.data.segments import SpeechSegment

        segment = SpeechSegment(
            content=samples,
            sample_rate=SAMPLE_RATE,
            finished=finished,
            tgt_lang=self.tgt_lang,
        )
        return _pieces_of(self._system.pushpop(segment, self._states))


class LocalEngine(TranslationEngine):
    name = "local"

    def __init__(self, lang_a, lang_b, out_queue, settings):
        super().__init__(lang_a, lang_b, out_queue, settings)
        self._pipe_a = _Pipeline(languages.get(lang_a)["seamless"], settings)
        self._pipe_b = _Pipeline(languages.get(lang_b)["seamless"], settings)
        self._block_bytes = int(settings.local.source_segment_size_ms * SAMPLE_RATE / 1000) * 2
        self._block_dur = self._block_bytes / 2 / SAMPLE_RATE
        self._audio_q: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def open(self) -> None:
        self._loop = asyncio.get_running_loop()
        # Building both pipelines loads the model — heavy, possibly a download.
        await asyncio.to_thread(self._build)
        self._worker = threading.Thread(target=self._run, name="seamless-stream", daemon=True)
        self._worker.start()

    def _build(self) -> None:
        self._pipe_a.build()
        self._pipe_b.build()

    async def send_audio(self, pcm16: bytes) -> None:
        self._audio_q.put(pcm16)

    async def close(self) -> None:
        self._audio_q.put(_FLUSH)
        if self._worker:
            await asyncio.to_thread(self._worker.join, 15)

    # ---- worker thread ---------------------------------------------------

    def _emit(self, kind: str, seg_id: str, parts_a: list[str], parts_b: list[str]) -> None:
        text_a = _join(parts_a, self.lang_a)
        text_b = _join(parts_b, self.lang_b)
        if not text_a and not text_b:
            return
        event = TranslationEvent(
            kind=kind, segment_id=seg_id,
            lang_a=self.lang_a, lang_b=self.lang_b,
            text_a=text_a, text_b=text_b,
        )
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.out_queue.put_nowait, event)

    def _emit_error(self, message: str) -> None:
        event = TranslationEvent(
            kind="final", segment_id=new_segment_id(),
            lang_a=self.lang_a, lang_b=self.lang_b,
            text_a=f"[engine error] {message}", text_b="",
        )
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.out_queue.put_nowait, event)

    def _run(self) -> None:
        block = bytearray()
        seg_id = new_segment_id()
        parts_a: list[str] = []
        parts_b: list[str] = []
        silence = 0.0
        speech_seen = False

        def finalize(tail: np.ndarray) -> None:
            nonlocal seg_id, parts_a, parts_b, silence, speech_seen
            parts_a.extend(self._pipe_a.push(tail, finished=True))
            parts_b.extend(self._pipe_b.push(tail, finished=True))
            self._emit("final", seg_id, parts_a, parts_b)
            self._pipe_a.reset()
            self._pipe_b.reset()
            seg_id = new_segment_id()
            parts_a, parts_b = [], []
            silence = 0.0
            speech_seen = False

        empty = np.zeros(0, dtype=np.float32)

        while True:
            item = self._audio_q.get()
            if item is _FLUSH:
                if speech_seen:
                    try:
                        finalize(empty)
                    except Exception as exc:  # noqa: BLE001
                        self._emit_error(str(exc))
                return

            block.extend(item)
            try:
                while len(block) >= self._block_bytes:
                    chunk = bytes(block[: self._block_bytes])
                    del block[: self._block_bytes]

                    ints = np.frombuffer(chunk, dtype=np.int16)
                    rms = float(np.sqrt(np.mean(ints.astype(np.float32) ** 2)))
                    is_speech = rms >= _SILENCE_RMS

                    # Drop leading silence so the decoder isn't fed dead air.
                    if not speech_seen and not is_speech:
                        continue
                    speech_seen = True

                    samples = ints.astype(np.float32) / 32768.0
                    new_a = self._pipe_a.push(samples, finished=False)
                    new_b = self._pipe_b.push(samples, finished=False)
                    if new_a or new_b:
                        parts_a.extend(new_a)
                        parts_b.extend(new_b)
                        self._emit("partial", seg_id, parts_a, parts_b)

                    silence = 0.0 if is_speech else silence + self._block_dur
                    if silence >= _SILENCE_HANG:
                        finalize(empty)
            except Exception as exc:  # noqa: BLE001 - surface model failures to the client
                self._emit_error(str(exc))
                return
