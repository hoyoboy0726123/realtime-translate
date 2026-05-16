"""Standalone test for the MLX-Whisper + NLLB local engine.

    python test_whisper.py /tmp/real.wav
"""
import asyncio
import sys
import time
import wave

from app.config import load_settings
from app.engines.local_whisper import WhisperEngine


async def run(wav_path: str) -> None:
    with wave.open(wav_path, "rb") as w:
        assert w.getframerate() == 16000, f"expected 16 kHz, got {w.getframerate()}"
        pcm = w.readframes(w.getnframes())
    print(f"loaded {wav_path}: {len(pcm)/2/16000:.2f}s of audio")

    settings = load_settings()
    out_queue: asyncio.Queue = asyncio.Queue()
    engine = WhisperEngine("zh", "en", out_queue, settings)

    print("building engine (downloads/loads Whisper + NLLB on first run)...")
    t0 = time.time()
    await engine.open()
    print(f"engine ready in {time.time()-t0:.1f}s")

    # Feed audio in 320 ms blocks at real time (one block every 320 ms).
    block = int(0.32 * 16000) * 2
    t0 = time.time()
    for i in range(0, len(pcm), block):
        await engine.send_audio(pcm[i:i + block])
        await asyncio.sleep(0.32)
    for _ in range(4):  # trailing silence to finalize
        await engine.send_audio(b"\x00" * block)
        await asyncio.sleep(0.32)

    deadline = time.time() + 40
    while time.time() < deadline:
        try:
            ev = await asyncio.wait_for(out_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            if engine._audio_q.empty():
                break
            continue
        print(f"  [{ev.kind:7}] spoken={ev.spoken}  A(zh)={ev.text_a!r}  B(en)={ev.text_b!r}")

    print(f"done in {time.time()-t0:.1f}s")
    await engine.close()


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1] if len(sys.argv) > 1 else "/tmp/real.wav"))
