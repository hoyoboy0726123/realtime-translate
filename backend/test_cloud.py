"""Standalone test for the cloud OpenAI translation engine.

Feeds a 16 kHz PCM16 WAV through CloudEngine and prints every TranslationEvent.

    python test_cloud.py /tmp/long.wav
"""
import asyncio
import logging
import sys
import time
import wave

logging.basicConfig(level=logging.INFO, format="%(message)s")

from app.config import load_settings
from app.engines.cloud_openai import CloudEngine


async def run(wav_path: str) -> None:
    with wave.open(wav_path, "rb") as w:
        assert w.getframerate() == 16000, f"expected 16 kHz, got {w.getframerate()}"
        pcm = w.readframes(w.getnframes())
    print(f"loaded {wav_path}: {len(pcm)/2/16000:.2f}s of audio")

    settings = load_settings()
    settings.engine = "cloud"
    out_queue: asyncio.Queue = asyncio.Queue()
    engine = CloudEngine("zh", "en", out_queue, settings)

    print("opening two OpenAI translation sessions...")
    t0 = time.time()
    await engine.open()
    print(f"sessions open in {time.time()-t0:.1f}s")

    async def drain() -> None:
        while True:
            ev = await out_queue.get()
            print(f"  [{ev.kind:7}] spoken={ev.spoken}  A(zh)={ev.text_a!r}  B(en)={ev.text_b!r}")

    drainer = asyncio.create_task(drain())

    # Feed audio in 320 ms blocks at real time.
    block = int(0.32 * 16000) * 2
    t0 = time.time()
    for i in range(0, len(pcm), block):
        await engine.send_audio(pcm[i:i + block])
        await asyncio.sleep(0.32)
    print(f"fed {time.time()-t0:.1f}s of audio; draining 15s more...")
    await asyncio.sleep(15)

    drainer.cancel()
    await engine.close()
    print("done")


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1] if len(sys.argv) > 1 else "/tmp/long.wav"))
