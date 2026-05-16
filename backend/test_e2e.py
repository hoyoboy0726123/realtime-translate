"""End-to-end local pipeline test: live subtitles -> recording -> analyze.

Streams a meeting WAV through the real WebSocket, collects the live subtitles,
then triggers post-session analysis and prints the diarized transcript + summary.

    python test_e2e.py /tmp/meeting.wav
"""
import asyncio
import json
import sys
import time
import urllib.request
import wave

import websockets

WS = "ws://localhost:8000/ws/translate"
API = "http://localhost:8000"


async def run(wav_path: str) -> None:
    with wave.open(wav_path, "rb") as w:
        pcm = w.readframes(w.getnframes())
    print(f"meeting audio: {len(pcm)/2/16000:.1f}s\n")

    finals: list[dict] = []
    async with websockets.connect(WS, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "session_name": "地端一條龍測試會議"}))
        started = json.loads(await ws.recv())
        session_id = started["session_id"]
        print(f"session {session_id} · engine={started['engine']}\n")

        async def reader() -> None:
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") == "segment" and msg.get("kind") == "final":
                        finals.append(msg)
                        print(f"  [live] A={msg['text_a']}  |  B={msg['text_b']}")
            except Exception:
                pass

        rt = asyncio.create_task(reader())

        print("=== streaming audio (real time) ===")
        block = int(0.32 * 16000) * 2
        t0 = time.time()
        for i in range(0, len(pcm), block):
            await ws.send(pcm[i:i + block])
            await asyncio.sleep(0.32)
        for _ in range(5):  # trailing silence
            await ws.send(b"\x00" * block)
            await asyncio.sleep(0.32)
        await asyncio.sleep(4)  # let trailing finals arrive
        await ws.send(json.dumps({"type": "stop"}))
        rt.cancel()
        print(f"\nstreamed in {time.time()-t0:.0f}s, {len(finals)} live final segments")

    # --- post-session analysis ---
    print("\n=== triggering analyze ===")
    req = urllib.request.Request(f"{API}/api/transcripts/{session_id}/analyze", method="POST")
    print("  ", urllib.request.urlopen(req).read().decode())

    t0 = time.time()
    while True:
        await asyncio.sleep(10)
        data = json.loads(
            urllib.request.urlopen(f"{API}/api/transcripts/{session_id}").read()
        )
        print(f"  status: {data['process_status']} ({time.time()-t0:.0f}s)")
        if data["process_status"] != "processing":
            break

    print(f"\n=== DIARIZED TRANSCRIPT ({len(data['diarized'])} utterances) ===")
    for d in data["diarized"]:
        print(f"  {d['speaker']} [{d['start_ms']//1000:>3}s]  {d['text_a']}")
        print(f"             {' ' * 6}{d['text_b']}")
    print("\n=== MEETING SUMMARY ===")
    print(data["summary"])


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1] if len(sys.argv) > 1 else "/tmp/meeting.wav"))
