"""WebSocket endpoint that streams audio in and bilingual subtitles out.

Client protocol
---------------
1. Client connects to `/ws/translate`.
2. Client sends a JSON text frame: `{"type": "start", "session_name": "..."}`.
3. Server replies `{"type": "started", "session_id", "engine", "lang_a", "lang_b"}`.
4. Client streams binary frames of 16 kHz mono PCM16 audio.
5. Server pushes `{"type": "segment", ...}` frames (partial + final).
6. Client sends `{"type": "stop"}` or simply disconnects to end the session.
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from . import db
from .config import load_settings
from .engines.base import TranslationEvent
from .engines.registry import create_engine

router = APIRouter()


async def _forward(websocket: WebSocket, queue: "asyncio.Queue[TranslationEvent]", session_id: str) -> None:
    """Relay engine events to the client and persist finalised segments."""
    while True:
        event = await queue.get()
        message = event.to_message()
        try:
            await websocket.send_json(message)
        except RuntimeError:
            return
        if event.kind == "final" and (event.text_a or event.text_b):
            await asyncio.to_thread(db.add_segment, session_id, message)


@router.websocket("/ws/translate")
async def translate(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        first = await websocket.receive_json()
    except (WebSocketDisconnect, json.JSONDecodeError):
        await websocket.close()
        return

    if first.get("type") != "start":
        await websocket.send_json({"type": "error", "message": "expected a 'start' message"})
        await websocket.close()
        return

    settings = load_settings()
    session_name = first.get("session_name") or "Untitled session"
    session_id = await asyncio.to_thread(
        db.create_session, session_name, settings.engine, settings.lang_a, settings.lang_b,
    )

    queue: asyncio.Queue[TranslationEvent] = asyncio.Queue()
    engine = create_engine(settings, queue)

    try:
        await engine.open()
    except Exception as exc:  # noqa: BLE001 - surface any engine startup failure to the client
        await websocket.send_json({"type": "error", "message": f"engine failed to start: {exc}"})
        await asyncio.to_thread(db.end_session, session_id)
        await websocket.close()
        return

    await websocket.send_json({
        "type": "started",
        "session_id": session_id,
        "engine": settings.engine,
        "lang_a": settings.lang_a,
        "lang_b": settings.lang_b,
    })

    forwarder = asyncio.create_task(_forward(websocket, queue, session_id))

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if (data := message.get("bytes")) is not None:
                await engine.send_audio(data)
            elif (text := message.get("text")) is not None:
                payload = json.loads(text)
                if payload.get("type") == "stop":
                    break
    except WebSocketDisconnect:
        pass
    finally:
        await engine.close()
        forwarder.cancel()
        await asyncio.to_thread(db.end_session, session_id)
        try:
            await websocket.close()
        except RuntimeError:
            pass
