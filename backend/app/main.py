"""FastAPI application entry point for the real-time translation backend."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import db, ws
from .config import cors_origins
from .routers import settings, transcripts


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="Realtime Translate", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(settings.router)
app.include_router(transcripts.router)
app.include_router(ws.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
