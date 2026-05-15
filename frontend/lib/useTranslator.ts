"use client";

import { useCallback, useRef, useState } from "react";
import { WS_BASE } from "./api";

export type Status = "idle" | "connecting" | "live" | "error";

export interface LiveSegment {
  id: string;
  textA: string;
  textB: string;
  kind: "partial" | "final";
  spoken: string | null;
  ts: number;
}

export interface SessionInfo {
  sessionId?: string;
  engine?: string;
  langA?: string;
  langB?: string;
}

const MAX_SEGMENTS = 40;

export function useTranslator() {
  const [status, setStatus] = useState<Status>("idle");
  const [segments, setSegments] = useState<LiveSegment[]>([]);
  const [info, setInfo] = useState<SessionInfo>({});
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const nodeRef = useRef<AudioWorkletNode | null>(null);

  const teardownAudio = useCallback(() => {
    nodeRef.current?.disconnect();
    nodeRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    ctxRef.current?.close().catch(() => {});
    ctxRef.current = null;
  }, []);

  const stop = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stop" }));
    }
    ws?.close();
    wsRef.current = null;
    teardownAudio();
    setStatus("idle");
  }, [teardownAudio]);

  const startAudio = useCallback(async (ws: WebSocket) => {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    streamRef.current = stream;

    const ctx = new AudioContext();
    ctxRef.current = ctx;
    await ctx.audioWorklet.addModule("/pcm-worklet.js");

    const source = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, "pcm-worklet");
    nodeRef.current = node;
    node.port.onmessage = (e: MessageEvent<ArrayBuffer>) => {
      if (ws.readyState === WebSocket.OPEN) ws.send(e.data);
    };
    source.connect(node);
    node.connect(ctx.destination); // processor emits silence; keeps the graph alive
  }, []);

  const start = useCallback(
    (sessionName: string) => {
      setError(null);
      setSegments([]);
      setInfo({});
      setStatus("connecting");

      const ws = new WebSocket(`${WS_BASE}/ws/translate`);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({ type: "start", session_name: sessionName }));
      };

      ws.onmessage = async (ev) => {
        const msg = JSON.parse(ev.data as string);
        if (msg.type === "started") {
          setInfo({
            sessionId: msg.session_id,
            engine: msg.engine,
            langA: msg.lang_a,
            langB: msg.lang_b,
          });
          try {
            await startAudio(ws);
            setStatus("live");
          } catch (e) {
            setError(`Microphone error: ${(e as Error).message}`);
            setStatus("error");
            ws.close();
          }
        } else if (msg.type === "segment") {
          setSegments((prev) => {
            const seg: LiveSegment = {
              id: msg.segment_id,
              textA: msg.text_a,
              textB: msg.text_b,
              kind: msg.kind,
              spoken: msg.spoken,
              ts: msg.ts,
            };
            const idx = prev.findIndex((s) => s.id === seg.id);
            const next = idx >= 0 ? [...prev] : [...prev, seg];
            if (idx >= 0) next[idx] = seg;
            return next.slice(-MAX_SEGMENTS);
          });
        } else if (msg.type === "error") {
          setError(msg.message);
          setStatus("error");
        }
      };

      ws.onerror = () => {
        setError("Connection error — is the backend running?");
        setStatus("error");
      };

      ws.onclose = () => {
        teardownAudio();
        setStatus((s) => (s === "error" ? s : "idle"));
      };
    },
    [startAudio, teardownAudio],
  );

  return { status, segments, info, error, start, stop };
}
