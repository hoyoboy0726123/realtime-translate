"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  API_BASE,
  analyzeTranscript,
  deleteTranscript,
  getTranscript,
  getTranscripts,
  type SessionDetail,
  type SessionSummary,
} from "@/lib/api";

const fmt = (ms: number) => new Date(ms).toLocaleString();

const clock = (ms: number) => {
  const s = Math.floor(ms / 1000);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
};

// Stable colour per speaker label.
const SPEAKER_COLORS = ["#2563eb", "#db2777", "#16a34a", "#d97706", "#7c3aed"];
const speakerColor = (speaker: string) => {
  const n = parseInt(speaker.replace(/\D/g, ""), 10) || 1;
  return SPEAKER_COLORS[(n - 1) % SPEAKER_COLORS.length];
};

export default function HistoryPage() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const refresh = () => {
    getTranscripts()
      .then((s) => {
        setSessions(s);
        setLoading(false);
      })
      .catch((e) => {
        setErr(`無法載入記錄：${e.message}`);
        setLoading(false);
      });
  };

  useEffect(refresh, []);

  // While a session is being analysed, poll it until the job finishes.
  useEffect(() => {
    const stop = () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
    if (detail?.process_status === "processing" && !pollRef.current) {
      pollRef.current = setInterval(() => {
        getTranscript(detail.id)
          .then((d) => {
            setDetail(d);
            if (d.process_status !== "processing") stop();
          })
          .catch(() => {});
      }, 5000);
    }
    return stop;
  }, [detail?.id, detail?.process_status]);

  const remove = async (id: string) => {
    await deleteTranscript(id);
    if (detail?.id === id) setDetail(null);
    refresh();
  };

  const startAnalyze = useCallback(async () => {
    if (!detail) return;
    try {
      await analyzeTranscript(detail.id);
      const d = await getTranscript(detail.id);
      setDetail(d);
    } catch (e) {
      setErr(`無法開始分析：${(e as Error).message}`);
    }
  }, [detail]);

  if (detail) {
    const status = detail.process_status;
    return (
      <div className="page">
        <h1>{detail.name}</h1>
        <p className="sub">
          <button onClick={() => setDetail(null)}>← 返回列表</button>
          <span style={{ marginLeft: 12 }}>
            {detail.engine} · {detail.lang_a}/{detail.lang_b} ·{" "}
            {fmt(detail.started_at)}
          </span>
        </p>

        {/* --- Recording analysis: speaker diarization + meeting summary --- */}
        <div className="card">
          <h2 style={{ marginTop: 0 }}>錄音分析</h2>
          {!detail.audio_path ? (
            <p className="sub">此記錄沒有錄音檔（在錄音功能加入前建立），無法分析。</p>
          ) : status === "processing" ? (
            <p className="sub">
              分析中… 講者辨識與會議摘要進行中，約需數分鐘，完成後會自動顯示。
            </p>
          ) : status === "failed" ? (
            <div>
              <p className="sub">分析失敗。</p>
              <button onClick={startAnalyze}>重新分析</button>
            </div>
          ) : status === "done" ? (
            <button onClick={startAnalyze}>重新分析</button>
          ) : (
            <div>
              <p className="sub">
                辨識不同講者、整理帶講者標籤的逐句記錄，並產生會議摘要（全程在本機執行）。
              </p>
              <button className="primary" onClick={startAnalyze}>
                分析錄音
              </button>
            </div>
          )}
        </div>

        {/* --- Meeting summary --- */}
        {detail.summary && (
          <div className="card">
            <h2 style={{ marginTop: 0 }}>會議摘要</h2>
            <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.7 }}>
              {detail.summary}
            </div>
          </div>
        )}

        {/* --- Speaker-attributed transcript --- */}
        {detail.diarized.length > 0 && (
          <div className="card">
            <h2 style={{ marginTop: 0 }}>逐句記錄（含講者）</h2>
            {detail.diarized.map((d) => (
              <div key={d.idx} className="seg">
                <div className="seg-ts">
                  <span style={{ color: speakerColor(d.speaker), fontWeight: 600 }}>
                    {d.speaker}
                  </span>
                  <span style={{ marginLeft: 8 }}>{clock(d.start_ms)}</span>
                </div>
                <div className="seg-a">
                  ({detail.lang_a}) {d.text_a}
                </div>
                <div className="seg-b">
                  ({detail.lang_b}) {d.text_b}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* --- Live subtitle log --- */}
        <div className="card">
          <h2 style={{ marginTop: 0 }}>即時字幕記錄</h2>
          <div style={{ display: "flex", gap: 10, marginBottom: 12 }}>
            <a
              href={`${API_BASE}/api/transcripts/${detail.id}/export.md`}
              target="_blank"
              rel="noreferrer"
            >
              <button>匯出 Markdown</button>
            </a>
            <a
              href={`${API_BASE}/api/transcripts/${detail.id}/export.json`}
              target="_blank"
              rel="noreferrer"
            >
              <button>匯出 JSON</button>
            </a>
          </div>

          {detail.segments.length === 0 ? (
            <p className="sub">此記錄沒有即時字幕。</p>
          ) : (
            detail.segments.map((s) => (
              <div key={s.id} className="seg">
                <div className="seg-ts">{fmt(s.ts)}</div>
                <div className="seg-a">
                  ({s.lang_a}) {s.text_a}
                </div>
                <div className="seg-b">
                  ({s.lang_b}) {s.text_b}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      <h1>翻譯記錄</h1>
      <p className="sub">
        所有即時翻譯都會被保存，方便日後製作摘要或會議記錄。
        <Link className="navlink" href="/" style={{ marginLeft: 12 }}>
          ← 回到即時翻譯
        </Link>
      </p>

      {err && <div className="notice err">{err}</div>}

      <div className="card">
        {loading ? (
          <p className="sub">載入中…</p>
        ) : sessions.length === 0 ? (
          <p className="sub">目前沒有任何記錄。</p>
        ) : (
          sessions.map((s) => (
            <div key={s.id} className="session-row">
              <div className="meta">
                <div className="title">
                  {s.name}
                  {s.process_status === "done" && (
                    <span className="sub" style={{ marginLeft: 8 }}>✓ 已分析</span>
                  )}
                  {s.process_status === "processing" && (
                    <span className="sub" style={{ marginLeft: 8 }}>分析中…</span>
                  )}
                </div>
                <div className="when">
                  {fmt(s.started_at)} · {s.engine} · {s.lang_a}/{s.lang_b} ·{" "}
                  {s.segment_count} 句
                </div>
              </div>
              <button onClick={() => getTranscript(s.id).then(setDetail)}>
                檢視
              </button>
              <button className="stop" onClick={() => remove(s.id)}>
                刪除
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
