"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  API_BASE,
  analyzeTranscript,
  cancelAnalysis,
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

// Analysis pipeline stages, in order.
const STAGES = [
  { key: "diarizing", label: "轉錄與講者辨識" },
  { key: "translating", label: "翻譯" },
  { key: "summarizing", label: "產生摘要" },
];
const PROCESSING = [
  "processing", "downloading", "diarizing", "translating", "summarizing",
];
const isProcessing = (s: string | null | undefined) => PROCESSING.includes(s ?? "");
const stageIndex = (s: string | null | undefined) => {
  const i = STAGES.findIndex((st) => st.key === s);
  return i < 0 ? 0 : i; // legacy "processing" -> first stage
};

const audioUrl = (id: string) => `${API_BASE}/api/transcripts/${id}/audio`;

export default function HistoryPage() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);     // seconds the analysis has run
  const [playingId, setPlayingId] = useState<string | null>(null);
  const startRef = useRef<number | null>(null);

  const refresh = () => {
    getTranscripts()
      .then((s) => {
        // Uploaded files live on their own page (/transcribe).
        setSessions(s.filter((x) => x.engine !== "upload"));
        setLoading(false);
      })
      .catch((e) => {
        setErr(`無法載入記錄：${e.message}`);
        setLoading(false);
      });
  };

  useEffect(refresh, []);

  // While a session is being analysed: tick the elapsed timer every second and
  // poll the session every few seconds until the job finishes.
  useEffect(() => {
    if (!detail || !isProcessing(detail.process_status)) {
      startRef.current = null;
      return;
    }
    if (startRef.current == null) {
      startRef.current = Date.now();
      setElapsed(0);
    }
    let tick = 0;
    const id = setInterval(() => {
      setElapsed(Math.floor((Date.now() - (startRef.current ?? Date.now())) / 1000));
      tick += 1;
      if (tick % 4 === 0) {
        getTranscript(detail.id).then(setDetail).catch(() => {});
      }
    }, 1000);
    return () => clearInterval(id);
  }, [detail?.id, detail?.process_status]);

  const remove = async (id: string) => {
    await deleteTranscript(id);
    if (detail?.id === id) setDetail(null);
    if (playingId === id) setPlayingId(null);
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

  const cancelAnalyze = useCallback(async () => {
    if (!detail) return;
    try {
      await cancelAnalysis(detail.id);
      setDetail(await getTranscript(detail.id));
    } catch (e) {
      setErr(`無法停止分析：${(e as Error).message}`);
    }
  }, [detail]);

  if (detail) {
    const status = detail.process_status;
    const cur = stageIndex(status);
    return (
      <div className="page">
        <h1>{detail.name}</h1>
        <p className="sub">
          <button
            onClick={() => {
              setDetail(null);
              refresh(); // re-fetch so the list shows up-to-date analysis status
            }}
          >
            ← 返回列表
          </button>
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
          ) : status === "downloading" ? (
            <div>
              <p className="sub">
                ⬇ 首次使用，正在下載所需模型…（語音辨識／翻譯／摘要模型，合計約數 GB，
                只需下載一次；詳細進度可在後端終端機查看）　已 {clock(elapsed * 1000)}
              </p>
              <button className="stop" style={{ marginTop: 10 }} onClick={cancelAnalyze}>
                停止
              </button>
            </div>
          ) : isProcessing(status) ? (
            <div>
              <p className="sub">分析中… 已 {clock(elapsed * 1000)}（離線重新轉錄＋摘要，約需數分鐘）</p>
              <div style={{ display: "flex", gap: 8, marginTop: 10, flexWrap: "wrap" }}>
                {STAGES.map((st, i) => {
                  const done = i < cur;
                  const active = i === cur;
                  return (
                    <div
                      key={st.key}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                        padding: "6px 12px",
                        borderRadius: 999,
                        border: "1px solid var(--border)",
                        opacity: done || active ? 1 : 0.4,
                        background: active ? "var(--panel)" : "transparent",
                      }}
                    >
                      <span
                        style={{
                          color: done
                            ? "var(--ok)"
                            : active
                              ? "var(--accent-b)"
                              : "var(--muted)",
                          fontWeight: 700,
                        }}
                      >
                        {done ? "✓" : active ? "●" : i + 1}
                      </span>
                      <span>{st.label}</span>
                    </div>
                  );
                })}
              </div>
              <button className="stop" style={{ marginTop: 10 }} onClick={cancelAnalyze}>
                停止
              </button>
            </div>
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
            <div style={{ display: "flex", gap: 10, marginBottom: 12, flexWrap: "wrap" }}>
              {[
                { track: "both", label: "匯出 SRT（雙語）" },
                { track: "a", label: `匯出 SRT（${detail.lang_a}）` },
                { track: "b", label: `匯出 SRT（${detail.lang_b}）` },
              ].map((o) => (
                <a
                  key={o.track}
                  href={`${API_BASE}/api/transcripts/${detail.id}/export.srt?track=${o.track}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  <button>{o.label}</button>
                </a>
              ))}
            </div>
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
        <Link className="navlink" href="/transcribe" style={{ marginLeft: 8 }}>
          檔案轉錄
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
            <div key={s.id}>
              <div className="session-row">
                <div className="meta">
                  <div className="title">
                    {s.name}
                    {s.process_status === "done" && (
                      <span className="sub" style={{ marginLeft: 8 }}>✓ 已分析</span>
                    )}
                    {isProcessing(s.process_status) && (
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
                {s.audio_path && (
                  <button
                    onClick={() =>
                      setPlayingId(playingId === s.id ? null : s.id)
                    }
                  >
                    {playingId === s.id ? "⏸ 收合" : "▶ 播放錄音"}
                  </button>
                )}
                <button className="stop" onClick={() => remove(s.id)}>
                  刪除
                </button>
              </div>
              {playingId === s.id && s.audio_path && (
                <audio
                  controls
                  autoPlay
                  src={audioUrl(s.id)}
                  style={{ width: "100%", marginTop: 6 }}
                />
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
