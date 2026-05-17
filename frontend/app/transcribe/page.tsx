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
  uploadMedia,
  type AnalyzeStage,
  type SessionDetail,
  type SessionSummary,
} from "@/lib/api";

const fmt = (ms: number) => new Date(ms).toLocaleString();

const clock = (ms: number) => {
  const s = Math.floor(ms / 1000);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
};

const SPEAKER_COLORS = ["#2563eb", "#db2777", "#16a34a", "#d97706", "#7c3aed"];
const speakerColor = (speaker: string) => {
  const n = parseInt(speaker.replace(/\D/g, ""), 10) || 1;
  return SPEAKER_COLORS[(n - 1) % SPEAKER_COLORS.length];
};

// Analysis pipeline stages, in order.
const STAGES = [
  { key: "diarizing", label: "轉錄" },
  { key: "translating", label: "翻譯" },
  { key: "summarizing", label: "產生摘要" },
];
const PROCESSING = ["processing", "diarizing", "translating", "summarizing"];
const isProcessing = (s: string | null | undefined) => PROCESSING.includes(s ?? "");
const stageIndex = (s: string | null | undefined) => {
  const i = STAGES.findIndex((st) => st.key === s);
  return i < 0 ? 0 : i;
};

export default function TranscribePage() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [identifySpeakers, setIdentifySpeakers] = useState(false);
  const [recFilter, setRecFilter] = useState("");
  const startRef = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(() => {
    getTranscripts()
      .then((all) => {
        setSessions(all);
        setLoading(false);
      })
      .catch((e) => {
        setErr(`無法載入清單：${e.message}`);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // While a stage is running: tick the timer and poll until it finishes.
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
        getTranscript(detail.id)
          .then((d) => {
            setDetail(d);
            if (!isProcessing(d.process_status)) refresh();
          })
          .catch(() => {});
      }
    }, 1000);
    return () => clearInterval(id);
  }, [detail?.id, detail?.process_status, refresh]);

  const handleUpload = useCallback(
    async (f: File) => {
      setUploading(true);
      setErr(null);
      try {
        const { session_id } = await uploadMedia(f);
        const d = await getTranscript(session_id);
        setDetail(d);
        refresh();
      } catch (e) {
        setErr(`上傳失敗：${(e as Error).message}`);
      } finally {
        setUploading(false);
        if (fileInputRef.current) fileInputRef.current.value = "";
      }
    },
    [refresh],
  );

  const runStage = useCallback(
    async (stage: AnalyzeStage) => {
      if (!detail) return;
      setErr(null);
      try {
        await analyzeTranscript(detail.id, stage, identifySpeakers);
        setDetail(await getTranscript(detail.id));
      } catch (e) {
        setErr(`無法開始：${(e as Error).message}`);
      }
    },
    [detail, identifySpeakers],
  );

  const runCancel = useCallback(async () => {
    if (!detail) return;
    setErr(null);
    try {
      await cancelAnalysis(detail.id);
      setDetail(await getTranscript(detail.id));
    } catch (e) {
      setErr(`無法停止：${(e as Error).message}`);
    }
  }, [detail]);

  const remove = useCallback(
    async (id: string) => {
      await deleteTranscript(id);
      if (detail?.id === id) setDetail(null);
      refresh();
    },
    [detail, refresh],
  );

  const open = useCallback(
    (id: string) => {
      getTranscript(id).then(setDetail).catch(() => {});
    },
    [],
  );

  const uploads = sessions.filter((s) => s.engine === "upload");
  const recordings = sessions.filter(
    (s) => s.engine !== "upload" && s.audio_path,
  );
  const recQuery = recFilter.trim().toLowerCase();
  const filteredRecordings = recQuery
    ? recordings.filter((s) => s.name.toLowerCase().includes(recQuery))
    : recordings;

  return (
    <div className="page">
      <h1>音訊／影片轉錄</h1>
      <p className="sub">
        上傳外部檔案，或挑選本 App 既有的錄音檔，獨立轉錄、翻譯、製作字幕或摘要。
        <Link className="navlink" href="/" style={{ marginLeft: 12 }}>
          ← 回到即時翻譯
        </Link>
        <Link className="navlink" href="/history" style={{ marginLeft: 8 }}>
          翻譯記錄
        </Link>
      </p>

      {err && <div className="notice err">{err}</div>}

      {/* --- Upload --- */}
      <div className="card">
        <h2 style={{ marginTop: 0 }}>上傳檔案</h2>
        <p className="sub">
          支援 mp3、m4a、wav、flac、aac、ogg、mp4、mov、mkv、webm 等格式（由 ffmpeg 解碼）。
        </p>
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*,video/*"
          disabled={uploading}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleUpload(f);
          }}
        />
        {uploading && (
          <p className="sub" style={{ marginTop: 8 }}>
            上傳並解碼中…（大型影片可能需要一點時間）
          </p>
        )}
      </div>

      {/* --- Selected file: staged workflow --- */}
      {detail && (
        <FileWorkflow
          detail={detail}
          elapsed={elapsed}
          runStage={runStage}
          runCancel={runCancel}
          identifySpeakers={identifySpeakers}
          setIdentifySpeakers={setIdentifySpeakers}
        />
      )}

      {/* --- Project recordings --- */}
      <div className="card">
        <h2 style={{ marginTop: 0 }}>專案內的錄音檔</h2>
        <p className="sub" style={{ marginTop: 0 }}>
          本 App 即時翻譯時錄下的錄音，可在此挑選並做轉錄、字幕或摘要處理。
        </p>
        {recordings.length > 0 && (
          <input
            type="text"
            placeholder="輸入名稱搜尋錄音檔…"
            value={recFilter}
            onChange={(e) => setRecFilter(e.target.value)}
            style={{ width: "100%", marginBottom: 10, padding: "6px 10px" }}
          />
        )}
        {loading ? (
          <p className="sub">載入中…</p>
        ) : recordings.length === 0 ? (
          <p className="sub">目前沒有任何錄音檔。</p>
        ) : filteredRecordings.length === 0 ? (
          <p className="sub">找不到符合「{recFilter}」的錄音檔。</p>
        ) : (
          filteredRecordings.map((s) => (
            <SessionRow key={s.id} s={s} onOpen={() => open(s.id)} />
          ))
        )}
      </div>

      {/* --- Uploaded files --- */}
      <div className="card">
        <h2 style={{ marginTop: 0 }}>已上傳的檔案</h2>
        {loading ? (
          <p className="sub">載入中…</p>
        ) : uploads.length === 0 ? (
          <p className="sub">尚未上傳任何檔案。</p>
        ) : (
          uploads.map((s) => (
            <SessionRow
              key={s.id}
              s={s}
              onOpen={() => open(s.id)}
              onDelete={() => remove(s.id)}
            />
          ))
        )}
      </div>
    </div>
  );
}

function SessionRow({
  s,
  onOpen,
  onDelete,
}: {
  s: SessionSummary;
  onOpen: () => void;
  onDelete?: () => void;
}) {
  return (
    <div className="session-row">
      <div className="meta">
        <div className="title">
          {s.name}
          {s.process_status === "done" && (
            <span className="sub" style={{ marginLeft: 8 }}>
              {s.summary ? "✓ 含摘要" : "✓ 已轉錄"}
            </span>
          )}
          {isProcessing(s.process_status) && (
            <span className="sub" style={{ marginLeft: 8 }}>處理中…</span>
          )}
          {s.process_status === "failed" && (
            <span className="sub" style={{ marginLeft: 8 }}>✗ 失敗</span>
          )}
        </div>
        <div className="when">
          {fmt(s.started_at)} · {s.engine} · {s.lang_a}/{s.lang_b}
        </div>
      </div>
      <button onClick={onOpen}>開啟</button>
      {onDelete && (
        <button className="stop" onClick={onDelete}>
          刪除
        </button>
      )}
    </div>
  );
}

function FileWorkflow({
  detail,
  elapsed,
  runStage,
  runCancel,
  identifySpeakers,
  setIdentifySpeakers,
}: {
  detail: SessionDetail;
  elapsed: number;
  runStage: (stage: AnalyzeStage) => void;
  runCancel: () => void;
  identifySpeakers: boolean;
  setIdentifySpeakers: (v: boolean) => void;
}) {
  const status = detail.process_status;
  const processing = isProcessing(status);
  const cur = stageIndex(status);
  const hasTranscript = detail.diarized.length > 0;
  const hasSummary = !!detail.summary;

  return (
    <div className="card" style={{ borderColor: "var(--accent-b)" }}>
      <h2 style={{ marginTop: 0 }}>{detail.name}</h2>

      {/* Recording player */}
      {detail.audio_path && (
        <audio
          controls
          src={`${API_BASE}/api/transcripts/${detail.id}/audio`}
          style={{ width: "100%", marginBottom: 12 }}
        />
      )}

      {/* Progress timeline */}
      {processing && (
        <div style={{ marginBottom: 14 }}>
          <p className="sub">處理中… 已 {clock(elapsed * 1000)}（全程在本機執行）</p>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
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
                      color: done ? "var(--ok)" : active ? "var(--accent-b)" : "var(--muted)",
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
          <button className="stop" style={{ marginTop: 10 }} onClick={runCancel}>
            停止
          </button>
        </div>
      )}

      {status === "failed" && (
        <p className="notice err">處理失敗，可重新嘗試下方步驟。</p>
      )}

      {/* Step 1 — transcribe + translate */}
      <div style={{ marginBottom: 14 }}>
        <h3 style={{ margin: "0 0 6px" }}>
          步驟一 · 轉錄與翻譯 {hasTranscript && <span className="sub">✓ 已完成</span>}
        </h3>
        <p className="sub" style={{ marginTop: 0 }}>
          逐句轉錄並翻譯成中英雙語。完成後即可匯出 SRT 字幕或逐字稿。
        </p>
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            marginBottom: 8,
            cursor: processing ? "default" : "pointer",
          }}
        >
          <input
            type="checkbox"
            checked={identifySpeakers}
            disabled={processing}
            onChange={(e) => setIdentifySpeakers(e.target.checked)}
          />
          <span>識別講者（分辨不同說話者；不勾選則僅轉錄文字）</span>
        </label>
        <button onClick={() => runStage("transcript")} disabled={processing}>
          {hasTranscript ? "重新轉錄" : "開始轉錄與翻譯"}
        </button>
      </div>

      {/* Step 2 — summary */}
      <div style={{ marginBottom: hasTranscript || hasSummary ? 14 : 0 }}>
        <h3 style={{ margin: "0 0 6px" }}>
          步驟二 · 會議摘要 {hasSummary && <span className="sub">✓ 已完成</span>}
        </h3>
        <p className="sub" style={{ marginTop: 0 }}>
          以本機 LLM 產生摘要。可單獨執行 —— 若尚未轉錄會自動先完成轉錄。
        </p>
        <button onClick={() => runStage("summary")} disabled={processing}>
          {hasSummary ? "重新產生摘要" : "產生摘要"}
        </button>
      </div>

      {/* Export */}
      {hasTranscript && (
        <div style={{ marginBottom: 14 }}>
          <h3 style={{ margin: "0 0 6px" }}>匯出字幕 / 文字</h3>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            {[
              {
                label: "SRT（雙語）",
                href: `${API_BASE}/api/transcripts/${detail.id}/export.srt?track=both`,
              },
              {
                label: `SRT（${detail.lang_a}）`,
                href: `${API_BASE}/api/transcripts/${detail.id}/export.srt?track=a`,
              },
              {
                label: `SRT（${detail.lang_b}）`,
                href: `${API_BASE}/api/transcripts/${detail.id}/export.srt?track=b`,
              },
              {
                label: "逐字稿（txt）",
                href: `${API_BASE}/api/transcripts/${detail.id}/export.txt?track=both`,
              },
            ].map((o) => (
              <a key={o.label} href={o.href} target="_blank" rel="noreferrer">
                <button>{o.label}</button>
              </a>
            ))}
          </div>
        </div>
      )}

      {/* Summary */}
      {hasSummary && (
        <div className="card" style={{ marginBottom: 14 }}>
          <h3 style={{ marginTop: 0 }}>會議摘要</h3>
          <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.7 }}>{detail.summary}</div>
        </div>
      )}

      {/* Transcript */}
      {hasTranscript && (
        <div>
          <h3>{detail.diarized.some((d) => d.speaker) ? "逐句記錄（含講者）" : "逐句記錄"}</h3>
          {detail.diarized.map((d) => (
            <div key={d.idx} className="seg">
              <div className="seg-ts">
                {d.speaker && (
                  <span style={{ color: speakerColor(d.speaker), fontWeight: 600 }}>
                    {d.speaker}
                  </span>
                )}
                <span style={{ marginLeft: d.speaker ? 8 : 0 }}>
                  {clock(d.start_ms)}
                </span>
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
    </div>
  );
}
