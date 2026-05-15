"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  API_BASE,
  deleteTranscript,
  getTranscript,
  getTranscripts,
  type SessionDetail,
  type SessionSummary,
} from "@/lib/api";

const fmt = (ms: number) => new Date(ms).toLocaleString();

export default function HistoryPage() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

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

  const remove = async (id: string) => {
    await deleteTranscript(id);
    if (detail?.id === id) setDetail(null);
    refresh();
  };

  if (detail) {
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

        <div className="card">
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
            <p className="sub">此記錄沒有任何字幕。</p>
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
                <div className="title">{s.name}</div>
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
