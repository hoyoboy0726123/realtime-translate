"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { API_BASE } from "@/lib/api";

/** Renders a meeting summary as formatted markdown, with md / docx download. */
export default function SummaryView({
  sessionId,
  summary,
}: {
  sessionId: string;
  summary: string;
}) {
  const base = `${API_BASE}/api/transcripts/${sessionId}`;
  return (
    <div>
      <div style={{ display: "flex", gap: 10, marginBottom: 12, flexWrap: "wrap" }}>
        <a href={`${base}/summary.md`} target="_blank" rel="noreferrer">
          <button>下載 Markdown</button>
        </a>
        <a href={`${base}/summary.docx`} target="_blank" rel="noreferrer">
          <button>下載 Word（docx）</button>
        </a>
      </div>
      <div className="markdown">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{summary}</ReactMarkdown>
      </div>
    </div>
  );
}
