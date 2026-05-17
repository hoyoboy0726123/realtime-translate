export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://localhost:8000";
export const WS_BASE = API_BASE.replace(/^http/, "ws");

export interface Language {
  code: string;
  name: string;
  native: string;
  seamless: string;
  openai: string;
}

export interface Settings {
  engine: string;
  lang_a: string;
  lang_b: string;
  chinese_variant: string;
  cloud: { model: string };
  local: {
    whisper_model: string;
    analyze_whisper_model: string;
    translate_model: string;
    summary_model: string;
  };
}

export interface SessionSummary {
  id: string;
  name: string;
  engine: string;
  lang_a: string;
  lang_b: string;
  started_at: number;
  ended_at: number | null;
  segment_count: number;
  audio_path: string | null;
  process_status: string | null;   // null | processing | done | failed
  processed_at: number | null;
  summary: string | null;
}

export interface StoredSegment {
  id: string;
  ts: number;
  lang_a: string;
  lang_b: string;
  text_a: string;
  text_b: string;
  spoken: string | null;
}

export interface DiarizedSegment {
  idx: number;
  speaker: string;
  start_ms: number;
  end_ms: number;
  text_a: string;
  text_b: string;
}

export interface SessionDetail extends SessionSummary {
  segments: StoredSegment[];
  diarized: DiarizedSegment[];
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const getLanguages = () =>
  get<{ languages: Language[] }>("/api/languages").then((r) => r.languages);

export const getEngines = () =>
  get<{ engines: string[] }>("/api/engines").then((r) => r.engines);

export const getSettings = () => get<Settings>("/api/settings");

export async function saveSettings(settings: Settings): Promise<Settings> {
  const res = await fetch(`${API_BASE}/api/settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const getTranscripts = () =>
  get<{ sessions: SessionSummary[] }>("/api/transcripts").then((r) => r.sessions);

export const getTranscript = (id: string) =>
  get<SessionDetail>(`/api/transcripts/${id}`);

export async function deleteTranscript(id: string): Promise<void> {
  await fetch(`${API_BASE}/api/transcripts/${id}`, { method: "DELETE" });
}

// stage "transcript" runs diarization + translation only; "summary" also
// produces the LLM summary (reusing an existing transcript if there is one).
export type AnalyzeStage = "transcript" | "summary";

export async function analyzeTranscript(
  id: string,
  stage: AnalyzeStage = "summary",
  diarize = true,
): Promise<void> {
  const res = await fetch(
    `${API_BASE}/api/transcripts/${id}/analyze?stage=${stage}&diarize=${diarize}`,
    { method: "POST" },
  );
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `${res.status} ${res.statusText}`);
  }
}

// Upload an arbitrary audio/video file; it becomes a session and is analysed
// with the same diarize -> translate -> summarize pipeline as a recording.
export async function uploadMedia(
  file: File,
): Promise<{ session_id: string; status: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/transcripts/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}
