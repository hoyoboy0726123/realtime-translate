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
  cloud: { model: string };
  local: {
    device: string;
    source_segment_size_ms: number;
    decision_threshold: number;
    unity_model: string;
    monotonic_decoder_model: string;
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

export interface SessionDetail extends SessionSummary {
  segments: StoredSegment[];
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
