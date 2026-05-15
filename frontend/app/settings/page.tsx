"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  getEngines,
  getLanguages,
  getSettings,
  saveSettings,
  type Language,
  type Settings,
} from "@/lib/api";

const ENGINE_LABEL: Record<string, string> = {
  cloud: "雲端 — OpenAI Realtime (gpt-realtime-translate)",
  local: "地端 — Meta SeamlessStreaming（逐字即時翻譯）",
  mock: "示範 — 免金鑰／免模型，重播範例對話",
};

export default function SettingsPage() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [langs, setLangs] = useState<Language[]>([]);
  const [engines, setEngines] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<{ kind: "ok" | "err"; text: string } | null>(null);

  useEffect(() => {
    Promise.all([getSettings(), getLanguages(), getEngines()])
      .then(([s, l, e]) => {
        setSettings(s);
        setLangs(l);
        setEngines(e);
      })
      .catch((e) => setMsg({ kind: "err", text: `載入失敗：${e.message}` }));
  }, []);

  if (!settings) {
    return (
      <div className="page">
        <p className="sub">{msg ? msg.text : "載入中…"}</p>
      </div>
    );
  }

  const update = (patch: Partial<Settings>) =>
    setSettings({ ...settings, ...patch });

  const onSave = async () => {
    setSaving(true);
    setMsg(null);
    try {
      const saved = await saveSettings(settings);
      setSettings(saved);
      setMsg({ kind: "ok", text: "設定已儲存，下次開始翻譯即會套用。" });
    } catch (e) {
      setMsg({ kind: "err", text: (e as Error).message });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="page">
      <h1>後台設定</h1>
      <p className="sub">
        選擇翻譯引擎並鎖定語言對。
        <Link className="navlink" href="/" style={{ marginLeft: 12 }}>
          ← 回到即時翻譯
        </Link>
      </p>

      {msg && <div className={`notice ${msg.kind}`}>{msg.text}</div>}

      <div className="card">
        <div className="field">
          <label>翻譯引擎</label>
          <select
            value={settings.engine}
            onChange={(e) => update({ engine: e.target.value })}
          >
            {engines.map((e) => (
              <option key={e} value={e}>
                {ENGINE_LABEL[e] ?? e}
              </option>
            ))}
          </select>
        </div>

        <div className="field-row">
          <div className="field">
            <label>語言 A（畫面上方字幕）</label>
            <select
              value={settings.lang_a}
              onChange={(e) => update({ lang_a: e.target.value })}
            >
              {langs.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.native} · {l.name}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>語言 B（畫面下方字幕）</label>
            <select
              value={settings.lang_b}
              onChange={(e) => update({ lang_b: e.target.value })}
            >
              {langs.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.native} · {l.name}
                </option>
              ))}
            </select>
          </div>
        </div>
        <p className="sub" style={{ marginBottom: 0 }}>
          鎖定後，無論講者說語言 A 或 B，模型只會在這兩種語言之間互譯。
        </p>
      </div>

      {settings.engine === "cloud" && (
        <div className="card">
          <div className="field">
            <label>OpenAI Realtime 模型名稱</label>
            <input
              type="text"
              value={settings.cloud.model}
              onChange={(e) =>
                update({ cloud: { ...settings.cloud, model: e.target.value } })
              }
            />
          </div>
          <p className="sub" style={{ marginBottom: 0 }}>
            需在後端 <code>.env</code> 設定 <code>OPENAI_API_KEY</code>。
          </p>
        </div>
      )}

      {settings.engine === "local" && (
        <div className="card">
          <div className="field-row">
            <div className="field">
              <label>運算裝置</label>
              <select
                value={settings.local.device}
                onChange={(e) =>
                  update({ local: { ...settings.local, device: e.target.value } })
                }
              >
                <option value="mps">mps（Apple Silicon）</option>
                <option value="cpu">cpu</option>
                <option value="cuda">cuda</option>
              </select>
            </div>
            <div className="field">
              <label>音訊區塊大小（毫秒）</label>
              <input
                type="number"
                min={160}
                max={1000}
                step={80}
                value={settings.local.source_segment_size_ms}
                onChange={(e) =>
                  update({
                    local: {
                      ...settings.local,
                      source_segment_size_ms: Number(e.target.value),
                    },
                  })
                }
              />
            </div>
            <div className="field">
              <label>輸出門檻 decision_threshold</label>
              <input
                type="number"
                min={0.1}
                max={0.9}
                step={0.05}
                value={settings.local.decision_threshold}
                onChange={(e) =>
                  update({
                    local: {
                      ...settings.local,
                      decision_threshold: Number(e.target.value),
                    },
                  })
                }
              />
            </div>
          </div>
          <p className="sub" style={{ marginBottom: 0 }}>
            使用 Meta SeamlessStreaming 逐字同步翻譯。區塊越小、門檻越低，字詞出現越快（延遲越低）；
            數值越大則翻譯越準。首次使用會自動下載模型權重，請先安裝{" "}
            <code>requirements-local.txt</code> 並 git 安裝 <code>seamless_communication</code>。
          </p>
        </div>
      )}

      <button className="primary" onClick={onSave} disabled={saving}>
        {saving ? "儲存中…" : "儲存設定"}
      </button>
    </div>
  );
}
