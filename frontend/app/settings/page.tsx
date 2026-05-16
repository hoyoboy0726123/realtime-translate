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
  local: "地端 — MLX-Whisper + NLLB（Apple Silicon 原生）",
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
        <div className="field">
          <label>中文字體</label>
          <select
            value={settings.chinese_variant}
            onChange={(e) => update({ chinese_variant: e.target.value })}
          >
            <option value="traditional">繁體中文</option>
            <option value="simplified">简体中文</option>
          </select>
        </div>
        <p className="sub" style={{ marginBottom: 0 }}>
          鎖定後，無論講者說語言 A 或 B，模型只會在這兩種語言之間互譯。
          中文字幕可選繁體或簡體（繁體以 OpenCC 轉換為台灣用語），雲端與地端引擎皆適用。
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
              <label>語音辨識模型（即時字幕）</label>
              <input
                type="text"
                value={settings.local.whisper_model}
                onChange={(e) =>
                  update({
                    local: { ...settings.local, whisper_model: e.target.value },
                  })
                }
              />
            </div>
            <div className="field">
              <label>語音辨識模型（錄音分析）</label>
              <input
                type="text"
                value={settings.local.analyze_whisper_model}
                onChange={(e) =>
                  update({
                    local: {
                      ...settings.local,
                      analyze_whisper_model: e.target.value,
                    },
                  })
                }
              />
            </div>
            <div className="field">
              <label>翻譯模型（NLLB）</label>
              <input
                type="text"
                value={settings.local.translate_model}
                onChange={(e) =>
                  update({
                    local: { ...settings.local, translate_model: e.target.value },
                  })
                }
              />
            </div>
            <div className="field">
              <label>摘要模型（錄音分析 LLM）</label>
              <input
                type="text"
                value={settings.local.summary_model}
                onChange={(e) =>
                  update({
                    local: { ...settings.local, summary_model: e.target.value },
                  })
                }
              />
            </div>
          </div>
          <p className="sub" style={{ marginBottom: 0 }}>
            地端引擎：MLX-Whisper 做語音辨識（Apple Silicon 原生加速），NLLB-200 做翻譯。
            即時字幕用較快的 turbo 模型；錄音分析離線執行，可用較大、較準的完整模型。
            首次使用會自動下載模型權重，請先安裝 <code>requirements-local.txt</code>。
          </p>
        </div>
      )}

      <button className="primary" onClick={onSave} disabled={saving}>
        {saving ? "儲存中…" : "儲存設定"}
      </button>
    </div>
  );
}
