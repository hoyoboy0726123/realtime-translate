"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import RollingSubtitles from "@/components/RollingSubtitles";
import { getLanguages, getSettings, type Language } from "@/lib/api";
import { useTranslator } from "@/lib/useTranslator";

const STATUS_TEXT: Record<string, string> = {
  idle: "未連線",
  connecting: "連線中…",
  live: "即時翻譯中",
  error: "錯誤",
};

type Layout = "vertical" | "horizontal";

export default function Home() {
  const { status, segments, info, error, start, stop } = useTranslator();
  const [langs, setLangs] = useState<Language[]>([]);
  const [pair, setPair] = useState({ a: "zh", b: "en", engine: "mock" });
  const [sessionName, setSessionName] = useState("");
  const [layout, setLayout] = useState<Layout>("vertical");

  useEffect(() => {
    getLanguages().then(setLangs).catch(() => {});
    getSettings()
      .then((s) => setPair({ a: s.lang_a, b: s.lang_b, engine: s.engine }))
      .catch(() => {});
    setSessionName(`會議 ${new Date().toLocaleString()}`);
    const saved = localStorage.getItem("subtitleLayout");
    if (saved === "horizontal" || saved === "vertical") setLayout(saved);
  }, []);

  const toggleLayout = () => {
    setLayout((cur) => {
      const next: Layout = cur === "vertical" ? "horizontal" : "vertical";
      localStorage.setItem("subtitleLayout", next);
      return next;
    });
  };

  const langA = info.langA ?? pair.a;
  const langB = info.langB ?? pair.b;
  const engine = info.engine ?? pair.engine;

  const label = (code: string) => {
    const l = langs.find((x) => x.code === code);
    return l ? `${l.native} · ${l.name}` : code;
  };

  const busy = status === "live" || status === "connecting";

  return (
    <main className={`stage stage-${layout}`}>
      <div className="panes">
        <RollingSubtitles segments={segments} field="a" label={label(langA)} />
        <RollingSubtitles segments={segments} field="b" label={label(langB)} />
      </div>

      <div className="controlbar">
        <span className="brand">Realtime Translate</span>
        <span className={`status-dot ${status}`} />
        <span className="badge">{STATUS_TEXT[status]}</span>
        <span className="badge">引擎：{engine}</span>
        <span className="badge">
          {label(langA)} ⇄ {label(langB)}
        </span>

        <span className="spacer" />

        <button className="navlink" onClick={toggleLayout}>
          {layout === "vertical" ? "⇆ 改為左右顯示" : "⇅ 改為上下顯示"}
        </button>

        <input
          type="text"
          value={sessionName}
          onChange={(e) => setSessionName(e.target.value)}
          placeholder="會議名稱"
          disabled={busy}
          style={{ minWidth: 200 }}
        />
        {busy ? (
          <button className="stop" onClick={stop}>
            停止
          </button>
        ) : (
          <button
            className="primary"
            onClick={() => start(sessionName || "Untitled session")}
          >
            開始翻譯
          </button>
        )}
        <Link className="navlink" href="/settings">
          後台設定
        </Link>
        <Link className="navlink" href="/history">
          翻譯記錄
        </Link>
      </div>

      {error && <div className="errorbar">{error}</div>}
    </main>
  );
}
