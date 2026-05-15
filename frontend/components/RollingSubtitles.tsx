"use client";

import type { LiveSegment } from "@/lib/useTranslator";

interface Props {
  segments: LiveSegment[];
  field: "a" | "b";
  label: string;
  position: "top" | "bottom";
  /** How many lines stay visible. Kept small so the text can be large. */
  visible?: number;
}

/**
 * Vertical rolling subtitles: newest line enters at the bottom and pushes older
 * lines up and out of view. One extra line past `visible` is kept mounted so it
 * can fade out through the CSS mask instead of vanishing abruptly.
 */
export default function RollingSubtitles({
  segments,
  field,
  label,
  position,
  visible = 3,
}: Props) {
  const lines = segments.slice(-(visible + 1));

  return (
    <section className={`pane pane-${position}`}>
      <div className="pane-label">{label}</div>
      <div className="roller">
        {lines.length === 0 ? (
          <p className="line line-empty">等待語音輸入… / Waiting for speech…</p>
        ) : (
          lines.map((seg) => {
            const text = field === "a" ? seg.textA : seg.textB;
            return (
              <p
                key={seg.id}
                className={`line ${seg.kind === "partial" ? "line-partial" : ""} ${
                  text ? "" : "line-empty"
                }`}
              >
                {text || "…"}
              </p>
            );
          })
        )}
      </div>
    </section>
  );
}
