"use client";

import { SignalResult } from "@/lib/api";
import { useState } from "react";

const SEV_STYLE: Record<string, string> = {
  flag: "border-flag/50 bg-flag/10 text-flag",
  warn: "border-warn/50 bg-warn/10 text-warn",
  pass: "border-ok/50 bg-ok/10 text-ok",
  info: "border-ink-700 bg-ink-800/60 text-ink-300",
};

const SEV_LABEL: Record<string, string> = {
  flag: "FLAG",
  warn: "WARN",
  pass: "PASS",
  info: "INFO",
};

function Bar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-ink-800">
      <div
        className="h-full rounded-full"
        style={{
          width: `${pct}%`,
          background: `linear-gradient(90deg, #5fd28a 0%, #ffb547 50%, #ff5577 100%)`,
        }}
      />
    </div>
  );
}

export function SignalChecklist({ signals }: { signals: SignalResult[] }) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <section className="p-3">
      <ul className="space-y-2">
        {signals.map((s) => {
          const sev = s.error ? "info" : s.severity;
          const isOpen = open === s.id;
          return (
            <li
              key={s.id}
              className={[
                "rounded-xl border bg-ink-900/40 p-3",
                SEV_STYLE[sev] || SEV_STYLE.info,
              ].join(" ")}
            >
              <button
                onClick={() => setOpen(isOpen ? null : s.id)}
                className="flex w-full items-center gap-3 text-left"
              >
                <span className="rounded-md border border-current px-1.5 py-0.5 font-mono text-[10px]">
                  {SEV_LABEL[sev] || "INFO"}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="truncate font-medium text-ink-50">{s.name}</span>
                    <span className="font-mono text-xs text-ink-300">
                      L{s.layer} · p={s.p_ai.toFixed(2)}
                    </span>
                  </div>
                  <div className="mt-1.5">
                    <Bar value={s.p_ai} />
                  </div>
                </div>
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  className={`shrink-0 transition ${isOpen ? "rotate-180" : ""}`}
                >
                  <polyline points="6 9 12 15 18 9" />
                </svg>
              </button>
              {isOpen && (
                <div className="mt-3 space-y-2 border-t border-current/20 pt-3 text-sm text-ink-200">
                  {s.plain_language && <p>{s.plain_language}</p>}
                  {s.description && (
                    <p className="text-ink-300">{s.description}</p>
                  )}
                  {s.error && (
                    <p className="font-mono text-xs text-flag/90">
                      Error: {s.error}
                    </p>
                  )}
                  {s.evidence && Object.keys(s.evidence).length > 0 && (
                    <pre className="max-h-48 overflow-auto rounded-lg bg-ink-950 p-3 font-mono text-[11px] text-ink-200 ring-1 ring-ink-700">
                      {JSON.stringify(s.evidence, null, 2)}
                    </pre>
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
