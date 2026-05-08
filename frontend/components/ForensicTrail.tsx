"use client";

import { useState } from "react";
import { TrailItem } from "@/lib/api";

const SEV_META: Record<string, { label: string; bg: string; text: string; ring: string }> = {
  flag:  { label: "FLAG",  bg: "bg-flag/15",         text: "text-flag",         ring: "ring-flag/40" },
  warn:  { label: "WARN",  bg: "bg-amber-500/15",    text: "text-amber-400",    ring: "ring-amber-500/40" },
  pass:  { label: "PASS",  bg: "bg-emerald-500/15",  text: "text-emerald-400",  ring: "ring-emerald-500/40" },
  info:  { label: "INFO",  bg: "bg-ink-700/40",      text: "text-ink-300",      ring: "ring-ink-600/40" },
};

function PAiBar({ p_ai, severity }: { p_ai: number; severity: string }) {
  const pct = Math.round(p_ai * 100);
  const color =
    severity === "pass" ? "bg-emerald-500" :
    severity === "flag" ? "bg-flag" :
    severity === "warn" ? "bg-amber-500" :
    "bg-ink-500";
  return (
    <div className="mt-2 flex items-center gap-2">
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-ink-800">
        <div className={`h-full rounded-full transition-all ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-10 text-right font-mono text-[10px] text-ink-400">{pct}% AI</span>
    </div>
  );
}

function TrailRow({ item, index }: { item: TrailItem; index: number }) {
  const [open, setOpen] = useState(false);
  const meta = SEV_META[item.severity] ?? SEV_META.info;
  const hasEvidence = Object.keys(item.evidence ?? {}).length > 0;

  return (
    <li className="rounded-2xl border border-ink-700/60 bg-ink-900/40 p-4 transition hover:border-ink-600">
      <button
        className="w-full text-left"
        onClick={() => hasEvidence && setOpen((v) => !v)}
        aria-expanded={open}
      >
        <div className="flex items-start gap-3">
          <span className="mt-0.5 w-5 shrink-0 font-mono text-xs tabular-nums text-ink-500">
            {String(index + 1).padStart(2, "0")}
          </span>

          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-medium leading-snug text-ink-100">{item.name}</span>
              <span
                className={`rounded-md px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider ring-1 ${meta.bg} ${meta.text} ${meta.ring}`}
              >
                {meta.label}
              </span>
              <span className="rounded-md bg-ink-800 px-1.5 py-0.5 text-[10px] text-ink-400 ring-1 ring-ink-700">
                Layer {item.layer}
              </span>
            </div>

            <p className="mt-1 text-sm leading-relaxed text-ink-300">{item.plain_language}</p>

            <PAiBar p_ai={item.p_ai} severity={item.severity} />
          </div>

          {hasEvidence && (
            <svg
              className={`mt-1 h-4 w-4 shrink-0 text-ink-500 transition-transform ${open ? "rotate-180" : ""}`}
              viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
            >
              <path d="M6 9l6 6 6-6" />
            </svg>
          )}
        </div>
      </button>

      {open && hasEvidence && (
        <div className="mt-3 rounded-xl border border-ink-700/40 bg-ink-950/60 p-3">
          <h4 className="mb-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-ink-400">
            Evidence
          </h4>
          <dl className="space-y-1">
            {Object.entries(item.evidence).map(([k, v]) => {
              if (v === null || v === undefined) return null;
              const display =
                typeof v === "object" ? JSON.stringify(v, null, 2) : String(v);
              return (
                <div key={k} className="flex gap-2 font-mono text-xs">
                  <dt className="shrink-0 text-ink-500">{k}:</dt>
                  <dd className="break-all whitespace-pre-wrap text-ink-300">{display}</dd>
                </div>
              );
            })}
          </dl>
        </div>
      )}
    </li>
  );
}

export function ForensicTrail({
  items,
  legacyItems,
}: {
  items?: TrailItem[];
  legacyItems?: string[];
}) {
  const hasStructured = items && items.length > 0;
  const hasLegacy = legacyItems && legacyItems.length > 0;

  if (!hasStructured && !hasLegacy) {
    return (
      <section className="rounded-2xl border border-ink-800 bg-ink-900 p-5">
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-[0.2em] text-ink-300">
          Forensic trail
        </h3>
        <p className="text-sm text-ink-300">
          No high-impact signals fired. The file looks clean across our checks.
        </p>
      </section>
    );
  }

  return (
    <section className="rounded-2xl border border-ink-800 bg-ink-900 p-5">
      <h3 className="mb-4 text-sm font-semibold uppercase tracking-[0.2em] text-ink-300">
        Forensic trail
      </h3>

      {hasStructured ? (
        <ol className="space-y-3">
          {items.map((item, i) => (
            <TrailRow key={item.id} item={item} index={i} />
          ))}
        </ol>
      ) : (
        <ol className="space-y-2 text-sm text-ink-100">
          {legacyItems!.map((line, i) => (
            <li key={i} className="flex gap-3">
              <span className="font-mono text-xs text-ink-400">
                {String(i + 1).padStart(2, "0")}
              </span>
              <span className="flex-1">{line}</span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
