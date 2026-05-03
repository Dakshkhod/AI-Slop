"use client";

import { AnalyzeResponse } from "@/lib/api";

function verdictColor(v: AnalyzeResponse["verdict"]) {
  if (v === "likely_real") return "text-ok";
  if (v === "likely_ai") return "text-flag";
  return "text-warn";
}

function verdictBg(v: AnalyzeResponse["verdict"]) {
  if (v === "likely_real") return "from-emerald-500/20 via-ok/10 to-transparent";
  if (v === "likely_ai") return "from-rose-500/25 via-flag/15 to-transparent";
  return "from-amber-500/20 via-warn/15 to-transparent";
}

function ConfidenceArc({ score, uncertainty }: { score: number; uncertainty: number }) {
  const r = 90;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(100, score)) / 100;
  const dash = c * pct;
  return (
    <div className="relative h-56 w-56">
      <svg viewBox="0 0 220 220" className="h-full w-full -rotate-90">
        <defs>
          <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#7ee0ff" />
            <stop offset="100%" stopColor="#3ec5ff" />
          </linearGradient>
        </defs>
        <circle cx="110" cy="110" r={r} stroke="#191c39" strokeWidth="14" fill="none" />
        <circle
          cx="110"
          cy="110"
          r={r}
          stroke="url(#grad)"
          strokeWidth="14"
          strokeLinecap="round"
          fill="none"
          strokeDasharray={`${dash} ${c}`}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-5xl font-bold tabular-nums text-ink-50">{Math.round(score)}</span>
        <span className="text-[11px] uppercase tracking-[0.2em] text-ink-300">authenticity</span>
        <span className="mt-1 text-xs text-ink-300">± {uncertainty.toFixed(1)}</span>
      </div>
    </div>
  );
}

export function ScoreCard({ result }: { result: AnalyzeResponse }) {
  const pct = (result.p_ai * 100).toFixed(0);
  return (
    <section
      className={[
        "relative overflow-hidden rounded-3xl border border-ink-700 bg-ink-900/60 p-6 backdrop-blur",
        "shadow-[0_24px_80px_-30px_rgba(0,0,0,0.5)]",
      ].join(" ")}
    >
      <div
        className={`pointer-events-none absolute inset-0 bg-gradient-to-br ${verdictBg(
          result.verdict
        )}`}
      />
      <div className="relative flex flex-col items-start gap-6 md:flex-row md:items-center">
        <ConfidenceArc score={result.score} uncertainty={result.score_uncertainty} />
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span
              className={`text-xs font-semibold uppercase tracking-[0.2em] ${verdictColor(
                result.verdict
              )}`}
            >
              {result.verdict.replace("_", " ")}
            </span>
            <span className="rounded-full bg-ink-800/70 px-2 py-0.5 text-[10px] uppercase tracking-widest text-ink-300 ring-1 ring-ink-700">
              {result.modality}
            </span>
          </div>
          <h2 className="mt-1 text-2xl font-semibold text-ink-50">{result.verdict_label}</h2>
          <p className="mt-2 max-w-prose text-sm text-ink-300">
            We estimate <span className="font-semibold text-ink-50">{pct}%</span> probability the file is
            AI-generated, fused from {result.signals.length} independent signals across
            6 detection layers.
          </p>
          <dl className="mt-5 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
            <div className="rounded-xl bg-ink-800/60 px-3 py-2 ring-1 ring-ink-700">
              <dt className="text-[10px] uppercase tracking-widest text-ink-400">file</dt>
              <dd className="mt-0.5 truncate text-ink-100">{result.filename}</dd>
            </div>
            <div className="rounded-xl bg-ink-800/60 px-3 py-2 ring-1 ring-ink-700">
              <dt className="text-[10px] uppercase tracking-widest text-ink-400">size</dt>
              <dd className="mt-0.5 text-ink-100">
                {(result.bytes / 1024).toFixed(1)} KB
              </dd>
            </div>
            <div className="rounded-xl bg-ink-800/60 px-3 py-2 ring-1 ring-ink-700">
              <dt className="text-[10px] uppercase tracking-widest text-ink-400">latency</dt>
              <dd className="mt-0.5 text-ink-100">{result.processing_ms} ms</dd>
            </div>
            <div className="rounded-xl bg-ink-800/60 px-3 py-2 ring-1 ring-ink-700">
              <dt className="text-[10px] uppercase tracking-widest text-ink-400">request</dt>
              <dd className="mt-0.5 font-mono text-xs text-ink-100">{result.request_id}</dd>
            </div>
          </dl>
        </div>
      </div>
    </section>
  );
}
