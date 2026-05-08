"use client";

import { AnalyzeResponse } from "@/lib/api";

function verdictColor(v: AnalyzeResponse["verdict"]) {
  if (v === "likely_real") return "text-ok";
  if (v === "likely_ai") return "text-flag";
  return "text-warn";
}

function verdictRingColor(v: AnalyzeResponse["verdict"]) {
  if (v === "likely_real") return "#5fae7e";
  if (v === "likely_ai") return "#e2545b";
  return "#d6a14a";
}

function ConfidenceArc({
  score,
  uncertainty,
  color,
}: {
  score: number;
  uncertainty: number;
  color: string;
}) {
  const r = 90;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(100, score)) / 100;
  const dash = c * pct;
  return (
    <div className="relative h-48 w-48 shrink-0">
      <svg viewBox="0 0 220 220" className="h-full w-full -rotate-90">
        <circle cx="110" cy="110" r={r} stroke="#1a1a1a" strokeWidth="12" fill="none" />
        <circle
          cx="110"
          cy="110"
          r={r}
          stroke={color}
          strokeWidth="12"
          strokeLinecap="round"
          fill="none"
          strokeDasharray={`${dash} ${c}`}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-5xl font-semibold tabular-nums text-ink-50">{Math.round(score)}</span>
        <span className="mt-0.5 text-[10px] uppercase tracking-[0.2em] text-ink-400">authenticity</span>
        <div className="mt-1 flex items-center gap-1 text-[11px] text-ink-400">
          <span>± {uncertainty.toFixed(1)}</span>
          <span
            className="inline-grid h-3.5 w-3.5 place-content-center rounded-full border border-ink-600 text-[9px]"
            title="Scores can change between uploads when metadata is stripped or files are recompressed."
          >
            i
          </span>
        </div>
      </div>
    </div>
  );
}

export function ScoreCard({ result }: { result: AnalyzeResponse }) {
  const pct = (result.p_ai * 100).toFixed(0);
  const ringColor = verdictRingColor(result.verdict);
  return (
    <section className="rounded-2xl border border-ink-800 bg-ink-900 p-6">
      <div className="flex flex-col items-start gap-6 md:flex-row md:items-center">
        <ConfidenceArc
          score={result.score}
          uncertainty={result.score_uncertainty}
          color={ringColor}
        />
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span
              className={`text-[11px] font-semibold uppercase tracking-[0.2em] ${verdictColor(
                result.verdict
              )}`}
            >
              {result.verdict.replace("_", " ")}
            </span>
            <span className="rounded border border-ink-700 px-1.5 py-0.5 text-[9px] uppercase tracking-[0.18em] text-ink-400">
              {result.modality}
            </span>
          </div>
          <h2 className="mt-2 text-2xl font-semibold tracking-tight text-ink-50">
            {result.verdict_label}
          </h2>
          <p className="mt-2 max-w-prose text-sm leading-relaxed text-ink-300">
            {pct}% probability AI-generated, fused from {result.signals.length} signals across 6 detection layers.
          </p>
          <dl className="mt-5 grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
            <Stat k="file" v={result.filename} mono />
            <Stat k="size" v={`${(result.bytes / 1024).toFixed(1)} KB`} />
            <Stat k="latency" v={`${result.processing_ms} ms`} />
            <Stat k="request" v={result.request_id} mono />
          </dl>
        </div>
      </div>
    </section>
  );
}

function Stat({ k, v, mono }: { k: string; v: string; mono?: boolean }) {
  return (
    <div className="rounded-lg border border-ink-800 bg-black/40 px-3 py-2">
      <dt className="text-[9px] uppercase tracking-[0.2em] text-ink-500">{k}</dt>
      <dd
        className={`mt-0.5 truncate text-ink-200 ${
          mono ? "font-mono text-xs" : "text-sm"
        }`}
      >
        {v}
      </dd>
    </div>
  );
}
