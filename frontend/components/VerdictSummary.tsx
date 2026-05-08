"use client";

import { SignalResult } from "@/lib/api";

type Bucket = {
  label: string;
  emoji: string;
  desc: string;
  bg: string;
  border: string;
  chipBg: string;
  chipText: string;
  signals: SignalResult[];
};

function bucketSignals(signals: SignalResult[]): {
  problems: SignalResult[];
  passed: SignalResult[];
  ambiguous: SignalResult[];
} {
  const problems: SignalResult[] = [];
  const passed: SignalResult[] = [];
  const ambiguous: SignalResult[] = [];

  for (const s of signals) {
    if (s.error) continue; // skip errored — surfaced elsewhere
    if (s.severity === "flag") problems.push(s);
    else if (s.severity === "pass") passed.push(s);
    else if (s.severity === "warn") ambiguous.push(s);
    // 'info' → skip (neutral, low confidence)
  }

  // Sort within each bucket by impact (|p_ai - 0.5| × confidence)
  const impact = (s: SignalResult) => Math.abs(s.p_ai - 0.5) * s.confidence;
  problems.sort((a, b) => impact(b) - impact(a));
  passed.sort((a, b) => impact(b) - impact(a));
  ambiguous.sort((a, b) => impact(b) - impact(a));

  return { problems, passed, ambiguous };
}

function SignalRow({ s }: { s: SignalResult }) {
  return (
    <li className="rounded-xl border border-ink-700/40 bg-ink-950/40 p-3">
      <div className="flex items-start justify-between gap-2">
        <span className="text-sm font-medium leading-tight text-ink-100">
          {s.name}
        </span>
        <span className="shrink-0 rounded-md bg-ink-800/80 px-1.5 py-0.5 font-mono text-[10px] text-ink-400 ring-1 ring-ink-700">
          L{s.layer}
        </span>
      </div>
      {s.plain_language && (
        <p className="mt-1.5 text-xs leading-relaxed text-ink-300">
          {s.plain_language}
        </p>
      )}
    </li>
  );
}

function Column({ bucket }: { bucket: Bucket }) {
  const empty = bucket.signals.length === 0;
  return (
    <div
      className={`flex flex-col rounded-2xl border ${bucket.border} ${bucket.bg} p-4`}
    >
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-lg leading-none">{bucket.emoji}</span>
          <h4 className="text-sm font-semibold tracking-tight text-ink-50">
            {bucket.label}
          </h4>
        </div>
        <span
          className={`rounded-full px-2 py-0.5 text-[11px] font-semibold tabular-nums ring-1 ${bucket.chipBg} ${bucket.chipText}`}
        >
          {bucket.signals.length}
        </span>
      </div>
      <p className="mb-3 text-[11px] leading-relaxed text-ink-400">
        {bucket.desc}
      </p>
      {empty ? (
        <div className="flex flex-1 items-center justify-center rounded-xl border border-dashed border-ink-700/60 px-3 py-6 text-center text-xs text-ink-500">
          Nothing in this bucket.
        </div>
      ) : (
        <ul className="space-y-2">
          {bucket.signals.map((s) => (
            <SignalRow key={s.id} s={s} />
          ))}
        </ul>
      )}
    </div>
  );
}

export function VerdictSummary({ signals }: { signals: SignalResult[] }) {
  const { problems, passed, ambiguous } = bucketSignals(signals);

  const buckets: Bucket[] = [
    {
      label: "Problems found",
      emoji: "⚠",
      desc: "Signals pointing toward AI generation. The stronger the impact, the higher in the list.",
      bg: "bg-flag/5",
      border: "border-flag/30",
      chipBg: "bg-flag/15 ring-flag/40",
      chipText: "text-flag",
      signals: problems,
    },
    {
      label: "Ambiguous",
      emoji: "?",
      desc: "Signals that fired but with limited confidence — context-sensitive checks.",
      bg: "bg-warn/5",
      border: "border-warn/30",
      chipBg: "bg-warn/15 ring-warn/40",
      chipText: "text-warn",
      signals: ambiguous,
    },
    {
      label: "Passed",
      emoji: "✓",
      desc: "Signals consistent with a real, unmodified file.",
      bg: "bg-ok/5",
      border: "border-ok/30",
      chipBg: "bg-ok/15 ring-ok/40",
      chipText: "text-ok",
      signals: passed,
    },
  ];

  return (
    <section className="rounded-3xl border border-ink-700 bg-ink-900/60 p-5 backdrop-blur">
      <div className="mb-4 flex items-baseline justify-between gap-3">
        <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-ink-300">
          What we found
        </h3>
        <span className="text-xs text-ink-400">
          {problems.length} problem{problems.length === 1 ? "" : "s"} ·{" "}
          {ambiguous.length} ambiguous · {passed.length} passed
        </span>
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        {buckets.map((b) => (
          <Column key={b.label} bucket={b} />
        ))}
      </div>
    </section>
  );
}
