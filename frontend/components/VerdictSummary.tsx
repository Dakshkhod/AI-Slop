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
    <li className="rounded-lg border border-ink-800 bg-black/30 p-2.5">
      <div className="flex items-start justify-between gap-2">
        <span className="text-[13px] font-medium leading-tight text-ink-100">
          {s.name}
        </span>
        <span className="shrink-0 font-mono text-[10px] text-ink-500">
          L{s.layer}
        </span>
      </div>
      {s.plain_language && (
        <p className="mt-1 text-[11px] leading-relaxed text-ink-400">
          {s.plain_language}
        </p>
      )}
    </li>
  );
}

function Column({ bucket }: { bucket: Bucket }) {
  const empty = bucket.signals.length === 0;
  return (
    <div className={`flex flex-col rounded-lg border bg-black/20 p-4 ${bucket.border}`}>
      <div className="mb-2 flex items-center justify-between">
        <h4 className="text-[13px] font-semibold tracking-tight text-ink-100">
          {bucket.label}
        </h4>
        <span
          className={`rounded px-1.5 py-0.5 text-[10px] font-semibold tabular-nums ring-1 ${bucket.chipBg} ${bucket.chipText}`}
        >
          {bucket.signals.length}
        </span>
      </div>
      <p className="mb-3 text-[11px] leading-relaxed text-ink-500">
        {bucket.desc}
      </p>
      {empty ? (
        <div className="flex flex-1 items-center justify-center px-3 py-6 text-center text-[11px] text-ink-600">
          —
        </div>
      ) : (
        <ul className="space-y-1.5">
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
      label: "Problems",
      emoji: "",
      desc: "Signals pointing toward AI generation, sorted by impact.",
      bg: "bg-ink-900",
      border: "border-l-2 border-l-flag border-ink-800",
      chipBg: "bg-flag/10 ring-flag/30",
      chipText: "text-flag",
      signals: problems,
    },
    {
      label: "Ambiguous",
      emoji: "",
      desc: "Signals fired with limited confidence — context-sensitive.",
      bg: "bg-ink-900",
      border: "border-l-2 border-l-warn border-ink-800",
      chipBg: "bg-warn/10 ring-warn/30",
      chipText: "text-warn",
      signals: ambiguous,
    },
    {
      label: "Passed",
      emoji: "",
      desc: "Signals consistent with a real, unmodified file.",
      bg: "bg-ink-900",
      border: "border-l-2 border-l-ok border-ink-800",
      chipBg: "bg-ok/10 ring-ok/30",
      chipText: "text-ok",
      signals: passed,
    },
  ];

  return (
    <section className="rounded-2xl border border-ink-800 bg-ink-900 p-5">
      <div className="mb-4 flex items-baseline justify-between gap-3">
        <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-ink-300">
          What we found
        </h3>
        <span className="text-xs text-ink-500">
          {problems.length} · {ambiguous.length} · {passed.length}
        </span>
      </div>
      <div className="grid gap-3 md:grid-cols-3">
        {buckets.map((b) => (
          <Column key={b.label} bucket={b} />
        ))}
      </div>
    </section>
  );
}
