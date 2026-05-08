"use client";

import { DomainBreakdown as DBT } from "@/lib/api";

const LABELS: Record<keyof DBT, { label: string; hint: string; color: string }> = {
  provenance: {
    label: "Provenance",
    hint: "EXIF, C2PA, screenshot fingerprints",
    color: "from-sky-500 to-cyan-400",
  },
  quantum: {
    label: "Quantum / sensor",
    hint: "PRNU residual, periodic noise, RGB-noise correlation",
    color: "from-violet-500 to-fuchsia-400",
  },
  thermodynamic: {
    label: "Thermodynamic",
    hint: "FFT slope, anisotropy, HF energy, Benford, wavelets",
    color: "from-emerald-500 to-teal-400",
  },
  biological: {
    label: "Biological",
    hint: "Facial asymmetry, eye reflections, rPPG, vocal tract",
    color: "from-rose-500 to-pink-400",
  },
  semantic: {
    label: "Semantic",
    hint: "Lighting, colour, edges",
    color: "from-amber-500 to-yellow-400",
  },
  ml: {
    label: "ML classifier",
    hint: "Pretrained transformer ensemble",
    color: "from-indigo-500 to-blue-400",
  },
};

export function DomainBreakdown({ data }: { data: DBT }) {
  const entries = Object.entries(data) as [keyof DBT, number][];
  return (
    <section className="rounded-2xl border border-ink-800 bg-ink-900 p-5">
      <h3 className="mb-4 text-sm font-semibold uppercase tracking-[0.2em] text-ink-300">
        Per-domain real confidence
      </h3>
      <div className="space-y-4">
        {entries.map(([k, v]) => {
          const meta = LABELS[k];
          const pct = Math.round(v * 100);
          return (
            <div key={k}>
              <div className="mb-1 flex items-baseline justify-between gap-3">
                <div className="text-sm font-medium text-ink-100">{meta.label}</div>
                <div className="font-mono text-xs text-ink-300">
                  {pct}% real
                </div>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-ink-800 ring-1 ring-ink-700">
                <div
                  className={`h-full rounded-full bg-gradient-to-r ${meta.color}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <div className="mt-1 text-[11px] text-ink-400">{meta.hint}</div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
