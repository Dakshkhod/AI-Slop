"use client";

import { Heatmap } from "@/lib/api";
import { useState } from "react";

export function HeatmapView({ heatmaps }: { heatmaps: Heatmap[] }) {
  const [idx, setIdx] = useState(0);
  if (!heatmaps.length) return null;
  const h = heatmaps[idx];
  return (
    <section className="overflow-hidden rounded-3xl border border-ink-700 bg-ink-900/60 p-5 backdrop-blur">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-ink-300">
          Visual evidence
        </h3>
        {heatmaps.length > 1 && (
          <div className="flex gap-1">
            {heatmaps.map((_, i) => (
              <button
                key={i}
                onClick={() => setIdx(i)}
                className={`h-7 rounded-md px-3 text-xs font-medium ${
                  i === idx
                    ? "bg-accent-500 text-ink-950"
                    : "bg-ink-800 text-ink-200 hover:bg-ink-700"
                }`}
              >
                {heatmaps[i].kind}
              </button>
            ))}
          </div>
        )}
      </div>
      <div className="overflow-hidden rounded-2xl bg-black/40 ring-1 ring-ink-700">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`data:${h.mime};base64,${h.data_base64}`}
          alt={h.description}
          className="block h-auto w-full"
        />
      </div>
      <p className="mt-3 text-sm text-ink-300">{h.description}</p>
    </section>
  );
}
