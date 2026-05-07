"use client";

import { DomainBreakdown } from "@/components/DomainBreakdown";
import { ForensicTrail } from "@/components/ForensicTrail";
import { HeatmapView } from "@/components/HeatmapView";
import { ProvenanceCard } from "@/components/ProvenanceCard";
import { ScoreCard } from "@/components/ScoreCard";
import { SignalChecklist } from "@/components/SignalChecklist";
import { UploadZone } from "@/components/UploadZone";
import { AnalyzeResponse, analyzeFile, analyzeUrl } from "@/lib/api";
import { useState } from "react";

export default function HomePage() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  async function handleFile(file: File) {
    setBusy(true);
    setError(null);
    setResult(null);
    setPreviewUrl(null);
    try {
      if (file.type.startsWith("image/")) {
        setPreviewUrl(URL.createObjectURL(file));
      }
      const res = await analyzeFile(file);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function handleUrl(url: string) {
    setBusy(true);
    setError(null);
    setResult(null);
    setPreviewUrl(url);
    try {
      const res = await analyzeUrl(url);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="min-h-screen">
      <Header />

      <div className="mx-auto max-w-6xl px-6 pb-24">
        <Hero />

        <div className="mt-10">
          <UploadZone onFile={handleFile} onUrl={handleUrl} busy={busy} />
        </div>

        {busy && <BusyBlock />}
        {error && (
          <div className="mt-8 rounded-2xl border border-flag/40 bg-flag/10 p-4 text-flag">
            {error}
          </div>
        )}

        {result && (
          <div className="mt-10 space-y-6">
            <ScoreCard result={result} />

            <div className="grid gap-6 lg:grid-cols-5">
              <div className="space-y-6 lg:col-span-3">
                {result.heatmaps?.length ? (
                  <HeatmapView heatmaps={result.heatmaps} />
                ) : previewUrl && result.modality === "image" ? (
                  <PreviewBlock url={previewUrl} />
                ) : null}
                <ForensicTrail
                  items={result.provenance_trail}
                  legacyItems={result.forensic_trail}
                />
                <SignalChecklist signals={result.signals} />
              </div>
              <div className="space-y-6 lg:col-span-2">
                <ProvenanceCard trail={result.provenance_trail ?? []} />
                <DomainBreakdown data={result.domain_real_confidence} />
                <ModelBlock versions={result.model_versions} />
                <PrivacyBlock />
              </div>
            </div>
          </div>
        )}

        {!result && !busy && <FeatureGrid />}
      </div>
    </main>
  );
}

function Header() {
  return (
    <header className="border-b border-ink-700/60 bg-ink-950/40 backdrop-blur sticky top-0 z-10">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-3">
        <div className="flex items-center gap-2">
          <div className="grid h-8 w-8 place-content-center rounded-xl bg-accent-500/20 ring-1 ring-accent-500/40">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#7ee0ff" strokeWidth="2">
              <circle cx="12" cy="12" r="3" />
              <path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7-10-7-10-7z" />
            </svg>
          </div>
          <span className="text-lg font-semibold tracking-tight">TruthLens</span>
          <span className="ml-1 rounded-full bg-ink-800 px-2 py-0.5 text-[10px] uppercase tracking-widest text-ink-300 ring-1 ring-ink-700">
            beta
          </span>
        </div>
        <nav className="hidden gap-6 text-sm text-ink-300 sm:flex">
          <a href="#how" className="hover:text-ink-50">How it works</a>
          <a href="#layers" className="hover:text-ink-50">Detection layers</a>
          <a href="/docs" className="hover:text-ink-50">API</a>
        </nav>
      </div>
    </header>
  );
}

function Hero() {
  return (
    <section className="bg-grid relative -mx-6 mt-6 overflow-hidden rounded-3xl border border-ink-700/60 px-6 py-16 text-center">
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-transparent via-ink-950/40 to-ink-950" />
      <div className="relative">
        <span className="inline-flex items-center gap-2 rounded-full border border-accent-500/40 bg-accent-500/10 px-3 py-1 text-xs font-medium uppercase tracking-widest text-accent-400">
          Multi-modal · Explainable · Physics-based
        </span>
        <h1 className="mx-auto mt-4 max-w-3xl text-balance text-4xl font-bold tracking-tight text-ink-50 md:text-5xl">
          Detect AI slop across <span className="text-accent-400">images</span>,{" "}
          <span className="text-accent-400">video</span>, and{" "}
          <span className="text-accent-400">audio</span>.
        </h1>
        <p className="mx-auto mt-4 max-w-2xl text-pretty text-base text-ink-300">
          A 6-layer pipeline grounded in physics, not pixel-guessing. Every flag is
          explained — you see <em>why</em>, not just <em>what</em>.
        </p>
      </div>
    </section>
  );
}

function BusyBlock() {
  return (
    <div className="mt-8 overflow-hidden rounded-3xl border border-ink-700 bg-ink-900/60 p-6 text-center">
      <div className="mx-auto h-1.5 w-48 overflow-hidden rounded-full bg-ink-800">
        <div className="h-full w-1/3 animate-pulse rounded-full bg-accent-500" />
      </div>
      <p className="mt-3 text-sm text-ink-300">Running 6-layer pipeline…</p>
    </div>
  );
}

function PreviewBlock({ url }: { url: string }) {
  return (
    <section className="overflow-hidden rounded-3xl border border-ink-700 bg-ink-900/60 p-5 backdrop-blur">
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-[0.2em] text-ink-300">
        Source
      </h3>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={url} alt="upload" className="block max-h-96 w-full rounded-2xl object-contain" />
    </section>
  );
}

function ModelBlock({ versions }: { versions: Record<string, string> }) {
  return (
    <section className="rounded-3xl border border-ink-700 bg-ink-900/60 p-5 backdrop-blur">
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-[0.2em] text-ink-300">
        Models in use
      </h3>
      <ul className="space-y-1.5 font-mono text-xs text-ink-200">
        {Object.entries(versions).map(([k, v]) => (
          <li key={k} className="flex justify-between gap-3 truncate">
            <span className="text-ink-400">{k}</span>
            <span className="truncate">{v}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

function PrivacyBlock() {
  return (
    <section className="rounded-3xl border border-ink-700 bg-ink-900/60 p-5 backdrop-blur">
      <h3 className="mb-2 text-sm font-semibold uppercase tracking-[0.2em] text-ink-300">
        Privacy
      </h3>
      <p className="text-sm text-ink-300">
        Files are analysed in-memory and dropped after the response is built. We
        keep only a short perceptual hash to power local reverse-search.
      </p>
    </section>
  );
}

function FeatureGrid() {
  const items = [
    {
      h: "Provenance forensics",
      p: "EXIF, C2PA, software signatures, screenshot fingerprints — checks every byte of context.",
      d: "Layer 2",
    },
    {
      h: "Physics signals",
      p: "FFT 1/f, Benford's law on DCT, wavelet kurtosis, PRNU, double-JPEG — five tests grounded in real physics.",
      d: "Layer 3",
    },
    {
      h: "ML ensemble",
      p: "Pretrained transformer detector + model-attribution heatmaps. Optional, lazy-loaded, swappable.",
      d: "Layer 4",
    },
    {
      h: "Biological coherence",
      p: "Heartbeat (rPPG), microsaccades, vocal-tract physics, lighting consistency.",
      d: "Layer 5",
    },
    {
      h: "Bayesian fusion",
      p: "Confidence-weighted log-odds pool. Disagreement = honest 'inconclusive'.",
      d: "Layer 6",
    },
    {
      h: "Reverse search",
      p: "Local pHash cache + optional Google CSE. If we have seen it before, we say so.",
      d: "Layer 1",
    },
  ];
  return (
    <section id="layers" className="mt-16">
      <h2 className="text-2xl font-semibold tracking-tight">What we check</h2>
      <p className="mt-1 max-w-2xl text-ink-300">
        Six layers, sequenced cheap-to-expensive. Drop a file above to see them in
        action.
      </p>
      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {items.map((it) => (
          <article
            key={it.h}
            className="rounded-2xl border border-ink-700 bg-ink-900/60 p-5 transition hover:border-accent-600 hover:shadow-glow"
          >
            <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-accent-400">
              {it.d}
            </div>
            <h3 className="mt-2 text-lg font-semibold text-ink-50">{it.h}</h3>
            <p className="mt-1 text-sm text-ink-300">{it.p}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
