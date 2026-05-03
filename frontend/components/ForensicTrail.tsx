"use client";

export function ForensicTrail({ items }: { items: string[] }) {
  if (!items.length) {
    return (
      <section className="rounded-3xl border border-ink-700 bg-ink-900/60 p-5 backdrop-blur">
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
    <section className="rounded-3xl border border-ink-700 bg-ink-900/60 p-5 backdrop-blur">
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-[0.2em] text-ink-300">
        Forensic trail
      </h3>
      <ol className="space-y-2 text-sm text-ink-100">
        {items.map((line, i) => (
          <li key={i} className="flex gap-3">
            <span className="font-mono text-xs text-accent-400">
              {String(i + 1).padStart(2, "0")}
            </span>
            <span className="flex-1">{line}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}
