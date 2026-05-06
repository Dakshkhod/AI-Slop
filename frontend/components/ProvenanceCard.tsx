"use client";

import { TrailItem } from "@/lib/api";

type C2paEvidence = {
  present?: boolean;
  verified?: boolean;
  issuer?: string | null;
  claim_generator?: string | null;
  assertions?: string[];
  marker?: string | null;
  error?: string | null;
};

function C2paBadge({ verified, present }: { verified?: boolean; present?: boolean }) {
  if (verified) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-semibold text-emerald-400 ring-1 ring-emerald-500/40">
        <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <path d="M20 6L9 17l-5-5" />
        </svg>
        Cryptographically verified
      </span>
    );
  }
  if (present) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-500/15 px-3 py-1 text-xs font-semibold text-amber-400 ring-1 ring-amber-500/40">
        <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10" />
          <path d="M12 8v4M12 16h.01" />
        </svg>
        Manifest present (not verified)
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-ink-700/40 px-3 py-1 text-xs text-ink-400 ring-1 ring-ink-600/40">
      <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10" />
        <path d="M15 9l-6 6M9 9l6 6" />
      </svg>
      No manifest
    </span>
  );
}

export function ProvenanceCard({ trail }: { trail: TrailItem[] }) {
  const c2paSignal = trail.find((t) => t.id === "c2pa_presence");
  const exifSignal = trail.find((t) => t.id === "exif_coherence");

  if (!c2paSignal && !exifSignal) return null;

  const c2pa = (c2paSignal?.evidence ?? {}) as C2paEvidence;
  const exifFields = (exifSignal?.evidence as { fields?: Record<string, string> })?.fields ?? {};

  const hasExifCameraFields = Object.keys(exifFields).some((k) =>
    ["make", "camera_model", "datetime", "iso", "aperture", "shutter"].includes(k)
  );

  return (
    <section className="rounded-3xl border border-ink-700 bg-ink-900/60 p-5 backdrop-blur">
      <h3 className="mb-4 text-sm font-semibold uppercase tracking-[0.2em] text-ink-300">
        Provenance
      </h3>

      {/* C2PA block */}
      {c2paSignal && (
        <div className="mb-4">
          <div className="mb-2 flex items-center justify-between gap-2 flex-wrap">
            <span className="text-sm font-medium text-ink-100">C2PA / Content Credentials</span>
            <C2paBadge verified={c2pa.verified} present={c2pa.present} />
          </div>

          {c2pa.verified && (
            <dl className="mt-2 space-y-1 rounded-xl bg-ink-950/50 p-3 font-mono text-xs">
              {c2pa.issuer && (
                <div className="flex gap-2">
                  <dt className="text-ink-500">Issuer:</dt>
                  <dd className="text-ink-200">{c2pa.issuer}</dd>
                </div>
              )}
              {c2pa.claim_generator && (
                <div className="flex gap-2">
                  <dt className="text-ink-500">Generator:</dt>
                  <dd className="text-ink-200">{c2pa.claim_generator}</dd>
                </div>
              )}
              {c2pa.assertions && c2pa.assertions.length > 0 && (
                <div className="flex gap-2">
                  <dt className="text-ink-500">Assertions:</dt>
                  <dd className="text-ink-200">{c2pa.assertions.join(", ")}</dd>
                </div>
              )}
            </dl>
          )}

          {!c2pa.verified && c2pa.present && c2pa.marker && (
            <p className="mt-1 text-xs text-ink-400">{c2pa.marker}</p>
          )}

          {!c2pa.present && (
            <p className="mt-1 text-xs text-ink-400">
              No Content Credentials manifest embedded. This is normal for most internet images —
              C2PA adoption is still early.
            </p>
          )}

          {c2pa.error && !c2pa.verified && (
            <p className="mt-1 text-xs text-ink-500 italic">{c2pa.error}</p>
          )}
        </div>
      )}

      {/* Divider */}
      {c2paSignal && exifSignal && (
        <div className="my-3 border-t border-ink-700/60" />
      )}

      {/* EXIF block */}
      {exifSignal && (
        <div>
          <div className="mb-1 flex items-center justify-between gap-2 flex-wrap">
            <span className="text-sm font-medium text-ink-100">Camera EXIF</span>
            {hasExifCameraFields ? (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-semibold text-emerald-400 ring-1 ring-emerald-500/40">
                <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path d="M20 6L9 17l-5-5" />
                </svg>
                Present
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-ink-700/40 px-3 py-1 text-xs text-ink-400 ring-1 ring-ink-600/40">
                <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <path d="M15 9l-6 6M9 9l6 6" />
                </svg>
                Missing
              </span>
            )}
          </div>

          {hasExifCameraFields && (
            <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 rounded-xl bg-ink-950/50 p-3 font-mono text-xs">
              {Object.entries(exifFields).map(([k, v]) =>
                v ? (
                  <div key={k} className="flex gap-1 col-span-2 sm:col-span-1">
                    <dt className="capitalize text-ink-500">{k}:</dt>
                    <dd className="text-ink-200 truncate">{String(v)}</dd>
                  </div>
                ) : null
              )}
            </dl>
          )}

          {!hasExifCameraFields && (
            <p className="mt-1 text-xs text-ink-400">{exifSignal.plain_language}</p>
          )}
        </div>
      )}
    </section>
  );
}
