"use client";

import { useCallback, useRef, useState } from "react";

type Props = {
  onFile: (file: File) => void;
  onUrl: (url: string) => void;
  busy: boolean;
};

export function UploadZone({ onFile, onUrl, busy }: Props) {
  const [drag, setDrag] = useState(false);
  const [url, setUrl] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDrag(false);
      if (busy) return;
      const file = e.dataTransfer.files?.[0];
      if (file) onFile(file);
    },
    [onFile, busy]
  );

  return (
    <div className="space-y-4">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={handleDrop}
        onClick={() => !busy && inputRef.current?.click()}
        className={[
          "group relative cursor-pointer rounded-3xl border-2 border-dashed p-10 text-center transition",
          "bg-ink-900/40 backdrop-blur",
          drag
            ? "border-accent-500 shadow-glow"
            : "border-ink-700 hover:border-accent-600 hover:shadow-glow",
          busy ? "opacity-60 cursor-not-allowed" : "",
        ].join(" ")}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*,video/*,audio/*"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onFile(f);
          }}
        />
        <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-ink-800 text-accent-400 ring-1 ring-ink-700 group-hover:ring-accent-600">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
            <path d="M12 16V4M12 4l-4 4M12 4l4 4" />
            <rect x="4" y="16" width="16" height="4" rx="1.5" />
          </svg>
        </div>
        <h3 className="text-xl font-semibold tracking-tight text-ink-50">
          Drop an image, video, or audio file
        </h3>
        <p className="mx-auto mt-1 max-w-md text-sm text-ink-300">
          Or click to browse. JPG / PNG / WebP / MP4 / MOV / WAV / MP3 — up to 50 MB.
          Files are analysed in-memory and never stored.
        </p>
      </div>

      <div className="flex items-center gap-3 text-xs uppercase tracking-widest text-ink-400">
        <div className="h-px flex-1 bg-ink-700" />
        or analyse a URL
        <div className="h-px flex-1 bg-ink-700" />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (url.trim() && !busy) onUrl(url.trim());
        }}
        className="flex gap-2"
      >
        <input
          type="url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://example.com/photo.jpg"
          className="flex-1 rounded-xl border border-ink-700 bg-ink-900/60 px-4 py-3 text-ink-50 placeholder:text-ink-400 focus:border-accent-500 focus:outline-none focus:ring-2 focus:ring-accent-500/40"
          disabled={busy}
        />
        <button
          type="submit"
          disabled={busy || !url.trim()}
          className="rounded-xl bg-accent-500 px-5 py-3 font-semibold text-ink-950 transition hover:bg-accent-400 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Analyse
        </button>
      </form>
    </div>
  );
}
