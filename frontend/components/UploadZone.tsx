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
          "group relative cursor-pointer rounded-2xl border-2 border-dashed p-10 text-center transition",
          "bg-ink-900",
          drag
            ? "border-ink-100"
            : "border-ink-700 hover:border-ink-500",
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
        <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl border border-ink-700 bg-ink-800 text-ink-200">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
            <path d="M12 16V4M12 4l-4 4M12 4l4 4" />
            <rect x="4" y="16" width="16" height="4" rx="1.5" />
          </svg>
        </div>
        <h3 className="text-lg font-semibold tracking-tight text-ink-50">
          Drop an image, video, or audio file
        </h3>
        <p className="mx-auto mt-1 max-w-md text-[13px] text-ink-400">
          Or click to browse. JPG / PNG / WebP / MP4 / MOV / WAV / MP3 — up to 50 MB.
        </p>
      </div>

      <div className="flex items-center gap-3 text-[10px] uppercase tracking-[0.2em] text-ink-500">
        <div className="h-px flex-1 bg-ink-800" />
        or analyse a URL
        <div className="h-px flex-1 bg-ink-800" />
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
          className="flex-1 rounded-lg border border-ink-700 bg-ink-900 px-4 py-2.5 text-sm text-ink-50 placeholder:text-ink-500 focus:border-ink-500 focus:outline-none"
          disabled={busy}
        />
        <button
          type="submit"
          disabled={busy || !url.trim()}
          className="rounded-lg bg-ink-100 px-5 py-2.5 text-sm font-semibold text-black transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-40"
        >
          Analyse
        </button>
      </form>
    </div>
  );
}
