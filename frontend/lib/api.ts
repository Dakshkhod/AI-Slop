export type Verdict = "likely_real" | "inconclusive" | "likely_ai";
export type Severity = "info" | "warn" | "flag" | "pass";

export type DomainBreakdown = {
  provenance: number;
  quantum: number;
  thermodynamic: number;
  biological: number;
  semantic: number;
  ml: number;
};

export type SignalResult = {
  id: string;
  layer: number;
  name: string;
  description: string;
  severity: Severity;
  p_ai: number;
  confidence: number;
  domain: keyof DomainBreakdown;
  evidence: Record<string, unknown>;
  plain_language: string;
  error?: string | null;
};

export type TrailItem = {
  layer: number;
  id: string;
  name: string;
  severity: Severity;
  p_ai: number;
  confidence: number;
  plain_language: string;
  evidence: Record<string, unknown>;
};

export type Heatmap = {
  kind: "gradcam" | "noise" | "fft" | "block";
  mime: string;
  data_base64: string;
  description: string;
};

export type AnalyzeResponse = {
  request_id: string;
  modality: "image" | "video" | "audio";
  filename: string;
  bytes: number;
  score: number;
  score_uncertainty: number;
  verdict: Verdict;
  verdict_label: string;
  p_ai: number;
  domain_real_confidence: DomainBreakdown;
  forensic_trail: string[];
  provenance_trail: TrailItem[];
  checklist: string[];
  signals: SignalResult[];
  heatmaps: Heatmap[];
  processing_ms: number;
  model_versions: Record<string, string>;
};

export async function analyzeFile(file: File): Promise<AnalyzeResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch("/api/analyze", { method: "POST", body: fd });
  if (!r.ok) {
    const t = await r.text().catch(() => "");
    throw new Error(`Analyze failed (${r.status}): ${t}`);
  }
  return (await r.json()) as AnalyzeResponse;
}

export async function analyzeUrl(url: string): Promise<AnalyzeResponse> {
  const fd = new FormData();
  fd.append("url", url);
  const r = await fetch("/api/analyze-url", { method: "POST", body: fd });
  if (!r.ok) {
    const t = await r.text().catch(() => "");
    throw new Error(`Analyze failed (${r.status}): ${t}`);
  }
  return (await r.json()) as AnalyzeResponse;
}

// ─── Feedback ──────────────────────────────────────────────────────────────

export type FeedbackRating = "up" | "down";

export type FeedbackPayload = {
  request_id: string;
  rating: FeedbackRating;
  phash?: string;
  image_url?: string;
  system_verdict?: Record<string, unknown>;
  comment?: string;
};

export type ReportWrongPayload = {
  request_id: string;
  user_verdict: "real" | "ai";
  phash?: string;
  image_url?: string;
  filename?: string;
  file_size?: number;
  system_verdict?: Record<string, unknown>;
  comment?: string;
};

export type FeedbackStats = {
  total: number;
  report_wrong: number;
  feedback_up: number;
  feedback_down: number;
};

export async function sendFeedback(p: FeedbackPayload): Promise<{ status: string; rating: string }> {
  const r = await fetch("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(p),
  });
  if (!r.ok) throw new Error(`feedback failed (${r.status})`);
  return r.json();
}

export type ReportWrongResponse = {
  status: string;
  message?: string;
};

export async function reportWrong(p: ReportWrongPayload): Promise<ReportWrongResponse> {
  const r = await fetch("/api/report_wrong", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ phash: "", ...p }),
  });
  if (!r.ok) throw new Error(`report_wrong failed (${r.status})`);
  return r.json();
}

export async function feedbackStats(): Promise<FeedbackStats> {
  const r = await fetch("/api/feedback/stats");
  if (!r.ok) throw new Error(`feedback_stats failed (${r.status})`);
  return r.json();
}
