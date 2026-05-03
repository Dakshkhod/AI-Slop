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
