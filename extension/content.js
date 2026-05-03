// TruthLens — content script. Renders an overlay with results sent from
// the background service worker.

function ensureOverlay() {
  let el = document.getElementById("truthlens-overlay");
  if (el) return el;
  el = document.createElement("div");
  el.id = "truthlens-overlay";
  document.documentElement.appendChild(el);
  return el;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function header(title) {
  return `
    <div class="tl-head">
      <div class="tl-title">${escapeHtml(title)}</div>
      <button class="tl-close" aria-label="close">×</button>
    </div>`;
}

function attachClose(el) {
  const btn = el.querySelector(".tl-close");
  if (btn) btn.addEventListener("click", () => el.remove());
}

function renderBusy(url) {
  const el = ensureOverlay();
  el.innerHTML = `
    ${header("TruthLens")}
    <div class="tl-busy">
      <span class="tl-spinner"></span>
      <span>Analysing <span style="opacity:0.7">${escapeHtml(url.slice(0, 60))}…</span></span>
    </div>`;
  attachClose(el);
}

function renderError(message) {
  const el = ensureOverlay();
  el.innerHTML = `
    ${header("TruthLens — error")}
    <div style="color:#ff5577">${escapeHtml(message)}</div>
    <p style="margin-top:8px; color:#9aa1c2; font-size:12px">
      Make sure the backend is running and the API URL is set correctly in
      the extension popup.
    </p>`;
  attachClose(el);
}

function renderResult(result) {
  const el = ensureOverlay();
  const verdict = result.verdict;
  const cls = verdict === "likely_real" ? "real" : verdict === "likely_ai" ? "ai" : "inc";
  const score = Math.round(result.score);
  const trail = (result.forensic_trail || []).slice(0, 6);
  const sigs = (result.signals || [])
    .filter((s) => !s.error)
    .slice()
    .sort((a, b) => Math.abs(b.p_ai - 0.5) - Math.abs(a.p_ai - 0.5))
    .slice(0, 6);

  el.innerHTML = `
    ${header("TruthLens result")}
    <div class="tl-score">
      <div class="tl-num">${score}</div>
      <div>
        <div class="tl-verdict ${cls}">${escapeHtml(result.verdict_label)}</div>
        <div style="color:#9aa1c2; font-size:11px; margin-top:2px;">
          ± ${result.score_uncertainty.toFixed(1)} · p(AI) = ${result.p_ai.toFixed(2)}
          · ${result.processing_ms} ms
        </div>
      </div>
    </div>

    ${trail.length ? `<h4>Forensic trail</h4><ol>${trail.map((t) => `<li>${escapeHtml(t)}</li>`).join("")}</ol>` : ""}

    <h4>Top signals</h4>
    <ul>
      ${sigs
        .map(
          (s) =>
            `<li><div class="tl-row"><span>${escapeHtml(s.name)}</span><span>p=${s.p_ai.toFixed(2)}</span></div></li>`
        )
        .join("")}
    </ul>

    <p style="margin-top:10px; color:#9aa1c2; font-size:11px;">
      Open the full report in TruthLens for heatmaps and per-domain breakdown.
    </p>`;
  attachClose(el);
}

chrome.runtime.onMessage.addListener((msg) => {
  if (!msg || !msg.type) return;
  if (msg.type === "TL_BUSY") renderBusy(msg.url || "");
  else if (msg.type === "TL_RESULT") renderResult(msg.result);
  else if (msg.type === "TL_ERROR") renderError(msg.message || "Unknown error");
});
