// TruthLens content script — auto-badge scanner + result overlay.
//
// Scans visible images (>200x200 px) using IntersectionObserver, computes a
// client-side dHash for each, and requests a verdict from the background
// worker. Renders a small badge in the top-right corner of each image.
// Also handles the right-click overlay messages from the previous version.

(function () {
  "use strict";

  // -------------------------------------------------------------------------
  // Constants
  // -------------------------------------------------------------------------
  const MIN_SIDE = 200;           // minimum dimension to bother scanning
  const THROTTLE_MS = 200;        // 5 images / second max
  const BADGE_SIZE = 24;          // px, the circular badge
  const STORAGE_TTL_MS = 4 * 60 * 60 * 1000; // 4 hours local cache TTL

  // -------------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------------
  const scanned = new Map();      // imageId -> { phash, badge element }
  let badgeContainer = null;
  let paused = false;
  let imageCounter = 0;
  let lastDispatch = 0;
  const queue = [];
  let queueTimer = null;

  // Tab-level stats (sent to popup on request)
  const stats = { total: 0, real: 0, ai: 0, inconclusive: 0 };

  // -------------------------------------------------------------------------
  // Pause state — check on load and listen for changes
  // -------------------------------------------------------------------------
  function isPausedForSite() {
    const key = "paused_" + location.hostname;
    return chrome.storage.local.get(key).then((r) => !!r[key]);
  }

  chrome.storage.onChanged.addListener((changes) => {
    const key = "paused_" + location.hostname;
    if (key in changes) {
      paused = !!changes[key].newValue;
    }
  });

  // -------------------------------------------------------------------------
  // Badge container — one fixed overlay for all badges
  // -------------------------------------------------------------------------
  function ensureBadgeContainer() {
    if (badgeContainer) return badgeContainer;
    badgeContainer = document.createElement("div");
    badgeContainer.id = "tl-badge-container";
    document.documentElement.appendChild(badgeContainer);
    return badgeContainer;
  }

  // -------------------------------------------------------------------------
  // dHash — 8x8 difference hash, returns 16-char hex string or null on error
  // -------------------------------------------------------------------------
  function dhash(img) {
    try {
      const COLS = 9, ROWS = 8;
      const canvas = document.createElement("canvas");
      canvas.width = COLS;
      canvas.height = ROWS;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(img, 0, 0, COLS, ROWS);
      const data = ctx.getImageData(0, 0, COLS, ROWS).data;
      const gray = new Float32Array(COLS * ROWS);
      for (let i = 0; i < COLS * ROWS; i++) {
        gray[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
      }
      let bits = "";
      for (let row = 0; row < ROWS; row++) {
        for (let col = 0; col < 8; col++) {
          bits += gray[row * COLS + col] < gray[row * COLS + col + 1] ? "1" : "0";
        }
      }
      let hex = "";
      for (let i = 0; i < 64; i += 4) {
        hex += parseInt(bits.slice(i, i + 4), 2).toString(16);
      }
      return hex;
    } catch (_) {
      return null; // canvas tainted (cross-origin without CORS)
    }
  }

  // -------------------------------------------------------------------------
  // Badge element management
  // -------------------------------------------------------------------------
  function createBadge(imageId) {
    const el = document.createElement("div");
    el.className = "tl-badge tl-badge--pending";
    el.dataset.imageId = imageId;
    el.title = "TruthLens — analysing…";
    el.innerHTML = '<span class="tl-badge__dot"></span>';
    el.addEventListener("click", (e) => {
      e.stopPropagation();
      onBadgeClick(imageId, el);
    });
    ensureBadgeContainer().appendChild(el);
    return el;
  }

  function positionBadge(badge, img) {
    const r = img.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return;
    badge.style.top = (r.top + window.scrollY + 4) + "px";
    badge.style.left = (r.left + window.scrollX + r.width - BADGE_SIZE - 4) + "px";
  }

  function updateBadge(badge, verdict, score, phash, imageUrl) {
    badge.dataset.verdict = verdict;
    badge.dataset.phash = phash || "";
    badge.dataset.imageUrl = imageUrl || "";
    badge.dataset.score = score != null ? String(score) : "";

    if (verdict === "likely_real") {
      badge.className = "tl-badge tl-badge--real";
      badge.title = `TruthLens: Likely real (score ${Math.round(score)})`;
      badge.innerHTML = '<span class="tl-badge__icon">&#10003;</span>';
    } else if (verdict === "likely_ai") {
      badge.className = "tl-badge tl-badge--ai";
      badge.title = `TruthLens: Likely AI-generated (score ${Math.round(score)})`;
      badge.innerHTML = '<span class="tl-badge__icon">!</span>';
    } else {
      badge.className = "tl-badge tl-badge--inc";
      badge.title = `TruthLens: Inconclusive (score ${Math.round(score)})`;
      badge.innerHTML = '<span class="tl-badge__icon">?</span>';
    }
  }

  function markBadgeError(badge) {
    badge.className = "tl-badge tl-badge--error";
    badge.title = "TruthLens: Could not analyse";
    badge.innerHTML = '<span class="tl-badge__icon">&#x2013;</span>';
  }

  // -------------------------------------------------------------------------
  // Badge click — show detail overlay
  // -------------------------------------------------------------------------
  function onBadgeClick(imageId, badge) {
    const verdict = badge.dataset.verdict;
    const score = parseFloat(badge.dataset.score);
    const phash = badge.dataset.phash;
    const imageUrl = badge.dataset.imageUrl;

    if (!verdict) return;

    showDetailOverlay({ verdict, score, phash, imageUrl });
  }

  // -------------------------------------------------------------------------
  // Local verdict cache (chrome.storage.local)
  // -------------------------------------------------------------------------
  async function getCachedVerdict(phash) {
    if (!phash) return null;
    try {
      const key = "tl_verdict_" + phash;
      const result = await chrome.storage.local.get(key);
      const entry = result[key];
      if (!entry) return null;
      if (Date.now() - entry.ts > STORAGE_TTL_MS) return null;
      return entry;
    } catch (_) {
      return null;
    }
  }

  async function setCachedVerdict(phash, data) {
    if (!phash) return;
    try {
      const key = "tl_verdict_" + phash;
      await chrome.storage.local.set({ [key]: { ...data, ts: Date.now() } });
    } catch (_) {}
  }

  // -------------------------------------------------------------------------
  // Analysis queue — throttled dispatch
  // -------------------------------------------------------------------------
  function enqueue(imageId) {
    if (!queue.includes(imageId)) queue.push(imageId);
    scheduleFlush();
  }

  function scheduleFlush() {
    if (queueTimer) return;
    const delay = Math.max(0, lastDispatch + THROTTLE_MS - Date.now());
    queueTimer = setTimeout(flushQueue, delay);
  }

  function flushQueue() {
    queueTimer = null;
    if (paused || queue.length === 0) return;
    const imageId = queue.shift();
    const entry = scanned.get(imageId);
    if (entry) dispatchAnalysis(imageId, entry);
    if (queue.length > 0) scheduleFlush();
    lastDispatch = Date.now();
  }

  async function dispatchAnalysis(imageId, entry) {
    const { img, badge } = entry;
    const phash = dhash(img);
    entry.phash = phash;

    // 1. Check local storage cache
    if (phash) {
      const cached = await getCachedVerdict(phash);
      if (cached) {
        updateBadge(badge, cached.verdict, cached.score, phash, img.src);
        recordStat(cached.verdict);
        return;
      }
    }

    // 2. Ask background worker (which checks backend cache, then runs pipeline)
    chrome.runtime.sendMessage(
      {
        type: "TL_BADGE_REQUEST",
        imageId,
        phash: phash || "",
        imageUrl: img.src || img.currentSrc || "",
      },
      (response) => {
        if (chrome.runtime.lastError) return;
        if (!response) return;
        if (response.error) {
          markBadgeError(badge);
          return;
        }
        updateBadge(badge, response.verdict, response.score, phash, response.imageUrl);
        recordStat(response.verdict);
        if (phash) {
          setCachedVerdict(phash, {
            verdict: response.verdict,
            score: response.score,
          });
        }
      }
    );
  }

  // -------------------------------------------------------------------------
  // Stats
  // -------------------------------------------------------------------------
  function recordStat(verdict) {
    stats.total++;
    if (verdict === "likely_real") stats.real++;
    else if (verdict === "likely_ai") stats.ai++;
    else stats.inconclusive++;
  }

  // -------------------------------------------------------------------------
  // IntersectionObserver — only process images that are actually visible
  // -------------------------------------------------------------------------
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const img = entry.target;
        const imageId = img.dataset.tlId;
        if (!imageId || scanned.has(imageId)) continue;
        if (paused) continue;

        // Check size at the point of intersection (img may have loaded by now)
        const natW = img.naturalWidth || img.width;
        const natH = img.naturalHeight || img.height;
        if (natW < MIN_SIDE && natH < MIN_SIDE) continue;

        const badge = createBadge(imageId);
        positionBadge(badge, img);
        scanned.set(imageId, { img, badge });
        enqueue(imageId);

        observer.unobserve(img); // only process once per image
      }
    },
    { rootMargin: "200px" }   // start loading slightly before fully in viewport
  );

  // -------------------------------------------------------------------------
  // Attach observer to all eligible images (existing + future)
  // -------------------------------------------------------------------------
  function observeImage(img) {
    if (img.dataset.tlId) return;       // already registered
    const w = img.naturalWidth || img.width;
    const h = img.naturalHeight || img.height;
    // Skip tiny decorative images immediately
    if (img.complete && w > 0 && (w < MIN_SIDE && h < MIN_SIDE)) return;

    imageCounter++;
    const id = String(imageCounter);
    img.dataset.tlId = id;
    observer.observe(img);
  }

  function scanExistingImages() {
    document.querySelectorAll("img").forEach(observeImage);
  }

  const mutationObserver = new MutationObserver((mutations) => {
    for (const m of mutations) {
      for (const node of m.addedNodes) {
        if (node.nodeType !== 1) continue;
        if (node.tagName === "IMG") {
          observeImage(node);
        } else {
          node.querySelectorAll && node.querySelectorAll("img").forEach(observeImage);
        }
      }
    }
  });

  // -------------------------------------------------------------------------
  // Reposition badges on scroll / resize
  // -------------------------------------------------------------------------
  let repositionFrame = null;
  function repositionAllBadges() {
    if (repositionFrame) return;
    repositionFrame = requestAnimationFrame(() => {
      repositionFrame = null;
      for (const [id, entry] of scanned) {
        if (entry.badge && entry.img) {
          positionBadge(entry.badge, entry.img);
        }
      }
    });
  }

  window.addEventListener("scroll", repositionAllBadges, { passive: true });
  window.addEventListener("resize", repositionAllBadges, { passive: true });

  // -------------------------------------------------------------------------
  // Detail overlay (shown when user clicks a badge)
  // -------------------------------------------------------------------------
  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function showDetailOverlay({ verdict, score, phash, imageUrl }) {
    let el = document.getElementById("truthlens-overlay");
    if (!el) {
      el = document.createElement("div");
      el.id = "truthlens-overlay";
      document.documentElement.appendChild(el);
    }

    const cls = verdict === "likely_real" ? "real" : verdict === "likely_ai" ? "ai" : "inc";
    const label = verdict === "likely_real" ? "Likely real"
      : verdict === "likely_ai" ? "Likely AI-generated"
      : "Inconclusive";
    const scoreInt = Math.round(score || 0);

    el.innerHTML = `
      <div class="tl-head">
        <div class="tl-title">TruthLens result</div>
        <button class="tl-close" aria-label="close">&times;</button>
      </div>
      <div class="tl-score">
        <div class="tl-num">${scoreInt}</div>
        <div>
          <div class="tl-verdict ${cls}">${escapeHtml(label)}</div>
          <div style="color:#9aa1c2;font-size:11px;margin-top:2px;">
            Authenticity score 0–100 (higher = more real)
          </div>
        </div>
      </div>
      <div style="margin-top:10px;font-size:11px;color:#9aa1c2;">
        ${phash ? `Hash: <code style="color:#c8cce0">${escapeHtml(phash)}</code>` : ""}
      </div>
      <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;">
        <button class="tl-btn" id="tl-report-btn">Report wrong verdict</button>
      </div>
      <p style="margin-top:10px;color:#9aa1c2;font-size:11px;">
        Open the TruthLens web app for full signal breakdown and heatmap.
      </p>`;

    el.querySelector(".tl-close").addEventListener("click", () => el.remove());
    el.querySelector("#tl-report-btn").addEventListener("click", () => {
      chrome.runtime.sendMessage({
        type: "TL_REPORT_WRONG",
        phash: phash || "",
        imageUrl: imageUrl || "",
        systemVerdict: { verdict, score },
      });
      el.querySelector("#tl-report-btn").textContent = "Report sent";
      el.querySelector("#tl-report-btn").disabled = true;
    });
  }

  // -------------------------------------------------------------------------
  // Messages from background worker (right-click overlay path)
  // -------------------------------------------------------------------------
  chrome.runtime.onMessage.addListener((msg) => {
    if (!msg || !msg.type) return;

    if (msg.type === "TL_BUSY") {
      showBusyOverlay(msg.url || "");
    } else if (msg.type === "TL_RESULT") {
      showFullResultOverlay(msg.result);
    } else if (msg.type === "TL_ERROR") {
      showErrorOverlay(msg.message || "Unknown error");
    } else if (msg.type === "TL_GET_STATS") {
      // Popup is requesting current tab stats
      chrome.runtime.sendMessage({ type: "TL_STATS", stats });
    }
  });

  function showBusyOverlay(url) {
    let el = document.getElementById("truthlens-overlay");
    if (!el) { el = document.createElement("div"); el.id = "truthlens-overlay"; document.documentElement.appendChild(el); }
    el.innerHTML = `
      <div class="tl-head"><div class="tl-title">TruthLens</div><button class="tl-close">&times;</button></div>
      <div class="tl-busy"><span class="tl-spinner"></span><span>Analysing&hellip;</span></div>`;
    el.querySelector(".tl-close").addEventListener("click", () => el.remove());
  }

  function showErrorOverlay(message) {
    let el = document.getElementById("truthlens-overlay");
    if (!el) { el = document.createElement("div"); el.id = "truthlens-overlay"; document.documentElement.appendChild(el); }
    el.innerHTML = `
      <div class="tl-head"><div class="tl-title">TruthLens &mdash; error</div><button class="tl-close">&times;</button></div>
      <div style="color:#ff5577">${escapeHtml(message)}</div>
      <p style="color:#9aa1c2;font-size:12px;margin-top:8px;">Make sure the backend is running and the API URL is correct.</p>`;
    el.querySelector(".tl-close").addEventListener("click", () => el.remove());
  }

  function showFullResultOverlay(result) {
    const trail = (result.forensic_trail || []).slice(0, 6);
    const sigs = (result.signals || [])
      .filter((s) => !s.error)
      .sort((a, b) => Math.abs(b.p_ai - 0.5) - Math.abs(a.p_ai - 0.5))
      .slice(0, 6);
    const cls = result.verdict === "likely_real" ? "real" : result.verdict === "likely_ai" ? "ai" : "inc";

    let el = document.getElementById("truthlens-overlay");
    if (!el) { el = document.createElement("div"); el.id = "truthlens-overlay"; document.documentElement.appendChild(el); }
    el.innerHTML = `
      <div class="tl-head"><div class="tl-title">TruthLens result</div><button class="tl-close">&times;</button></div>
      <div class="tl-score">
        <div class="tl-num">${Math.round(result.score)}</div>
        <div>
          <div class="tl-verdict ${cls}">${escapeHtml(result.verdict_label)}</div>
          <div style="color:#9aa1c2;font-size:11px;margin-top:2px;">
            &plusmn; ${result.score_uncertainty.toFixed(1)} &middot; p(AI) = ${result.p_ai.toFixed(2)} &middot; ${result.processing_ms} ms
          </div>
        </div>
      </div>
      ${trail.length ? `<h4>Forensic trail</h4><ol>${trail.map((t) => `<li>${escapeHtml(t)}</li>`).join("")}</ol>` : ""}
      <h4>Top signals</h4>
      <ul>${sigs.map((s) => `<li><div class="tl-row"><span>${escapeHtml(s.name)}</span><span>p=${s.p_ai.toFixed(2)}</span></div></li>`).join("")}</ul>`;
    el.querySelector(".tl-close").addEventListener("click", () => el.remove());
  }

  // -------------------------------------------------------------------------
  // Initialise
  // -------------------------------------------------------------------------
  async function init() {
    paused = await isPausedForSite().catch(() => false);
    if (paused) return;

    ensureBadgeContainer();
    scanExistingImages();
    mutationObserver.observe(document.body || document.documentElement, {
      childList: true,
      subtree: true,
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
