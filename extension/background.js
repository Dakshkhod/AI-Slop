// TruthLens — MV3 service worker.
//
// Handles two flows:
//   1. Right-click context menu → fetch URL → full analysis → overlay result
//   2. Badge requests from content.js → check_hash → analyze_url → badge update

const DEFAULT_API = "http://127.0.0.1:8000";

async function getApiUrl() {
  const { apiUrl } = await chrome.storage.sync.get("apiUrl");
  return apiUrl || DEFAULT_API;
}

// -------------------------------------------------------------------------
// Context menu setup (right-click flow)
// -------------------------------------------------------------------------
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "truthlens-image",
    title: "TruthLens — analyse this image",
    contexts: ["image"],
  });
  chrome.contextMenus.create({
    id: "truthlens-video",
    title: "TruthLens — analyse this video",
    contexts: ["video"],
  });
  chrome.contextMenus.create({
    id: "truthlens-audio",
    title: "TruthLens — analyse this audio",
    contexts: ["audio"],
  });
  chrome.contextMenus.create({
    id: "truthlens-link",
    title: "TruthLens — analyse linked media",
    contexts: ["link"],
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (!tab?.id) return;
  const target = info.srcUrl || info.linkUrl;
  if (!target) return;

  await chrome.tabs.sendMessage(tab.id, { type: "TL_BUSY", url: target });
  try {
    const apiUrl = await getApiUrl();
    const fd = new FormData();
    fd.append("url", target);
    const res = await fetch(`${apiUrl}/api/analyze-url`, {
      method: "POST",
      body: fd,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
    const json = await res.json();
    await chrome.tabs.sendMessage(tab.id, { type: "TL_RESULT", url: target, result: json });
  } catch (err) {
    await chrome.tabs.sendMessage(tab.id, {
      type: "TL_ERROR",
      url: target,
      message: err?.message ?? String(err),
    });
  }
});

// -------------------------------------------------------------------------
// Badge request flow (from content.js)
// -------------------------------------------------------------------------
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (!msg || !msg.type) return false;

  if (msg.type === "TL_BADGE_REQUEST") {
    handleBadgeRequest(msg).then(sendResponse).catch((err) => {
      sendResponse({ error: err?.message ?? String(err) });
    });
    return true; // keep channel open for async response
  }

  if (msg.type === "TL_REPORT_WRONG") {
    handleReportWrong(msg).catch(() => {});
    return false;
  }

  return false;
});

async function handleBadgeRequest({ phash, imageUrl }) {
  if (!imageUrl) return { error: "no image URL" };

  const apiUrl = await getApiUrl();

  // Step 1: check backend cache
  if (phash) {
    try {
      const res = await fetch(`${apiUrl}/api/check_hash`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phash, image_url: imageUrl }),
      });
      if (res.ok) {
        const data = await res.json();
        if (data.status === "cached") {
          return {
            verdict: data.verdict,
            score: data.score,
            imageUrl,
          };
        }
      }
    } catch (_) {
      // Backend unreachable; fall through to direct analysis
    }
  }

  // Step 2: full synchronous analysis
  try {
    const res = await fetch(`${apiUrl}/api/analyze_url`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_url: imageUrl, phash: phash || "" }),
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`HTTP ${res.status}: ${txt.slice(0, 120)}`);
    }
    const data = await res.json();
    return {
      verdict: data.verdict,
      score: data.score,
      imageUrl,
    };
  } catch (err) {
    return { error: err?.message ?? String(err) };
  }
}

async function handleReportWrong({ phash, imageUrl, systemVerdict }) {
  const apiUrl = await getApiUrl();
  try {
    await fetch(`${apiUrl}/api/report_wrong`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        phash: phash || "",
        image_url: imageUrl || "",
        user_verdict: "unknown", // user hasn't specified; just flag it
        system_verdict: systemVerdict || {},
      }),
    });
  } catch (_) {}
}
