// TruthLens — MV3 service worker.
// Adds a context-menu entry for images / video / audio that POSTs the
// resource URL to the configured backend's /api/analyze-url endpoint and
// shows the result in an injected overlay on the page.

const DEFAULT_API = "http://127.0.0.1:8000";

async function getApiUrl() {
  const { apiUrl } = await chrome.storage.sync.get("apiUrl");
  return apiUrl || DEFAULT_API;
}

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
  const target =
    info.srcUrl ||
    info.linkUrl ||
    info.mediaType === "image"
      ? info.srcUrl
      : info.linkUrl;
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
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`HTTP ${res.status}: ${txt.slice(0, 200)}`);
    }
    const json = await res.json();
    await chrome.tabs.sendMessage(tab.id, {
      type: "TL_RESULT",
      url: target,
      result: json,
    });
  } catch (err) {
    await chrome.tabs.sendMessage(tab.id, {
      type: "TL_ERROR",
      url: target,
      message: err && err.message ? err.message : String(err),
    });
  }
});
