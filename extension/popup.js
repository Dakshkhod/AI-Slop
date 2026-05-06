const DEFAULT_API = "http://127.0.0.1:8000";

const apiInput   = document.getElementById("api-url");
const saveBtn    = document.getElementById("save-btn");
const savedMsg   = document.getElementById("saved-msg");
const pauseToggle = document.getElementById("pause-toggle");
const siteLabel  = document.getElementById("site-label");
const totalRow   = document.getElementById("total-row");
const countReal  = document.getElementById("count-real");
const countAi    = document.getElementById("count-ai");
const countInc   = document.getElementById("count-inc");

let currentHostname = "";

// -------------------------------------------------------------------------
// Load saved API URL
// -------------------------------------------------------------------------
chrome.storage.sync.get("apiUrl", ({ apiUrl }) => {
  apiInput.value = apiUrl || DEFAULT_API;
});

saveBtn.addEventListener("click", () => {
  const url = (apiInput.value || "").trim() || DEFAULT_API;
  chrome.storage.sync.set({ apiUrl: url }, () => {
    savedMsg.textContent = "Saved.";
    setTimeout(() => (savedMsg.textContent = ""), 1500);
  });
});

// -------------------------------------------------------------------------
// Pause toggle — scoped to the current site
// -------------------------------------------------------------------------
chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
  if (!tab) return;

  try {
    const url = new URL(tab.url);
    currentHostname = url.hostname;
    siteLabel.textContent = currentHostname || "unknown site";
  } catch (_) {
    siteLabel.textContent = "unknown site";
  }

  if (!currentHostname) return;

  const pauseKey = "paused_" + currentHostname;
  chrome.storage.local.get(pauseKey, (result) => {
    pauseToggle.checked = !!result[pauseKey];
  });

  pauseToggle.addEventListener("change", () => {
    const pauseKey = "paused_" + currentHostname;
    chrome.storage.local.set({ [pauseKey]: pauseToggle.checked });
  });

  // -------------------------------------------------------------------------
  // Request stats from the active tab's content script
  // -------------------------------------------------------------------------
  chrome.tabs.sendMessage(tab.id, { type: "TL_GET_STATS" }, (response) => {
    if (chrome.runtime.lastError || !response) {
      totalRow.textContent = "No images analysed yet on this tab.";
      countReal.textContent = "0";
      countAi.textContent   = "0";
      countInc.textContent  = "0";
      return;
    }
    renderStats(response.stats || response);
  });
});

function renderStats(stats) {
  const total = stats.total || 0;
  countReal.textContent = stats.real  || 0;
  countAi.textContent   = stats.ai    || 0;
  countInc.textContent  = stats.inconclusive || 0;
  totalRow.textContent  = total === 0
    ? "No images analysed yet on this tab."
    : `${total} image${total === 1 ? "" : "s"} analysed on this tab.`;
}
