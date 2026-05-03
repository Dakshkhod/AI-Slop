const DEFAULT_API = "http://127.0.0.1:8000";

const apiInput = document.getElementById("api");
const saveBtn = document.getElementById("save");
const status = document.getElementById("status");

chrome.storage.sync.get("apiUrl", ({ apiUrl }) => {
  apiInput.value = apiUrl || DEFAULT_API;
});

saveBtn.addEventListener("click", () => {
  const url = apiInput.value.trim() || DEFAULT_API;
  chrome.storage.sync.set({ apiUrl: url }, () => {
    status.textContent = "Saved.";
    setTimeout(() => (status.textContent = ""), 1500);
  });
});
