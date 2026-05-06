# TruthLens Browser Extension

Manifest V3 Chrome extension that automatically badges every image on a page
as likely real, likely AI-generated, or inconclusive. Right-clicking any image
opens a full forensic overlay.

---

## How it works

1. `content.js` runs on every page and watches for images larger than 200×200 px
   using `IntersectionObserver` (only visible images are processed).
2. For each qualifying image, a client-side dHash (64-bit difference hash) is
   computed from the image pixels via the Canvas API.
3. The hash is sent to the background service worker (`background.js`), which:
   - Checks `chrome.storage.local` for a cached verdict (4-hour TTL).
   - If not cached, calls `POST /api/check_hash` on the backend for a server-side
     cache lookup.
   - If still not cached, calls `POST /api/analyze_url` and waits for the full
     pipeline result.
4. The verdict is sent back to the content script, which renders a 24 px circular
   badge in the top-right corner of the image.
5. Clicking a badge shows a detail overlay. Clicking the toolbar icon shows
   per-tab stats (real / AI / inconclusive counts) and a pause toggle.

Badge colours:
- Green check — likely real (authenticity score > 70)
- Amber ?     — inconclusive (score 40–70)
- Red !       — likely AI-generated (score < 40)

---

## Install for local testing

### Prerequisites

- Chrome or any Chromium-based browser (Edge, Brave, Arc)
- TruthLens backend running locally (see [`backend/`](../backend/))

### Steps

1. Start the backend:
   ```bash
   cd backend
   .venv\Scripts\activate   # Windows
   uvicorn app.main:app --reload --port 8000
   ```

2. Open `chrome://extensions` in Chrome.

3. Enable **Developer mode** (toggle in the top-right corner).

4. Click **Load unpacked** and select the `extension/` folder from this repo.

5. The TruthLens icon appears in your toolbar. Click it to set the backend URL
   if you are not using the default `http://127.0.0.1:8000`.

6. Navigate to any page with images. Badges appear automatically on images
   larger than 200×200 px.

### Verify the install

Open `extension/test_extension.html` in Chrome (File → Open File or drag it
to a Chrome tab). Large images should receive coloured badges within a few
seconds once the backend has analysed them.

---

## Privacy

Only image hashes (64-bit dHash) leave the browser during the cache-check
step. The full image is only fetched by the backend when no cached verdict
exists for that hash. The backend fetches the image directly from its public
URL — your browser does not upload image data.

Reports submitted via "Report wrong verdict" send only the image hash, the
public image URL, and the system's verdict score. No personal information is
transmitted.

---

## Publishing to the Chrome Web Store (future)

1. Bump the `version` field in `manifest.json`.
2. Zip the contents of the `extension/` folder (not the folder itself):
   ```bash
   cd extension
   zip -r ../truthlens-extension.zip . --exclude "*.DS_Store" --exclude "test_extension.html"
   ```
3. Upload the zip at [Chrome Web Store Developer Dashboard](https://chrome.google.com/webstore/devconsole).
4. Fill in the store listing: description, screenshots, privacy policy (see above).
5. Submit for review (typically 1–3 business days).

Note: the production backend URL must be set in the published extension's
default or via the popup settings. Update `DEFAULT_API` in `background.js`
before packaging for production.

---

## File layout

```
extension/
├── manifest.json        Manifest V3 definition
├── background.js        Service worker — API calls, context menu, caching
├── content.js           Page scanner — IntersectionObserver, dHash, badges
├── content.css          Overlay styles (right-click result panel)
├── styles/
│   └── badge.css        Inline image badge styles
├── popup.html           Toolbar popup — stats + settings
├── popup.js             Popup logic
├── icons/
│   ├── icon-16.png
│   ├── icon-48.png
│   └── icon-128.png
└── test_extension.html  Local test page
```
