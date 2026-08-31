/* Chain of Custody — background service worker.
 *
 * Everything that touches the network lives here, for one specific reason: a
 * content script's fetch runs in the *page's* origin, so pulling bytes off
 * cdninstagram.com from inside instagram.com is blocked by CORS and the page
 * CSP. A service worker with host permissions has neither restriction, so it
 * can fetch the image, post it to the detector, and hand the score back.
 *
 * MV3 service workers are torn down when idle, so the score cache is
 * best-effort and deliberately small. Re-scoring a cache miss costs one
 * request; leaking memory across a long scroll would cost more.
 */

const DEFAULTS = {
  apiBase: "",
  ambient: false,
  // Deliberately far above the model's calibrated 0.51. Ambient mode makes
  // unprompted accusations about images nobody asked us to judge, so it
  // should trade recall for precision -- see README, "Why two thresholds".
  ambientThreshold: 0.9,
  minSize: 200,
};

const cache = new Map();          // image url -> {p_ai, verdict, threshold}
const CACHE_MAX = 400;

async function settings() {
  const s = await chrome.storage.sync.get(DEFAULTS);
  return { ...DEFAULTS, ...s };
}

function remember(url, value) {
  if (cache.size >= CACHE_MAX) cache.delete(cache.keys().next().value);
  cache.set(url, value);
}

async function score(url) {
  if (cache.has(url)) return { ...cache.get(url), cached: true };

  const { apiBase } = await settings();
  if (!apiBase) throw new Error("No detector URL set — open the extension popup.");

  const img = await fetch(url, { credentials: "omit" });
  if (!img.ok) throw new Error("Could not fetch image (" + img.status + ")");
  const blob = await img.blob();
  if (!blob.type.startsWith("image/")) throw new Error("Not an image");

  const body = new FormData();
  body.append("file", blob, "image");
  const r = await fetch(apiBase.replace(/\/$/, "") + "/predict", { method: "POST", body });
  if (!r.ok) {
    const detail = await r.json().catch(() => ({}));
    throw new Error(detail.detail || "Detector returned " + r.status);
  }
  const d = await r.json();
  const value = { p_ai: d.p_ai, verdict: d.verdict, threshold: d.threshold, confidence: d.confidence };
  remember(url, value);
  return value;
}

chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
  if (msg.type === "score") {
    score(msg.url)
      .then((v) => reply({ ok: true, ...v }))
      .catch((e) => reply({ ok: false, error: String(e.message || e) }));
    return true;                                    // keep the channel open
  }
  if (msg.type === "settings") {
    settings().then((s) => reply(s));
    return true;
  }
  return false;
});

/* ---- right-click a single image: the mode that is actually trustworthy ---- */
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "cocScan",
    title: "Check this image for AI",
    contexts: ["image"],
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "cocScan" || !tab?.id) return;
  try {
    const v = await score(info.srcUrl);
    chrome.tabs.sendMessage(tab.id, { type: "result", url: info.srcUrl, ...v, ok: true, forced: true });
  } catch (e) {
    chrome.tabs.sendMessage(tab.id, { type: "result", url: info.srcUrl, ok: false, error: String(e.message || e), forced: true });
  }
});

/* Health check for the popup, so a wrong URL fails there and not silently
 * in the middle of a scroll. */
chrome.runtime.onMessage.addListener((msg, _s, reply) => {
  if (msg.type !== "health") return false;
  settings().then(async (s) => {
    if (!s.apiBase) return reply({ ok: false, error: "no URL set" });
    try {
      const r = await fetch(s.apiBase.replace(/\/$/, "") + "/health");
      reply(await r.json());
    } catch (e) {
      reply({ ok: false, error: "unreachable" });
    }
  });
  return true;
});
