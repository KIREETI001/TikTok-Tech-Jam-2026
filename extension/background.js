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
  // Where the AI band starts, deliberately far above the model's calibrated
  // 0.51. Calling an image generated is an accusation, and the panel should
  // only make one when the model is well past merely leaning that way --
  // see README, "Why the AI band sits so far above the threshold".
  aiBand: 0.9,
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

async function postPredict(blob) {
  const { apiBase } = await settings();
  if (!apiBase) throw new Error("No detector URL set — open the extension popup.");
  const body = new FormData();
  body.append("file", blob, "image");
  const r = await fetch(apiBase.replace(/\/$/, "") + "/predict", { method: "POST", body });
  if (!r.ok) {
    const detail = await r.json().catch(() => ({}));
    throw new Error(detail.detail || "Detector returned " + r.status);
  }
  const d = await r.json();
  return { p_ai: d.p_ai, verdict: d.verdict, threshold: d.threshold, confidence: d.confidence };
}

async function score(url) {
  if (cache.has(url)) return { ...cache.get(url), cached: true };

  const img = await fetch(url, { credentials: "omit" });
  if (!img.ok) throw new Error("Could not fetch image (" + img.status + ")");
  const blob = await img.blob();
  if (!blob.type.startsWith("image/")) throw new Error("Not an image");

  const value = await postPredict(blob);
  remember(url, value);
  return value;
}

/* ---------------------------------------------------------------------------
 * Scoring what is on screen rather than what is at a URL.
 *
 * A <video> has no image URL to fetch, and a canvas frame grab is blocked
 * outright: drawing cross-origin video taints the canvas, so toBlob throws.
 * captureVisibleTab sidesteps both -- it is a browser-level screenshot, so
 * same-origin rules never enter into it.
 *
 * The cost is real, and is surfaced in the panel rather than hidden. A capture
 * is a *rendered* frame: already scaled to the player's size by the browser
 * and already through the site's video codec. This detector is built on the
 * finding that resizing destroys the artifact, so a reading taken this way is
 * weaker evidence than one taken from original bytes, and says so.
 * ------------------------------------------------------------------------ */

async function toDataURL(blob) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let s = "";
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return "data:" + blob.type + ";base64," + btoa(s);
}

async function captureAndScore(windowId, rect, view) {
  const shot = await chrome.tabs.captureVisibleTab(windowId, { format: "png" });
  const bmp = await createImageBitmap(await (await fetch(shot)).blob());

  // Derive the scale from the capture rather than from devicePixelRatio. Under
  // OS display scaling the screenshot comes back at 2x while the page still
  // reports a ratio of 1, and cropping by the wrong factor does not fail --
  // it silently returns a different part of the screen, which is worse than
  // an error because the reading looks perfectly plausible.
  const k = bmp.width / view.w;
  const sx = Math.max(0, Math.round(rect.x * k));
  const sy = Math.max(0, Math.round(rect.y * k));
  const sw = Math.min(bmp.width - sx, Math.round(rect.w * k));
  const sh = Math.min(bmp.height - sy, Math.round(rect.h * k));
  if (sw < 32 || sh < 32) { bmp.close(); throw new Error("Main element is too small to read"); }

  const full = new OffscreenCanvas(sw, sh);
  full.getContext("2d").drawImage(bmp, sx, sy, sw, sh, 0, 0, sw, sh);
  const blob = await full.convertToBlob({ type: "image/png" });

  // A thumbnail of exactly the pixels that were scored, so the panel shows
  // what the model saw rather than a stand-in for it.
  const t = Math.min(1, 140 / Math.max(sw, sh));
  const small = new OffscreenCanvas(Math.max(1, Math.round(sw * t)), Math.max(1, Math.round(sh * t)));
  small.getContext("2d").drawImage(bmp, sx, sy, sw, sh, 0, 0, small.width, small.height);
  const thumb = await toDataURL(await small.convertToBlob({ type: "image/jpeg", quality: 0.72 }));
  bmp.close();

  return { ...(await postPredict(blob)), thumb, viaCapture: true };
}

/* Ask the page which element is the main one, then score it by whichever route
 * gives the better evidence: original bytes for an image, a screen capture for
 * anything else. */
async function checkMain(tab) {
  const pick = await chrome.tabs.sendMessage(tab.id, { type: "findMain" });
  if (!pick || !pick.ok) {
    throw new Error((pick && pick.error) || "Nothing large enough to check on this page");
  }
  chrome.tabs.sendMessage(tab.id, { type: "scanning", pick: true });
  if (pick.kind === "img" && pick.url && !pick.url.startsWith("blob:")) {
    return { ...(await score(pick.url)), url: pick.url };
  }
  return await captureAndScore(tab.windowId, pick.rect, pick.view);
}

chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
  if (msg.type === "settings") {
    settings().then((s) => reply(s));
    return true;                                    // keep the channel open
  }
  return false;
});

/* ---- the two ways anything gets scored, both of them deliberate ---- */
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    // The main-content item is listed first and offered everywhere, because on
    // a feed it is almost always the one you want: the images you *can*
    // right-click there are the sidebar thumbnails and the avatars, and the
    // thing you are actually looking at is usually a <video> that offers no
    // image menu at all.
    chrome.contextMenus.create({
      id: "cocMain",
      title: "Check the main image on this page",
      contexts: ["page", "video", "image", "frame", "selection", "link"],
    });
    chrome.contextMenus.create({
      id: "cocScan",
      title: "Check this image for AI",
      contexts: ["image"],
    });
  });
});

async function run(tab, work, url) {
  // Acknowledge immediately. Fetching the image and running the forward pass
  // takes long enough that a silent gap reads as "the click did nothing".
  if (url) chrome.tabs.sendMessage(tab.id, { type: "scanning", url });
  try {
    const v = await work();
    chrome.tabs.sendMessage(tab.id, { type: "result", url, ...v, ok: true, forced: true });
  } catch (e) {
    chrome.tabs.sendMessage(tab.id, {
      type: "result", url, ok: false, error: String(e.message || e), forced: true,
    });
  }
}

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (!tab?.id) return;
  if (info.menuItemId === "cocScan") run(tab, () => score(info.srcUrl), info.srcUrl);
  // checkMain sends its own "scanning" once it knows what it picked.
  else if (info.menuItemId === "cocMain") run(tab, () => checkMain(tab));
});

/* Keyboard route, for a feed you are scrolling with one hand. */
chrome.commands.onCommand.addListener(async (command) => {
  if (command !== "check-main") return;
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab?.id) run(tab, () => checkMain(tab));
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
