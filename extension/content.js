/* Chain of Custody — page-side scanning and badges.
 *
 * Two modes, deliberately different in temperament:
 *
 *   on demand  right-click an image. One question asked, one answer given,
 *              reported at the model's calibrated threshold.
 *   ambient    every large image in the viewport gets scored as you scroll.
 *              Off by default, and held to a much higher bar, because this
 *              mode makes accusations nobody asked for. At 6.7% false-positive
 *              rate a hundred real photos produce about seven wrong flags --
 *              tolerable when you asked about one image, corrosive when it is
 *              happening to your friends' holiday pictures unprompted.
 */

const STATE = new WeakMap();     // img -> {status, p_ai, verdict, badge}
let CFG = { ambient: false, ambientThreshold: 0.9, minSize: 200 };
let layer = null;
let queued = [];
let flushing = false;

function ensureLayer() {
  // isConnected, not document.body.contains -- the layer is appended to
  // documentElement, so a body-based check is always false and every call
  // would build another layer.
  if (layer && layer.isConnected) return layer;
  layer = document.createElement("div");
  layer.className = "coc-layer";
  document.documentElement.appendChild(layer);
  return layer;
}

function bigEnough(img) {
  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  return w >= CFG.minSize && h >= CFG.minSize;
}

function badgeFor(img) {
  const st = STATE.get(img);
  if (st && st.badge) return st.badge;
  const b = document.createElement("div");
  b.className = "coc-badge coc-pending";
  b.textContent = "…";
  ensureLayer().appendChild(b);
  return b;
}

function place(img, badge) {
  const r = img.getBoundingClientRect();
  if (r.width < 8 || r.bottom < 0 || r.top > innerHeight) {
    badge.style.display = "none";
    return;
  }
  badge.style.display = "";
  badge.style.transform = "translate(" + Math.round(r.left + 8) + "px," +
                                          Math.round(r.top + 8) + "px)";
}

function paint(img, res) {
  const badge = badgeFor(img);
  const st = STATE.get(img) || {};
  st.badge = badge;

  if (!res.ok) {
    st.status = "error";
    badge.className = "coc-badge coc-error";
    badge.textContent = "—";
    badge.title = "Chain of Custody: " + res.error;
  } else {
    // The bar an image must clear to be called out. On demand we use the
    // model's own calibrated threshold; ambient uses the stricter one.
    const bar = res.forced ? res.threshold : Math.max(res.threshold, CFG.ambientThreshold);
    const flag = res.p_ai >= bar;
    st.status = "done"; st.p_ai = res.p_ai;
    badge.className = "coc-badge " + (flag ? "coc-ai" : "coc-real");
    badge.textContent = (res.p_ai * 100).toFixed(0) + "%";
    badge.title = "Chain of Custody — P(AI) " + (res.p_ai * 100).toFixed(1) + "%\n" +
      (flag ? "flagged AI-generated" : "not flagged") +
      "\nbar for this mode: " + (bar * 100).toFixed(0) + "%" +
      (res.cached ? "\n(cached)" : "");
  }
  STATE.set(img, st);
  place(img, badge);
}

function request(img) {
  const st = STATE.get(img) || {};
  if (st.status === "pending" || st.status === "done") return;
  const url = img.currentSrc || img.src;
  if (!url || url.startsWith("data:")) return;
  st.status = "pending";
  st.badge = badgeFor(img);
  STATE.set(img, st);
  place(img, st.badge);

  chrome.runtime.sendMessage({ type: "score", url }, (res) => {
    if (chrome.runtime.lastError) {
      paint(img, { ok: false, error: chrome.runtime.lastError.message });
      return;
    }
    paint(img, res || { ok: false, error: "no response" });
  });
}

/* Score only what is actually on screen, and only once it has settled. */
const io = new IntersectionObserver((entries) => {
  if (!CFG.ambient) return;
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    const img = e.target;
    if (!bigEnough(img)) { io.unobserve(img); continue; }
    queued.push(img);
  }
  flush();
}, { rootMargin: "100px", threshold: 0.25 });

function flush() {
  if (flushing || !queued.length) return;
  flushing = true;
  const batch = queued.splice(0, 4);          // keep the tab responsive
  batch.forEach(request);
  setTimeout(() => { flushing = false; flush(); }, 250);
}

function sweep(root) {
  const imgs = (root instanceof Element ? root : document).querySelectorAll("img");
  imgs.forEach((img) => {
    if (STATE.has(img)) return;
    STATE.set(img, { status: "seen" });
    if (img.complete) { if (bigEnough(img)) io.observe(img); }
    else img.addEventListener("load", () => { if (bigEnough(img)) io.observe(img); }, { once: true });
  });
}

/* Instagram replaces feed nodes constantly; without this the radar goes quiet
 * after the first screen. */
const mo = new MutationObserver((muts) => {
  for (const m of muts) {
    m.addedNodes.forEach((n) => { if (n.nodeType === 1) sweep(n); });
  }
});

let raf = 0;
function reposition() {
  if (raf) return;
  raf = requestAnimationFrame(() => {
    raf = 0;
    if (!layer) return;
    document.querySelectorAll("img").forEach((img) => {
      const st = STATE.get(img);
      if (st && st.badge) place(img, st.badge);
    });
  });
}

/* A right-click scan always paints, whatever mode we are in. */
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type !== "result") return;
  const img = [...document.querySelectorAll("img")]
    .find((i) => (i.currentSrc || i.src) === msg.url);
  if (img) { STATE.set(img, { ...(STATE.get(img) || {}), status: "seen" }); paint(img, msg); }
});

chrome.storage.onChanged.addListener((changes) => {
  for (const k of Object.keys(changes)) {
    if (k in CFG) CFG[k] = changes[k].newValue;
  }
  if (CFG.ambient) sweep(document);
});

chrome.runtime.sendMessage({ type: "settings" }, (s) => {
  if (chrome.runtime.lastError || !s) return;
  CFG = { ambient: s.ambient, ambientThreshold: s.ambientThreshold, minSize: s.minSize };
  ensureLayer();
  sweep(document);
  mo.observe(document.documentElement, { childList: true, subtree: true });
  addEventListener("scroll", reposition, { passive: true });
  addEventListener("resize", reposition, { passive: true });
});
