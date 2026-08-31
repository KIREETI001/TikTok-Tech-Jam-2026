/* Chain of Custody — page-side scanning and badges.
 *
 * Two modes, deliberately different in temperament:
 *
 *   on demand  right-click an image -> "Check this image for AI". One question
 *              asked, one answer given. This is the default and the mode the
 *              extension is really for.
 *   ambient    every large image in the viewport is scored as you scroll. Off
 *              by default, and -- unlike the first version -- silent unless a
 *              score lands in the top band. A badge on every image is not a
 *              radar, it is wallpaper: it spends the reader's attention on
 *              every photo to tell them what they already assumed about almost
 *              all of them.
 *
 * Three bands, taken from the score rather than from the mode:
 *
 *   authentic   p_ai <  threshold             the model's own calibrated point
 *   uncertain   threshold <= p_ai < high
 *   AI          p_ai >= high                  (CFG.ambientThreshold, def 0.90)
 *
 * The middle band exists because folding it into "authentic" was actively
 * misleading: a 0.80 image was painted the same colour as a 0.07 image, which
 * reads as "we checked, this is fine" when what the model actually said was
 * "probably generated, but not past the bar I will accuse someone over".
 */

const STATE = new WeakMap();     // img -> {status, p_ai, badge}
let CFG = { ambient: false, ambientThreshold: 0.9, minSize: 200 };
let layer = null;
let queued = [];
let flushing = false;

const pct = (x, d) => (x * 100).toFixed(d) + "%";

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
  // Dismissable, because an answer you asked for should also be one you can
  // put away -- otherwise the deliberate mode slowly accumulates the same
  // clutter the ambient mode was just cured of.
  b.addEventListener("click", () => {
    b.remove();
    const s = STATE.get(img);
    if (s) { delete s.badge; s.status = "dismissed"; STATE.set(img, s); }
  });
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

function bandOf(p, threshold, high) {
  if (p >= high) return { cls: "coc-ai", word: "likely AI-generated" };
  if (p >= threshold) return { cls: "coc-maybe", word: "uncertain" };
  return { cls: "coc-real", word: "likely authentic" };
}

function paint(img, res) {
  const st = STATE.get(img) || {};
  const forced = !!res.forced;

  if (!res.ok) {
    // Report a failure only for a scan someone actually asked for. An ambient
    // pass that cannot reach the detector should fail quietly rather than
    // stipple the page with dashes.
    if (!forced) { st.status = "done"; STATE.set(img, st); return; }
    const badge = badgeFor(img);
    st.badge = badge; st.status = "error";
    badge.className = "coc-badge coc-error";
    badge.textContent = "—";
    badge.title = "Chain of Custody: " + res.error + "\n\nclick to dismiss";
    STATE.set(img, st);
    place(img, badge);
    return;
  }

  const high = Math.max(res.threshold, CFG.ambientThreshold);
  const band = bandOf(res.p_ai, res.threshold, high);

  st.status = "done";
  st.p_ai = res.p_ai;

  // The point of the rework: ambient mode says nothing at all unless the score
  // reaches the band it would actually call AI.
  if (!forced && band.cls !== "coc-ai") { STATE.set(img, st); return; }

  const badge = badgeFor(img);
  st.badge = badge;
  badge.className = "coc-badge " + band.cls;
  badge.textContent = pct(res.p_ai, 0);
  // Every badge carries its own legend. There is nowhere else to put one: the
  // badge floats over somebody else's page, so if the bands are not explained
  // here they are not explained anywhere the reader is looking.
  badge.title =
    "Chain of Custody — P(AI) " + pct(res.p_ai, 1) + "\n" +
    band.word + "\n\n" +
    "authentic   below " + pct(res.threshold, 0) + "\n" +
    "uncertain   " + pct(res.threshold, 0) + " – " + pct(high, 0) + "\n" +
    "AI          " + pct(high, 0) + " and above\n\n" +
    "click to dismiss" + (res.cached ? "  ·  cached" : "");
  STATE.set(img, st);
  place(img, badge);
}

function request(img) {
  const st = STATE.get(img) || {};
  if (st.status === "pending" || st.status === "done" || st.status === "dismissed") return;
  const url = img.currentSrc || img.src;
  if (!url || url.startsWith("data:")) return;
  st.status = "pending";
  STATE.set(img, st);
  // No pending badge on this path. Ambient scoring is speculative, and a "..."
  // on every image while it resolves is exactly the noise this mode avoids.
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

function findByUrl(url) {
  return [...document.querySelectorAll("img")]
    .find((i) => (i.currentSrc || i.src) === url);
}

/* A right-click scan always paints, whatever mode we are in. */
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "scanning") {
    // The deliberate path does get a pending badge: you asked, so you are owed
    // an acknowledgement while the fetch and the forward pass run.
    const img = findByUrl(msg.url);
    if (!img) return;
    const st = STATE.get(img) || {};
    st.status = "pending";
    st.badge = badgeFor(img);
    st.badge.className = "coc-badge coc-pending";
    st.badge.textContent = "…";
    st.badge.title = "Chain of Custody — checking…";
    STATE.set(img, st);
    place(img, st.badge);
    return;
  }
  if (msg.type !== "result") return;
  const img = findByUrl(msg.url);
  if (img) {
    STATE.set(img, { ...(STATE.get(img) || {}), status: "seen" });
    paint(img, msg);
  }
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
