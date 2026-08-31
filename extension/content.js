/* Chain of Custody — one reading at a time.
 *
 * The interaction is deliberately narrow: you right-click an image, and one
 * panel in the corner reports on that image. Nothing else on the page is
 * scored, marked, or annotated.
 *
 * This replaces per-image badges, which were the wrong shape twice over. They
 * scaled with the page rather than with your attention, so a feed of thirty
 * photos became thirty verdicts nobody asked for; and once several were on
 * screen there was no way to tell which one you had actually asked about. A
 * single panel cannot have either problem -- there is only ever one reading,
 * and it names its own subject with a thumbnail.
 *
 * The reading holds for HOLD_MS and then falls back to an em dash. An answer
 * that stays forever becomes furniture: you stop reading it, and worse, you
 * lose track of which image it belonged to. Expiring it means a number on
 * screen is always about the thing you just asked about. The countdown pauses
 * while the pointer is over the panel, because a reading should not vanish
 * out from under someone who is still reading it.
 *
 * Three bands, from the score rather than the mode:
 *
 *   authentic   p_ai <  threshold      the model's own calibrated point
 *   uncertain   threshold <= p_ai < high
 *   AI          p_ai >= high           (CFG.aiBand, default 0.90)
 */

const HOLD_MS = 3000;

let CFG = { aiBand: 0.9 };
let layer = null;
let panel = null;
let ring = null;
const ui = {};
let holdTimer = 0;
let hovering = false;
let ringTarget = null;
let raf = 0;
let lastThreshold = null;

const pct = (x, d) => (x * 100).toFixed(d) + "%";

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

function ensureLayer() {
  // isConnected, not document.body.contains -- the layer is appended to
  // documentElement, so a body-based check is always false and every call
  // would build another layer.
  if (layer && layer.isConnected) return layer;
  layer = el("div", "coc-layer");
  document.documentElement.appendChild(layer);
  return layer;
}

/* ---------------------------------------------------------------- the panel */

function buildPanel() {
  ensureLayer();

  // Built with createElement rather than innerHTML: content scripts run in an
  // isolated world, but sites that enforce Trusted Types are exactly the sites
  // this most needs to work on, and there is no reason to depend on that
  // exemption holding.
  ring = el("div", "coc-ring");
  layer.appendChild(ring);

  panel = el("div", "coc-panel");
  panel.dataset.state = "idle";

  const head = el("div", "coc-head");
  head.appendChild(el("span", "coc-mark", "Chain of Custody"));
  const close = el("button", "coc-close", "×");
  close.type = "button";
  close.title = "Hide this panel";
  close.setAttribute("aria-label", "Hide the Chain of Custody panel");
  close.addEventListener("click", dismiss);
  head.appendChild(close);

  const body = el("div", "coc-body");
  ui.thumb = el("img", "coc-thumb");
  ui.thumb.alt = "";
  const read = el("div", "coc-read");
  ui.score = el("div", "coc-score", "—");
  ui.word = el("div", "coc-word", "right-click an image to check it");
  read.appendChild(ui.score);
  read.appendChild(ui.word);
  body.appendChild(ui.thumb);
  body.appendChild(read);

  ui.legend = el("div", "coc-legend");

  panel.appendChild(head);
  panel.appendChild(body);
  panel.appendChild(ui.legend);

  panel.addEventListener("mouseenter", () => { hovering = true; });
  panel.addEventListener("mouseleave", () => { hovering = false; });

  layer.appendChild(panel);
  renderLegend();
}

function ensurePanel() {
  if (!panel || !panel.isConnected) buildPanel();
  return panel;
}

/* The legend gains its numbers once a scan has told us the model's threshold.
 * Until then it names the bands without inventing boundaries for them. */
function renderLegend() {
  const bands = [
    ["real", "authentic", lastThreshold == null ? "" : "< " + pct(lastThreshold, 0)],
    ["maybe", "uncertain", lastThreshold == null ? "" :
      pct(lastThreshold, 0) + "–" + pct(highBar(), 0)],
    ["ai", "AI", lastThreshold == null ? "" : "≥ " + pct(highBar(), 0)],
  ];
  ui.legend.textContent = "";
  for (const [key, name, range] of bands) {
    const row = el("span", "coc-lg");
    row.appendChild(el("i", "coc-k coc-k-" + key));
    row.appendChild(el("b", null, name));
    // Always appended, even when empty: the row is a grid row, and a missing
    // cell would shift every band below it out of alignment.
    row.appendChild(el("small", null, range));
    ui.legend.appendChild(row);
  }
}

function highBar() {
  return Math.max(lastThreshold == null ? 0 : lastThreshold, CFG.aiBand);
}

function bandOf(p, threshold, high) {
  if (p >= high) return { key: "ai", word: "likely AI-generated" };
  if (p >= threshold) return { key: "maybe", word: "uncertain" };
  return { key: "real", word: "likely authentic" };
}

/* ------------------------------------------------------------------ the ring */

function ringBand(key) {
  if (ring) ring.className = "coc-ring" + (key ? " coc-ring-" + key : "");
}

function placeRing() {
  if (!ring) return;
  if (!ringTarget || !ringTarget.isConnected) { ring.style.display = "none"; return; }
  const r = ringTarget.getBoundingClientRect();
  if (r.width < 4 || r.bottom < 0 || r.top > innerHeight) {
    ring.style.display = "none";
    return;
  }
  // "block", not "" -- .coc-ring is display:none in the stylesheet, so clearing
  // the inline value falls back to hidden rather than showing it.
  ring.style.display = "block";
  ring.style.transform = "translate(" + Math.round(r.left) + "px," + Math.round(r.top) + "px)";
  ring.style.width = Math.round(r.width) + "px";
  ring.style.height = Math.round(r.height) + "px";
}

function reposition() {
  if (raf) return;
  raf = requestAnimationFrame(() => { raf = 0; placeRing(); });
}

/* ------------------------------------------------------------------- states */

function clearHold() { clearTimeout(holdTimer); holdTimer = 0; }

function scheduleIdle() {
  clearHold();
  holdTimer = setTimeout(() => {
    // Do not pull a reading away from someone who is still looking at it.
    if (hovering) { scheduleIdle(); return; }
    toIdle();
  }, HOLD_MS);
}

function toIdle() {
  clearHold();
  if (!panel) return;
  panel.dataset.state = "idle";
  delete panel.dataset.band;
  ui.score.textContent = "—";
  ui.word.textContent = "right-click an image to check it";
  ui.thumb.removeAttribute("src");
  ringTarget = null;
  ringBand(null);
  placeRing();
}

function dismiss() {
  clearHold();
  ringTarget = null;
  if (panel) { panel.remove(); panel = null; }
  if (ring) { ring.remove(); ring = null; }
}

function findByUrl(url) {
  return [...document.querySelectorAll("img")]
    .find((i) => (i.currentSrc || i.src) === url);
}

function showScanning(url) {
  ensurePanel();
  clearHold();
  const img = findByUrl(url);
  panel.dataset.state = "busy";
  delete panel.dataset.band;
  ui.score.textContent = "…";
  ui.word.textContent = "checking this image";
  if (img) { ui.thumb.src = img.currentSrc || img.src; ringTarget = img; }
  ringBand("busy");
  placeRing();
}

function showResult(msg) {
  ensurePanel();
  clearHold();

  if (!msg.ok) {
    panel.dataset.state = "error";
    delete panel.dataset.band;
    ui.score.textContent = "—";
    ui.word.textContent = msg.error || "could not check that image";
    ringTarget = null;
    ringBand(null);
    placeRing();
    scheduleIdle();
    return;
  }

  lastThreshold = msg.threshold;
  const band = bandOf(msg.p_ai, msg.threshold, highBar());
  const img = findByUrl(msg.url);

  if (img) { ui.thumb.src = img.currentSrc || img.src; ringTarget = img; }
  panel.dataset.state = "result";
  panel.dataset.band = band.key;
  ui.score.textContent = pct(msg.p_ai, 0);
  ui.word.textContent = band.word + (msg.cached ? " · cached" : "");
  renderLegend();
  ringBand(band.key);
  placeRing();
  scheduleIdle();
}

/* ---------------------------------------------------------------- the wiring */

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "scanning") showScanning(msg.url);
  else if (msg.type === "result") showResult(msg);
});

chrome.storage.onChanged.addListener((changes) => {
  if (changes.aiBand) { CFG.aiBand = changes.aiBand.newValue; renderLegend(); }
});

chrome.runtime.sendMessage({ type: "settings" }, (s) => {
  if (chrome.runtime.lastError || !s) return;
  CFG.aiBand = s.aiBand;
  // The panel is not built until the first scan. Until you ask a question,
  // this extension puts nothing on the page at all.
  addEventListener("scroll", reposition, { passive: true });
  addEventListener("resize", reposition, { passive: true });
});
