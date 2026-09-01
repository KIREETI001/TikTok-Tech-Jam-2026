/* Chain of Custody — one reading at a time.
 *
 * The interaction is deliberately narrow. You ask about one thing -- the main
 * element on the page, or a specific image you right-click -- and one panel in
 * the corner reports on it. Nothing else on the page is scored, marked, or
 * annotated.
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
const IDLE_HINT = "right-click → check the main image";

let CFG = { aiBand: 0.9, showButton: true };
let layer = null;
let panel = null;
let ring = null;
const ui = {};
let holdTimer = 0;
let hovering = false;
let movedAt = 0;      // when the pointer last moved *over the panel*
let ringTarget = null;
let raf = 0;
let lastThreshold = null;
let lastPick = null;   // the element pickMain chose, for the ring and the thumbnail

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

  // The panel at rest is the button. Making it a separate floating control
  // would put two pieces of our UI on someone else's page to do one job; this
  // way the resting state and the affordance are the same object, and the
  // reading simply grows out of the thing you pressed.
  const idle = el("div", "coc-idle");
  ui.go = el("button", "coc-go");
  ui.go.type = "button";
  ui.go.appendChild(el("span", "coc-dot"));
  ui.go.appendChild(el("span", "coc-golabel", "Check main image"));
  ui.go.title = "Score the main image on this page (Alt+Shift+A)";
  ui.go.addEventListener("click", () => {
    ui.go.disabled = true;
    chrome.runtime.sendMessage({ type: "checkMain" });
  });
  const hide = el("button", "coc-close", "×");
  hide.type = "button";
  hide.title = "Hide until the next check";
  hide.setAttribute("aria-label", "Hide the Chain of Custody panel");
  hide.addEventListener("click", dismiss);
  idle.appendChild(ui.go);
  idle.appendChild(hide);

  const body = el("div", "coc-body");
  ui.thumb = el("img", "coc-thumb");
  ui.thumb.alt = "";
  const read = el("div", "coc-read");
  ui.score = el("div", "coc-score", "—");
  ui.word = el("div", "coc-word", IDLE_HINT);
  read.appendChild(ui.score);
  read.appendChild(ui.word);
  body.appendChild(ui.thumb);
  body.appendChild(read);

  ui.legend = el("div", "coc-legend");

  panel.appendChild(idle);
  panel.appendChild(head);
  panel.appendChild(body);
  panel.appendChild(ui.legend);

  panel.addEventListener("mouseenter", () => { hovering = true; });
  panel.addEventListener("mouseleave", () => { hovering = false; });
  panel.addEventListener("mousemove", () => { movedAt = Date.now(); });

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
    // Do not pull a reading away from someone who is still looking at it -- but
    // "the pointer is over the panel" is not that test. Clicking the button
    // leaves the cursor parked right here, so hover alone would mean a reading
    // started by the button never expires at all. Require movement *since the
    // reading appeared*, which is what actually distinguishes reading it from
    // having just pressed it.
    if (hovering && movedAt && Date.now() - movedAt < HOLD_MS) { scheduleIdle(); return; }
    toIdle();
  }, HOLD_MS);
}

function toIdle() {
  clearHold();
  if (!panel) return;
  panel.dataset.state = "idle";
  delete panel.dataset.band;
  ui.score.textContent = "—";
  ui.word.textContent = IDLE_HINT;
  ui.thumb.removeAttribute("src");
  if (ui.go) ui.go.disabled = false;
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

/* ------------------------------------------------------- the main element
 *
 * On a feed, what you can right-click and what you are actually looking at are
 * different things. TikTok's player is a <video>, which offers no image menu;
 * the only <img> elements in reach are the sidebar recommendations and the
 * avatars -- exactly the noise.
 *
 * So rank candidates by visible area weighted toward the centre of the
 * viewport. Area alone picks a full-bleed background; centrality alone picks a
 * tiny centred icon. Together they pick the thing the page is built around,
 * which is what a reader means by "the main image".
 *
 * The ring drawn around the winner is not decoration here -- it is how you
 * check that the heuristic agreed with you before you trust the number.
 */
function pickMain() {
  const vw = innerWidth, vh = innerHeight;
  const cx = vw / 2, cy = vh / 2;
  let best = null, bestScore = 0;

  for (const el of document.querySelectorAll("img, video")) {
    const r = el.getBoundingClientRect();
    const visW = Math.max(0, Math.min(r.right, vw) - Math.max(r.left, 0));
    const visH = Math.max(0, Math.min(r.bottom, vh) - Math.max(r.top, 0));
    if (visW < 120 || visH < 120) continue;        // sidebar thumbs, avatars, icons

    const st = getComputedStyle(el);
    if (st.visibility === "hidden" || st.display === "none" || +st.opacity === 0) continue;
    if (el.tagName === "IMG" && !el.complete) continue;

    // Distance from the viewport centre, normalised so it is resolution-independent.
    const d = Math.hypot((r.left + r.width / 2 - cx) / vw, (r.top + r.height / 2 - cy) / vh);
    const s = visW * visH * (1 - Math.min(d, 1) * 0.7);
    if (s > bestScore) { bestScore = s; best = el; }
  }
  return best;
}

function describePick(el) {
  const r = el.getBoundingClientRect();
  return {
    ok: true,
    kind: el.tagName === "VIDEO" ? "video" : "img",
    url: el.tagName === "IMG" ? (el.currentSrc || el.src) : null,
    // Clamped to the viewport: a capture only contains what is on screen, so a
    // rect running off the edge would crop the wrong region.
    rect: {
      x: Math.max(0, r.left),
      y: Math.max(0, r.top),
      w: Math.min(r.width, innerWidth - Math.max(0, r.left)),
      h: Math.min(r.height, innerHeight - Math.max(0, r.top)),
    },
    // The viewport in CSS pixels. The worker divides the capture's own width by
    // this to get the scale, rather than trusting devicePixelRatio: under OS
    // display scaling the capture can come back at 2x while the page still
    // reports a ratio of 1, and cropping by the wrong factor silently reads a
    // completely different part of the screen.
    view: { w: innerWidth, h: innerHeight },
  };
}

function showScanning(msg) {
  ensurePanel();
  clearHold();
  // Either the page told the worker which element it picked, or the worker is
  // reporting a URL the reader right-clicked.
  const el = msg.pick ? lastPick : findByUrl(msg.url);
  movedAt = 0;
  panel.dataset.state = "busy";
  delete panel.dataset.band;
  ui.score.textContent = "…";
  ui.word.textContent = "checking this image";
  if (el) {
    if (el.tagName === "IMG") ui.thumb.src = el.currentSrc || el.src;
    else ui.thumb.removeAttribute("src");
    ringTarget = el;
  }
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
  const el = findByUrl(msg.url) || lastPick;

  // msg.thumb is the exact crop that was scored, so it beats re-deriving a
  // preview from the element -- especially for a video, where there is no
  // still to point an <img> at.
  if (msg.thumb) ui.thumb.src = msg.thumb;
  else if (el && el.tagName === "IMG") ui.thumb.src = el.currentSrc || el.src;
  if (el) ringTarget = el;

  movedAt = 0;
  panel.dataset.state = "result";
  panel.dataset.band = band.key;
  ui.score.textContent = pct(msg.p_ai, 0);
  // A capture has already been scaled by the browser and squeezed through the
  // site's video codec. This model's headline finding is that resizing
  // destroys the artifact, so that reading is weaker evidence and is labelled
  // as such rather than presented at the same confidence as original bytes.
  ui.word.textContent = band.word +
    (msg.viaCapture ? " · from screen capture" : "") +
    (msg.cached ? " · cached" : "");
  renderLegend();
  ringBand(band.key);
  placeRing();
  scheduleIdle();
}

/* ---------------------------------------------------------------- the wiring */

chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
  if (msg.type === "overlay") {
    if (layer) layer.style.display = msg.show ? "" : "none";
    // Two frames, so the change is actually painted before the screenshot is
    // taken -- otherwise the capture still contains the overlay we just hid.
    requestAnimationFrame(() => requestAnimationFrame(() => reply({ ok: true })));
    return true;
  }
  if (msg.type === "findMain") {
    const el = pickMain();
    lastPick = el;
    reply(el ? describePick(el)
             : { ok: false, error: "nothing large enough to check on this page" });
    return true;                                    // keep the channel open
  }
  if (msg.type === "scanning") showScanning(msg);
  else if (msg.type === "result") showResult(msg);
  return false;
});

chrome.storage.onChanged.addListener((changes) => {
  if (changes.aiBand) { CFG.aiBand = changes.aiBand.newValue; renderLegend(); }
  if (changes.showButton) {
    CFG.showButton = changes.showButton.newValue !== false;
    if (CFG.showButton) ensurePanel(); else dismiss();
  }
});

chrome.runtime.sendMessage({ type: "settings" }, (s) => {
  if (chrome.runtime.lastError || !s) return;
  CFG.aiBand = s.aiBand;
  CFG.showButton = s.showButton !== false;
  // With the button on, the panel is present from the start -- that is the
  // whole point of it. With the button off the old rule holds: nothing is put
  // on the page at all until you ask.
  if (CFG.showButton) ensurePanel();
  addEventListener("scroll", reposition, { passive: true });
  addEventListener("resize", reposition, { passive: true });
});
