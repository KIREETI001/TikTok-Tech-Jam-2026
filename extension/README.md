# PixelProof — Chrome extension (experimental)

Checks one thing at a time, on request, using the detector from this repository.

Click **Check main image** in the bottom-right corner, press **Alt+Shift+A**, or
right-click anywhere → **Check the main image on this page**. The panel
expands with a thumbnail of what was checked, its score, and which band the
score falls in. The element itself is
ringed on the page, so there is never a question about what the reading refers
to. After three seconds the reading clears to an em dash and waits for the next
question.

Right-click *directly on an image* and you also get **Check this image for AI**,
which checks that specific image instead.

Nothing on a page is scored, marked, or annotated until you ask.

## What counts as "the main image"

On a feed, what you can right-click and what you are actually looking at are
different things. TikTok's player is a `<video>`, which offers no image context
menu at all; the only `<img>` elements in reach are the sidebar recommendations
and the avatars — exactly the noise.

So candidates are ranked by **visible area weighted toward the centre of the
viewport**. Area alone picks a full-bleed background; centrality alone picks a
tiny centred icon. Together they pick the thing the page is built around.
Anything under 120×120 visible pixels is skipped before ranking begins, which
removes thumbnails, avatars and icons outright.

The ring is how you audit that. If the heuristic picked the wrong thing you can
see it immediately and fall back to right-clicking the image you meant.

## Video

Most of a feed is video, so it has to work — and it works differently.

A `<video>` has no image URL to fetch, and the obvious fix (draw the frame to a
canvas) is blocked outright: drawing cross-origin video taints the canvas, so
reading it back throws. Instead the extension takes a **browser-level screenshot
of the visible tab** and crops it to the player's rectangle, which same-origin
rules never touch.

That reading is honestly weaker, and the panel says so — it is labelled
**"from screen capture"**. The frame has already been scaled to the player's
size by the browser and already been through the site's video codec, and this
detector's headline finding is that resizing destroys the artifact. Treat a
capture as a hint; treat a reading taken from an image's original bytes as
evidence.

The crop scale is derived from the capture's own width divided by the viewport
width in CSS pixels — **not** from `devicePixelRatio`. Under OS display scaling
the screenshot comes back at 2× while the page still reports a ratio of 1, and
cropping by the wrong factor does not fail: it silently reads a different part
of the screen and returns a perfectly plausible number about the wrong pixels.
That bug existed briefly, and is why the test suite now asserts on the captured
pixels rather than only on the score.

## The three bands

| Band | Score | Colour |
| --- | --- | --- |
| Likely authentic | below **0.51** | cyan |
| Uncertain | **0.51 – 0.90** | amber, dashed |
| Likely AI-generated | **0.90** and above | orange |

The middle band is not decoration. Without it a 0.80 image was painted the same
colour as a 0.07 image, which reads as *"we checked, this is fine"* when what
the model actually said was *"probably generated, but not past the bar I will
accuse someone over"*.

The panel carries the full legend under every reading, with swatches that reuse
the reading's own colours. A legend that drifts from the thing it explains is
worse than no legend.

## Install

1. Start the detector and note its URL:
   `powershell -ExecutionPolicy Bypass -File ..\serve_demo.ps1`
2. Chrome → `chrome://extensions` → enable **Developer mode** →
   **Load unpacked** → select this `extension/` folder.
3. Click the extension icon and paste the detector URL into **Detector URL**.
   The status line turns cyan and reports the live model when it connects.

After editing any file here, press **reload** on the extension's card in
`chrome://extensions`. Chrome does not pick up changes to an unpacked extension
on its own, and a stale copy is indistinguishable from a bug.

## Why one reading at a time

An earlier version drew a badge on every image it scored. That was the wrong
shape twice over. It scaled with the page rather than with your attention, so a
feed of thirty photos became thirty verdicts nobody asked for; and once several
badges were on screen there was no way to tell which one you had actually asked
about. A single panel cannot have either problem — there is only ever one
reading, and it names its own subject with a thumbnail and a ring.

At rest the panel collapses to a single row and *is* the button, so the page
never carries a button and a panel to do one job. Turn it off in the popup and
the original rule holds: nothing is put on a page until you ask.

The reading expires because an answer that stays forever becomes furniture. You
stop reading it, and worse, you lose track of which image it belonged to.
Expiring it means a number on screen is always about the thing you just asked
about. The countdown pauses while the pointer *moves* over the panel → not
merely while it rests there, because clicking the button leaves the cursor
parked on it, and hover alone would mean a reading started by the button never
expired at all.

An ambient mode that scored every image as you scrolled was removed for the same
reason. It was the source of the noise, and it cannot coexist with one reading
at a time. `git log extension/` has it if it is ever wanted back.

## Why the AI band sits so far above the threshold

The model's calibrated operating point is **0.51**, chosen to balance the two
error types. At that point its clean false-positive rate is **6.7%** — roughly
seven wrong flags per hundred genuine photographs.

Calling an image generated is an accusation, so the AI band starts at **0.90**
by default, adjustable in the popup. Scores between the two are reported
honestly as *uncertain* rather than rounded into a verdict: the model is leaning
toward generated and is not confident enough to say so.

## Known limits

- **Domain shift is real.** The detector was trained and measured on
  research corpora. Instagram re-compresses aggressively, and our own numbers
  show ROC-AUC falling from 0.960 clean to **0.830 at JPEG q30**. Expect worse
  accuracy on a social feed than the headline figure suggests.
- **Edited is not generated.** The model answers *"was this image generated?"*,
  not *"was this image retouched?"*. Filters, portrait mode and skin smoothing
  push an authentic photo toward the statistics of a generated one. Our
  top false positives are already "high-contrast, low-noise studio shots" —
  which describes a lot of what people post.
- **It sends images to your detector.** Every scored image is fetched by the
  extension and posted to the URL you configured. Point it at your own server.
  Running the model in the browser instead — the model is 21.7M parameters and
  ~0.3 s/image on a laptop CPU, so this is realistic — is the obvious next step
  and is not implemented here.
- `<all_urls>` host permission is requested so the background worker can fetch
  image bytes that a content script cannot, because of the page's own CORS and
  CSP. That is a broad permission; read `background.js`, it is ~100 lines.
- Automated processing of another platform's content may sit awkwardly with
  its terms of service. This reads one image you already have on screen, on
  request, which is the mildest version of that — but it is your call.

## How it is wired

```
background.js  the only thing that touches the network. Owns both menu items
               and the keyboard command; fetches image bytes (host permissions
               sidestep the page's CORS and CSP, which a content script
               cannot) or crops a captureVisibleTab screenshot for video;
               posts to /predict, caches by URL, and sends the page two
               messages: "scanning" immediately, then "result"
content.js     picks the main element, draws the panel and the ring in one
               fixed overlay layer
               appended to documentElement, so the host page's own tree and
               layout are never touched. Assigns the score to a band, holds
               the reading for 3s, then falls back to idle
popup.js       detector URL, AI band boundary, live health readout
```

## Verified in a browser

Every change here is exercised end to end under Playwright, with the extension
loaded and the real detector running. The fixture is feed-shaped: an
**authentic** photo in the centre, **AI** images in every sidebar slot, so a
mis-pick moves the score in an unmistakable direction rather than a subtle one.

Across those suites the assertions cover the button appearing without a
right-click; the panel's resting size; the picker choosing the centre over
seven decoys; the ring matching the chosen element exactly; all three bands
landing on the right scores; the legend's numbers; a second scan replacing the
first rather than adding to it; a single overlay layer across scrolling; expiry
to an em dash with the cursor parked on the panel; the countdown pausing while
the pointer moves over it; the popup toggle removing and restoring the panel;
and a clean console.

**Six real bugs were caught this way**, each of which parsed cleanly and would
have shipped:

1. `ensureLayer` tested `document.body.contains` on a node parented to
   `documentElement`, so every call appended another overlay layer.
2. `placeRing` set `style.display = ""` on an element the stylesheet defaults to
   `display: none`, so the ring never appeared at all.
3. The capture crop scaled by `devicePixelRatio`, which reports 1 under OS
   display scaling while the screenshot returns at 2x — so it cropped a
   different part of the screen and returned a plausible score for the wrong
   pixels. The suite now asserts on the captured pixels, not only the score.
4. The capture included our own overlay: the ring sits on the element's edge
   and the panel can overlap the crop.
5. The hover-pause made a reading immortal. Clicking the button leaves the
   cursor on the panel, so the countdown rescheduled forever.
6. The panel was only built inside the callback of a message to the service
   worker, so a failed round-trip left the page with no button and no way to
   ask for one.
