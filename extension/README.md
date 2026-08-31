# Chain of Custody — Chrome extension (experimental)

Checks a single image, on request, using the detector from this repository.

Right-click an image → **Check this image for AI**. One small panel appears in
the corner with that image's thumbnail, its score, and which band the score
falls in. The image itself is ringed on the page, so there is never a question
about which one the reading refers to. After three seconds the reading clears to
an em dash and waits for the next question.

Nothing on a page is scored, marked, or annotated until you ask.

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

The reading expires because an answer that stays forever becomes furniture. You
stop reading it, and worse, you lose track of which image it belonged to.
Expiring it means a number on screen is always about the thing you just asked
about. The countdown pauses while the pointer is over the panel, because a
reading should not vanish out from under someone still reading it.

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
  ~146 ms/image on CPU, so this is realistic — is the obvious next step and is
  not implemented here.
- `<all_urls>` host permission is requested so the background worker can fetch
  image bytes that a content script cannot, because of the page's own CORS and
  CSP. That is a broad permission; read `background.js`, it is ~100 lines.
- Automated processing of another platform's content may sit awkwardly with
  its terms of service. This reads one image you already have on screen, on
  request, which is the mildest version of that — but it is your call.

## How it is wired

```
background.js  the only thing that touches the network. Owns the right-click
               menu item; fetches the image bytes (host permissions sidestep
               the page's CORS and CSP, which a content script cannot), posts
               to /predict, caches by URL, and sends the page two messages:
               "scanning" immediately, then "result"
content.js     draws the panel and the ring in one fixed overlay layer
               appended to documentElement, so the host page's own tree and
               layout are never touched. Assigns the score to a band, holds
               the reading for 3s, then falls back to idle
popup.js       detector URL, AI band boundary, live health readout
```

## Verified in a browser

The extension is exercised end to end under Playwright with the real detector
running — 17 assertions covering: nothing injected before the first request;
the pending state; the panel naming its subject by thumbnail and ring; the ring
tracking the right image and taking the band colour; all three bands landing on
the right scores; the legend's numbers; a second scan replacing the first rather
than adding to it; a single overlay layer; hover pausing the countdown; expiry
to an em dash; the ring clearing with it; re-scan after idle; the close button;
the panel returning on the next scan; and a clean console.

Two real bugs were caught this way and are fixed: `ensureLayer` tested
`document.body.contains` on a node parented to `documentElement`, so it appended
a fresh overlay layer on every call; and `placeRing` set `style.display = ""` on
an element the stylesheet defaults to `display: none`, so the ring never
appeared at all.
