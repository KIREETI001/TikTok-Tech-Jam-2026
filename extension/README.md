# Chain of Custody — Chrome extension (experimental)

Scores images on any page using the detector from this repository. Two modes,
and the difference between them is the whole design.

**On demand — the default.** Right-click an image → *Check this image for AI*.
Nothing on the page is scored until you ask. One question, one answer, in one of
the three bands below. Click a badge to dismiss it.

**Ambient radar.** Off by default. Every image above a size threshold gets
scored as it enters the viewport — but a badge is drawn *only* when the score
reaches the AI band. The other two bands stay silent. A badge on every image is
not a radar, it is wallpaper: it spends your attention on every photo to tell
you what you already assumed about almost all of them.

## The three bands

| Band | Score | Badge |
| --- | --- | --- |
| Likely authentic | below **0.51** | solid cyan |
| Uncertain | **0.51 – 0.90** | dashed amber |
| Likely AI-generated | **0.90** and above | solid orange |

The middle band is not decoration. Without it a 0.80 image was painted the same
colour as a 0.07 image, which reads as *"we checked, this is fine"* when what
the model actually said was *"probably generated, but not past the bar I will
accuse someone over"*. Amber gets a dashed border as well as its own hue,
because at 11px adjacent hues alone are not a reliable signal.

Every badge carries the whole legend in its tooltip — it floats over somebody
else's page, so if the bands are not explained there they are not explained
anywhere you are looking.

## Install

1. Start the detector and note its URL:
   `powershell -ExecutionPolicy Bypass -File ..\serve_demo.ps1`
2. Chrome → `chrome://extensions` → enable **Developer mode** →
   **Load unpacked** → select this `extension/` folder.
3. Click the extension icon and paste the detector URL into **Detector URL**.
   The status line turns cyan and reports the live model when it connects.

## Why the AI band sits so far above the threshold

The model's calibrated operating point is **0.51**, chosen to balance the two
error types. At that point its clean false-positive rate is **6.7%**.

That is a fine trade when you asked about one image. It is a bad trade across a
feed: scroll past a hundred genuine photographs and roughly **seven of them get
flagged as AI**. Those are your friends' pictures, and nobody asked. A tool that
is wrong seven times in a scroll gets uninstalled, and deserves to be.

So the AI band starts at **0.90** by default, adjustable in the popup, and that
same number is the bar ambient mode must clear before it says anything at all.
The rule is simply *ambient speaks only when it would say AI* — it is right far
more often when it does speak, and silent the rest of the time.

Scores between 0.51 and 0.90 are still shown for a scan you asked for, in the
amber uncertain band. That is the honest answer: the model is leaning toward
generated, and is not confident enough to make an accusation.

## Known limits

- **Domain shift is real.** The detector was trained and measured on
  research corpora. Instagram re-compresses aggressively, and our own numbers
  show ROC-AUC falling from 0.960 clean to **0.830 at JPEG q30**. Expect worse
  accuracy in a feed than the headline figure suggests.
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
  CSP. That is a broad permission; read `background.js`, it is 100 lines.
- Automated processing of another platform's content may sit awkwardly with
  its terms of service. This reads images already rendered in your own browser,
  for your own use, which is the mildest version of that — but it is your call.

## How it is wired

```
content.js   finds images, watches the viewport, assigns each score to a band,
             draws badges in a fixed overlay layer so the host page's layout
             is never touched
background.js does all networking: fetches image bytes (host permissions
             sidestep the page's CORS), posts to /predict, caches by URL,
             owns the right-click menu item
popup.js     detector URL, ambient toggle, band boundary, live health readout
```

## Verified in a browser

`content.js` is exercised end to end under Playwright with the extension loaded
and a live detector — 13 assertions covering: the default mode painting nothing,
the right-click pending badge, all three bands landing on the right scores,
tooltip legends, click-to-dismiss, ambient flagging only the AI band, a single
overlay layer across scrolling, and a clean console.
