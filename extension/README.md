# Chain of Custody — Chrome extension (experimental)

Scores images on any page using the detector from this repository. Two modes,
and the difference between them is the whole design.

**On demand.** Right-click an image → *Check this image for AI*. One question
asked, one answer given, judged at the model's own calibrated threshold. This
is the mode we trust.

**Ambient radar.** Off by default. Every image above a size threshold gets
scored as it enters the viewport, with a badge in its corner. Useful, and the
one to be careful with — see below.

## Install

1. Start the detector and note its URL:
   `powershell -ExecutionPolicy Bypass -File ..\serve_demo.ps1`
2. Chrome → `chrome://extensions` → enable **Developer mode** →
   **Load unpacked** → select this `extension/` folder.
3. Click the extension icon and paste the detector URL into **Detector URL**.
   The status line turns cyan and reports the live model when it connects.

## Why two thresholds

The model's calibrated operating point is **0.51**, chosen to balance the two
error types. At that point its clean false-positive rate is **6.7%**.

That is a fine trade when you asked about one image. It is a bad trade across a
feed: scroll past a hundred genuine photographs and roughly **seven of them get
flagged as AI**. Those are your friends' pictures, and nobody asked. A tool that
is wrong seven times in a scroll gets uninstalled, and deserves to be.

So ambient mode holds images to a much higher bar — **0.90** by default,
adjustable in the popup. It speaks less often and is right more often when it
does. On-demand scans still use the calibrated threshold, because there you
asked a direct question and deserve the model's actual opinion.

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
content.js   finds images, watches the viewport, draws badges in a fixed
             overlay layer so the host page's layout is never touched
background.js does all networking: fetches image bytes (host permissions
             sidestep the page's CORS), posts to /predict, caches by URL
popup.js     detector URL, ambient toggle, thresholds, live health readout
```
