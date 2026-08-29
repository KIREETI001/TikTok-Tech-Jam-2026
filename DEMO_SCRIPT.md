# Demo video script (2–4 min)

Target: 3:00. Screen recording + voiceover. Numbers are iter4; swap to the
final checkpoint's before recording.

---

### 0:00–0:25 — The problem, stated as a number

> "A detector that scores 99% on its own test set and 65% on next month's
> generator ships false confidence. So we optimised for one number only:
> ROC-AUC on generator families the model has never seen, after realistic
> redistribution — JPEG, blur, resize, noise."

Show: the briefing deck's Final Score formula, then our results table.

| Benchmark | Final Score |
|---|---|
| Organiser composition (6 unseen pixel-diffusion families) | **⟨final⟩** |
| DRAGON — 8 unseen latent-diffusion generators | 0.9959 |

---

### 0:25–1:10 — Live inference

Terminal:
```
python pipeline.py predict --input demo_images/ --output out.json --checkpoint runs/<final>/best.pt
```
`demo_images/` has 6: 2 real photos, 2 latent-diffusion fakes, 1 ADM (pixel
diffusion, hard), 1 real camera-RAW. Show `out.json` + the `.scores.csv`
probabilities. Then re-run on the same folder after `mogrify -quality 30`
JPEG compression — show the predictions barely move.

> "Same images, JPEG quality 30 — the kind of thing a messaging app does.
> Predictions hold because the training pipeline applied every one of these
> transforms during training. The augmentation *is* the robustness
> contribution."

---

### 1:10–2:00 — How it works

Show the architecture diagram (README).

> "A frozen Community Forensics ViT — 22M parameters, a purpose-built
> detector — gives the base decision. A frozen CLIP vision transformer adds
> a semantic correction: we measured that a *fine-tuned* backbone loses 0.2
> AUC on unseen generators while a *frozen* one loses half that. And a SAFE
> wavelet branch reads the high-frequency sub-band, where diffusion
> upsampling artifacts live.
>
> Each branch is a zero-initialised residual — the model starts identical to
> the proven ViT and only adds corrections. No per-branch loss: that's what
> made a parallel team's frequency branch memorise training generators and
> invert on new ones."

---

### 2:00–2:35 — Robustness table + error analysis

Show `ERROR_ANALYSIS.md`: the 15-condition table for the organiser set, then
the FP and FN montages.

> "Sensor noise is our weakest condition — additive noise directly
> overwrites the high-frequency evidence. Our hardest generator is ADM, 2021
> ImageNet pixel-diffusion, the furthest from anything in training. The
> false positives are mostly high-contrast studio photos; the false
> negatives are distilled fast samplers. We report these, not hide them."

---

### 2:35–3:00 — Trade-offs & close

> "Every choice traded something. Heavy augmentation cost clean accuracy —
> worth it. We hold six generator families fully out of training — a
> detector tuned on them scores higher here and breaks on the next one. And
> we kept the model small enough to run on a laptop CPU, because a 1%
> ensemble win that breaks the demo isn't a win.
>
> There's no silver bullet. This is a detector that degrades honestly."

Show: repo URL, HuggingFace model link.
