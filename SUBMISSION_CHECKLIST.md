# Submission checklist (Devpost)

Judging weights: Technical Execution 35% · Innovation & Insight 20% ·
Impact & Relevance 20% · Feasibility 15% · Presentation 10%.

**Submission model: `runs/iter7/best.pt`** — ViT-S + frozen CLIP-B/16,
organiser Final Score **0.9326** (clean 0.9548 / robust 0.9105), threshold
0.215, 107.7M inference params.

## Required deliverables

- [x] **Written project description** — `DEVPOST.md` (paste into Devpost)
- [x] **Public code repo + README** — `README.md`, repo pushed to
      `github.com/KIREETI001/TikTok-Tech-Jam-2026` (branch `kireeti-pipeline`)
- [ ] **Demo video** (public YouTube, 2–4 min) — script in `DEMO_SCRIPT.md`;
      run `webapp/` and screen-record (numbers already final)
- [x] **Robustness evaluation summary (table)** — `ERROR_ANALYSIS.md` §
      "15-condition matrix" + `TRAINING_REPORT.docx` §7.3
- [x] **Error-analysis note (FP/FN examples)** — `ERROR_ANALYSIS.md` +
      `errors_FP.png` / `errors_FN.png` (6 studio-photo FPs, 6 ADM/DDPM FNs)

## Rules compliance (violation = DQ)

- [x] Model < 2B params — 107.7M (22M ViT + 86M frozen CLIP-B/16)
- [x] Public pretrained backbones — `OwensLab/commfor-model-224`,
      `openai/clip-vit-base-patch16`
- [x] Custom code MIT/Apache — repo license
- [x] Public/licensed data only — SID_Set, DRAGON, Community-Forensics-Small
- [x] No test-label training — `assert_not_eval_only()` guards it in code
- [x] Augmentation scripts included — `detector/transforms.py`
- [ ] Open-source model weights → **upload `runs/iter7/best.pt` to HuggingFace**

## To finish (needs the user)

1. [ ] **HuggingFace upload** — `hf auth login` with a write token, then
       `python hf_upload/upload.py --repo kireeti26/ttj-aigc-detector`.
       Package ready: `hf_upload/{model_config.json, README.md, upload.py}`.
       Then add the HF link to `README.md` and `DEVPOST.md`.
2. [ ] **Demo video** — `DETECTOR_CHECKPOINT=runs/iter7/best.pt uvicorn
       webapp.server:app --port 8000`, follow `DEMO_SCRIPT.md`, upload to
       YouTube (unlisted → public).
3. [ ] **Devpost form** — paste `DEVPOST.md`; add repo URL + video URL.
4. [ ] Optional: merge `kireeti-pipeline` → `main`, or make it the default
       branch, so the repo root shows the current work.

## Done this sprint

- iter7: frozen CLIP-B semantic branch (fusion head on precomputed features)
  + feature-jitter augmentation → 0.9129 → 0.9326
- Deep research synthesis (`RESEARCH_SYNTHESIS.md`): generator-fingerprint
  taxonomy, real-image markers, dataset-shortcut analysis
- Negative results recorded: NPR frequency branch, SAFE DWT branch, CLIP-L
- Shortcut probe: training corpus is not format-poisoned
- `TRAINING_REPORT.docx` regenerated for iter7

## Not done (documented as limitations)

- CLIP ViT-L/14 — hangs on the Intel Arc iGPU (would run on NVIDIA)
- Composite degradations (screenshot, social-upload) in the table
- Camera-RAW FPR diagnostic slice
- The ≤3% FPR/FNR target — unreached by any published method on unseen
  generators; the deck sets no hard threshold
