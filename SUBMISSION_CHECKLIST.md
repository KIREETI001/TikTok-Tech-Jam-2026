# Submission checklist (Devpost)

Per briefing deck slide 16. Judging weights: Technical Execution 35% ·
Innovation & Insight 20% · Impact & Relevance 20% · Feasibility 15% ·
Presentation 10%.

## Required deliverables

- [ ] **Written project description** — `DEVPOST.md` (paste into Devpost)
- [ ] **Public code repo + clear README** — `README.md` ✅ drafted
- [ ] **Demo video** (public YouTube, 2–4 min) — script in `DEMO_SCRIPT.md`
- [ ] **Robustness evaluation summary (table)** — `ERROR_ANALYSIS.md` +
      `TRAINING_REPORT.docx` §7.3 (regen on final checkpoint via `finalize.sh`)
- [ ] **Error-analysis note (FP/FN examples)** — `ERROR_ANALYSIS.md` montages
      (`errors_FP.png` / `errors_FN.png`)

## Rules compliance (violation = DQ)

- [x] Model < 2B params — 22M (ViT) + 86M (frozen CLIP-B) = ~108M inference
- [x] Public pretrained backbones — `OwensLab/commfor-model-224`,
      `openai/clip-vit-base-patch16` (both public)
- [x] Custom code MIT/Apache — repo license
- [x] Public/licensed data only — SID_Set, DRAGON, Community-Forensics-Small
- [x] No test-label training — `assert_not_eval_only()` guards it in code
- [x] Augmentation scripts included — `detector/transforms.py`
- [ ] Winning-team open-source: training pipeline, hyperparameters, eval code,
      **model weights** → upload best `.pt` to HuggingFace

## Package steps (run in order on the final checkpoint)

1. [ ] `bash finalize.sh runs/<winner>/best.pt final`
       → 15-condition metrics, per-generator, montages, `ERROR_ANALYSIS.md`
2. [ ] `python -m detector.calibrate runs/<winner>/best.pt --genval ... --calval ... --rule minmax_fpfn --apply`
       (already applied by run_fast.sh; re-confirm the value)
3. [ ] `python scratchpad/make_report.py` → refresh `TRAINING_REPORT.docx`
4. [ ] Fill `⟨iter6⟩` / `⟨final⟩` placeholders in `README.md` + `DEVPOST.md`
5. [ ] `python pipeline.py predict --input demo_images/ --output demo_out.json --checkpoint runs/<winner>/best.pt`
       → sanity-check predictions, keep for the video
6. [ ] Upload `best.pt` + `model_config.json` to HuggingFace; link in README
7. [ ] `git add -A && git commit` on a branch; push; open PR or publish repo
8. [ ] Record demo video, upload to YouTube (unlisted → public)
9. [ ] Devpost form: paste `DEVPOST.md`, add repo + video links

## Nice-to-have (only if time)

- [ ] Composite degradation conditions (screenshot, social-upload) in the table
- [ ] TTA (`--tta` 5-crop) folded into `evaluation.py`
- [ ] CLIP ViT-L/14 via precomputed embeddings (bigger lever, needs the cache)
- [ ] Camera-RAW (RAISE-1k) FPR diagnostic slice
