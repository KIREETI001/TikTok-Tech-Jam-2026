# webapp — local demo

Upload an image, get a verdict, and see it **re-checked under the challenge's
redistribution transforms** (JPEG q30, blur σ2, noise σ0.1, resize 0.25×).
That strip is the point: the brief scores robustness, so the demo shows it.

## Run

```bash
CHECKPOINT=runs/iter4/best.pt python webapp/server.py
# open http://localhost:8000
```

Env vars: `CHECKPOINT` (default `runs/iter4/best.pt`), `PORT` (8000),
`DEVICE` (`auto` → XPU/CUDA/CPU). Point `CHECKPOINT` at the hybrid checkpoint
once it's baked.

## What it is

- `server.py` — stdlib `http.server`, no framework. Loads the checkpoint once,
  reuses the pipeline's exact preprocessing (`build_eval_transform`),
  probability head (`_probabilities`), and calibrated threshold (from the
  checkpoint metadata). `/predict?condition=<name>` applies one of the 15
  evaluation transforms before inference.
- `index.html` — one self-contained page, no build step, no external requests,
  theme-aware.

Binds `127.0.0.1` only — a local tool, not a deployment (the brief puts
production deployment out of scope).
