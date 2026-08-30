# webapp — local demo

Upload an image, get a calibrated P(AI), and see it re-scored under **every
one of the brief's 15 evaluation conditions** — the half of the score a
clean-only demo can't show.

## Run

```bash
pip install fastapi uvicorn
DETECTOR_CHECKPOINT=runs/iter7/best.pt uvicorn webapp.server:app --port 8000
# open http://localhost:8000
```

Env: `DETECTOR_CHECKPOINT` (default `runs/latest/best.pt`), `DETECTOR_DEVICE`
(`auto` → XPU/CUDA/CPU), `ALLOW_ORIGINS`.

## Endpoints

| route | returns |
|---|---|
| `GET /` | the single-page UI (single image · robustness grid · batch) |
| `GET /health` | checkpoint path, device, model type, calibrated threshold |
| `POST /predict` | `{p_ai, p_authentic, threshold}` for one image |
| `POST /predict/robust` | `p_ai` under each of the 15 conditions + the spread |
| `POST /predict/batch` | many images (or a folder) → a P(AI) + verdict table, sorted, with a CSV download. Batched inference, ≤200 files/request. Same result as `python pipeline.py predict --input <dir>`. |

`server.py` reuses `detector.evaluation._probabilities` and
`detector.transforms.build_eval_transform` / `apply_condition` — the same code
the offline evaluation runs, so the page and the metrics table can't drift.
One file, no auth, no database (slide 15: "over-engineering the UI" is a
rabbit hole; slide 13: complexity that threatens the demo isn't worth it).
