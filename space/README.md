---
title: Robust AI-Generated Image Detector
emoji: 🕵️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# Robust AI-Generated Image Detector — live demo

TikTok TechJam 2026 · PS5. FastAPI wrapper around the iteration-7 detector
(Community Forensics ViT-S/16 + a frozen CLIP ViT-B/16 semantic branch,
~108M params). CPU inference, model loads once at container start.

Organiser Final Score **0.9326** on 6 unseen pixel-diffusion generators.

## Endpoints

| method | path | body | returns |
|---|---|---|---|
| `GET`  | `/`              | — | the single-page UI |
| `GET`  | `/health`        | — | checkpoint, device, params, threshold |
| `POST` | `/predict`       | `multipart/form-data` `file` | `{p_ai, p_authentic, threshold}` |
| `POST` | `/predict/robust`| `file` | `p_ai` under each of the 15 evaluation conditions |
| `POST` | `/predict/batch` | repeated `files` | sorted P(AI) + verdict table for many images |
| `GET`  | `/docs`          | — | interactive OpenAPI page |

`p_ai >= threshold` (0.215, calibrated on withheld generators) is the hard
label; the score itself is threshold-free.

## Deploy

Authenticate once (`hf auth login`, or set `HF_TOKEN` to a write token),
then from the repo root:

```bash
bash space/deploy.sh <your-hf-username>/ttj-aigc-detector
```

Assembles `space/build/` (this folder + `detector/` + `webapp/` +
`runs/iter6/best.pt`) and uploads it via the `huggingface_hub` API — no git
or git-lfs. First build downloads and bakes in the CLIP-B weights (~350 MB),
so it takes a few minutes; after that the container starts in seconds.

## Config (Space → Settings → Variables and secrets)

- `ALLOW_ORIGINS` — CORS allowlist, comma-separated. Defaults to `*`.
- `DETECTOR_DEVICE` — `cpu` (default on the free tier). Set `cuda` on a GPU Space.
