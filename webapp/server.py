"""Inference API for the AI-vs-real image detector.

Two endpoints:

    POST /predict          image in, calibrated P(AI) out
    POST /predict/robust   image in, P(AI) under each of the brief's 15
                           evaluation conditions out

The second exists because the competition scores
``0.5 * AUC_clean + 0.5 * AUC_robust`` -- half the score is behaviour under
post-processing, so a demo that only shows a clean verdict shows half the
system. It reuses ``detector.transforms.apply_condition``, the same code
path the offline evaluation uses, so what the page displays and what the
metrics table reports cannot drift apart.

Deliberately small: one file, no auth, no database, no queue. Slide 15 of
the briefing deck lists "over-engineering the UI" as a rabbit hole and slide
13 warns that complexity threatening the demo is not worth the points.

Run locally:
    uvicorn webapp.server:app --reload --port 8000
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector.evaluation import _probabilities  # noqa: E402
from detector.model import load_checkpoint, resolve_device  # noqa: E402
from detector.transforms import (  # noqa: E402
    CONDITION_GROUPS,
    EVALUATION_CONDITIONS,
    apply_condition,
    build_eval_transform,
)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
CHECKPOINT = os.environ.get("DETECTOR_CHECKPOINT", "runs/latest/best.pt")
DEVICE = os.environ.get("DETECTOR_DEVICE", "auto")
# Comma-separated allowlist. "*" is fine here: no cookies, no auth, and the
# uploaded image is scored in memory and dropped rather than retained.
ALLOW_ORIGINS = [o.strip() for o in os.environ.get("ALLOW_ORIGINS", "*").split(",")]

app = FastAPI(title="AI-vs-real image detector", docs_url="/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_MODEL = None
_META: dict = {}
_DEVICE = None


def _load() -> None:
    """Load the checkpoint once, on first use rather than at import.

    Import-time loading makes the module unusable in any context that only
    wants its constants -- tests, a docs build, ``--help``.
    """
    global _MODEL, _META, _DEVICE
    if _MODEL is not None:
        return
    path = Path(CHECKPOINT)
    if not path.exists():
        raise RuntimeError(
            f"Checkpoint not found: {path}. Set DETECTOR_CHECKPOINT to a trained .pt file."
        )
    _DEVICE = resolve_device(DEVICE)
    _MODEL, _META = load_checkpoint(path, device=_DEVICE)
    _MODEL.eval()


def _read_image(raw: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(raw)) as handle:
            return handle.convert("RGB").copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Not a readable image: {exc}") from exc


@torch.no_grad()
def _score(image: Image.Image, condition: str = "clean") -> float:
    """P(AI) for one image under one evaluation condition."""
    crop_policy = _META.get("crop_policy") or "resize"
    transform = build_eval_transform(condition, crop_policy=crop_policy)
    tensor = transform(image).unsqueeze(0).to(_DEVICE)
    return float(_probabilities(_MODEL, tensor)[0])


async def _upload_bytes(file: UploadFile) -> bytes:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.",
        )
    return raw


@app.get("/health")
def health() -> JSONResponse:
    """Liveness, plus what is actually being served -- a stale or wrong
    checkpoint should be visible without reading logs."""
    try:
        _load()
    except RuntimeError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)
    return JSONResponse(
        {
            "ok": True,
            "checkpoint": str(CHECKPOINT),
            "device": str(_DEVICE),
            "model_type": _META.get("model_type") or "vit",
            "parameters": _META.get("parameter_count"),
            "crop_policy": _META.get("crop_policy") or "resize",
            "threshold": float(_META.get("threshold", 0.5)),
        }
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """Calibrated P(AI) for one image.

    Returns a probability, not a verdict. The task asks for a calibrated
    confidence and the score is threshold-free, so collapsing 0.62 into
    "AI" discards the only part of the output carrying how sure the model
    is. ``threshold`` is returned alongside for callers that need a label.
    """
    _load()
    image = _read_image(await _upload_bytes(file))
    probability = _score(image)
    return JSONResponse(
        {
            "p_ai": probability,
            "p_authentic": 1.0 - probability,
            "threshold": float(_META.get("threshold", 0.5)),
            "width": image.width,
            "height": image.height,
        }
    )


@app.post("/predict/robust")
async def predict_robust(file: UploadFile = File(...)) -> JSONResponse:
    """P(AI) under every condition in the brief's disclosed evaluation grid.

    This is the half of the score a clean-only demo cannot show. Conditions
    and severities come from CONDITION_SPECS, so this tracks the evaluation
    matrix automatically instead of hard-coding a second copy of it.
    """
    _load()
    image = _read_image(await _upload_bytes(file))

    results = []
    for condition in EVALUATION_CONDITIONS:
        degraded = apply_condition(image, condition, seed=0)
        results.append(
            {
                "condition": condition,
                "group": CONDITION_GROUPS.get(condition, "clean"),
                "p_ai": _score(degraded, "clean"),
            }
        )

    clean = next(r["p_ai"] for r in results if r["condition"] == "clean")
    transformed = [r for r in results if r["condition"] != "clean"]
    spread = max(r["p_ai"] for r in transformed) - min(r["p_ai"] for r in transformed)
    return JSONResponse(
        {
            "p_ai_clean": clean,
            "conditions": results,
            # How far the verdict moves across the grid. A small spread is
            # exactly the property the robust half of the score rewards.
            "spread": spread,
            "threshold": float(_META.get("threshold", 0.5)),
        }
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parent / "index.html")
