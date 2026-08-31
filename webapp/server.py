"""Inference API for the AI-vs-real image detector.

Endpoints:

    POST /predict          one image in, calibrated P(AI) out
    POST /predict/robust   one image in, P(AI) under each of the brief's 15
                           evaluation conditions out
    POST /predict/batch    many images in, a P(AI) + verdict table out
                           (the folder-scoring path the CLI exposes, in the UI)

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
import math
import os
import sys
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from typing import List
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
CHECKPOINT = os.environ.get("DETECTOR_CHECKPOINT", "runs/iter6/best.pt")
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


# Temperature for the reported probability. The detector's raw logits are
# large (median |logit| ~ 7.7 on held-out data) -- typical for a neural
# classifier trained with BCE, and why the competition scores threshold-free
# ROC-AUC rather than probability accuracy. T > 1 softens the probability
# without changing any verdict or the ranking. Fitted on withheld generators.
TEMPERATURE = float(os.environ.get("DETECTOR_TEMPERATURE", "1.33"))


def _confidence(logit: float) -> str:
    """A qualitative band from the logit magnitude (calibration-free)."""
    a = abs(logit)
    if a >= 4.0:   # p < 1.8%% or > 98.2%%
        return "decisive"
    if a >= 1.5:   # p < 18%% or > 82%%
        return "clear"
    return "borderline"


@torch.no_grad()
def _logit(tensor: torch.Tensor) -> torch.Tensor:
    out = _MODEL(tensor)
    if out.ndim == 2 and out.shape[1] == 1:
        out = out[:, 0]
    return out


@torch.no_grad()
def _score(image: Image.Image, condition: str = "clean") -> float:
    """P(AI) for one image under one evaluation condition."""
    crop_from_native = bool(_META.get("crop_from_native", False))
    transform = build_eval_transform(condition, crop_from_native=crop_from_native)
    tensor = transform(image).unsqueeze(0).to(_DEVICE)
    return float(_probabilities(_MODEL, tensor)[0])


@torch.no_grad()
def _score_many(images: list[Image.Image], chunk: int = 16) -> list[tuple[float, float]]:
    """(temperature-scaled P(AI), raw logit) per image, clean condition, batched."""
    crop_from_native = bool(_META.get("crop_from_native", False))
    transform = build_eval_transform("clean", crop_from_native=crop_from_native)
    out: list[tuple[float, float]] = []
    for i in range(0, len(images), chunk):
        batch = torch.stack([transform(im) for im in images[i : i + chunk]]).to(_DEVICE)
        for lg in _logit(batch).float().cpu().tolist():
            out.append((1.0 / (1.0 + math.exp(-lg / TEMPERATURE)), lg))
    return out


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
            "branch_kind": _META.get("branch_kind"),
            "clip_model": _META.get("clip_model"),
            "parameters": _META.get("parameter_count"),
            "crop_from_native": bool(_META.get("crop_from_native", False)),
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
    crop_from_native = bool(_META.get("crop_from_native", False))
    tensor = build_eval_transform("clean", crop_from_native=crop_from_native)(image).unsqueeze(0).to(_DEVICE)
    logit = float(_logit(tensor)[0])
    probability = 1.0 / (1.0 + math.exp(-logit / TEMPERATURE))
    thr = float(_META.get("threshold", 0.5))
    verdict = "AI-generated" if logit >= math.log(thr / (1.0 - thr)) else "authentic"
    return JSONResponse(
        {
            "p_ai": probability,
            "p_authentic": 1.0 - probability,
            "logit": logit,
            "verdict": verdict,
            "confidence": _confidence(logit),
            "threshold": thr,
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


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)) -> JSONResponse:
    """Calibrated P(AI) + verdict for many images at once.

    The same folder-scoring the CLI does (`pipeline.py predict --input <dir>`),
    exposed for the UI so a whole set can be checked in one pass. Capped at
    200 files per request; scored in memory, nothing retained.
    """
    _load()
    if len(files) > 200:
        raise HTTPException(status_code=413, detail="Max 200 images per batch.")

    names: list[str] = []
    images: list[Image.Image] = []
    errors: list[dict] = []
    for f in files:
        try:
            raw = await _upload_bytes(f)
            images.append(_read_image(raw))
            names.append(f.filename or f"image_{len(names)}")
        except HTTPException as exc:
            errors.append({"filename": f.filename, "error": exc.detail})

    scored = _score_many(images) if images else []
    thr = float(_META.get("threshold", 0.5))
    logit_thr = math.log(thr / (1.0 - thr))          # verdict in raw-logit space
    results = [
        {
            "filename": n,
            "p_ai": p,                                # temperature-scaled, for display
            "verdict": "AI-generated" if lg >= logit_thr else "authentic",
            "confidence": _confidence(lg),
            "width": im.width,
            "height": im.height,
        }
        for n, (p, lg), im in zip(names, scored, images)
    ]
    results.sort(key=lambda r: -r["p_ai"])
    n_ai = sum(1 for r in results if r["verdict"] == "AI-generated")
    return JSONResponse(
        {
            "threshold": thr,
            "count": len(results),
            "n_ai": n_ai,
            "n_authentic": len(results) - n_ai,
            "results": results,
            "errors": errors,
        }
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parent / "index.html")
