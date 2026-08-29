"""Choose the decision threshold on held-out generators, so it survives a
distribution shift the way the teammate's `calibrate.py` and DDA both show it
must.

    python -m detector.calibrate <checkpoint> --genval <dir> --calval <dir> [--apply]

Two withheld splits, neither in training:
  genval -- fit the threshold on this
  calval -- a *different* withheld family; score each candidate rule here, so
            the choice of rule never touches the reporting holdout.

Rules ported from the teammate's log. Fixed-false-positive-rate rules are
deliberately omitted: they anchor to the real-image distribution on the
theory that real photos are the stable half, and the teammate measured that
ranking come out inverted against the holdout.

The Final Score (0.5*AUC_clean + 0.5*AUC_robust) is threshold-free, so this
does not move it. It fixes the reported FPR/FNR table and the progress toward
the <=3% goal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .evaluation import _roc_auc
from .model import load_checkpoint, resolve_device
from .transforms import build_eval_transform

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
GRID = np.linspace(0.02, 0.98, 193)


# --- rules: (scores, is_fake) -> threshold ------------------------------------

def _balacc_curve(scores: np.ndarray, is_fake: np.ndarray) -> np.ndarray:
    fake, real = scores[is_fake], scores[~is_fake]
    return np.array([0.5 * ((fake >= t).mean() + (real < t).mean()) for t in GRID])


def rule_balacc_argmax(scores, is_fake):
    return float(GRID[int(np.argmax(_balacc_curve(scores, is_fake)))])


def rule_balacc_plateau(scores, is_fake, tol=0.005):
    curve = _balacc_curve(scores, is_fake)
    plateau = GRID[curve >= curve.max() - tol]
    return float((plateau.min() + plateau.max()) / 2)


def rule_equal_error(scores, is_fake):
    fake, real = scores[is_fake], scores[~is_fake]
    gaps = np.array([abs((real >= t).mean() - (fake < t).mean()) for t in GRID])
    return float(GRID[int(np.argmin(gaps))])


def rule_median_midpoint(scores, is_fake):
    return float((np.median(scores[is_fake]) + np.median(scores[~is_fake])) / 2)


def rule_minmax_fpfn(scores, is_fake):
    """Minimise max(FPR, FNR) -- the quantity the <=3% goal is stated in."""
    fake, real = scores[is_fake], scores[~is_fake]
    obj = np.array([max((real >= t).mean(), (fake < t).mean()) for t in GRID])
    return float(GRID[int(np.argmin(obj))])


RULES = {
    "balacc_argmax": rule_balacc_argmax,
    "balacc_plateau": rule_balacc_plateau,
    "equal_error": rule_equal_error,
    "median_midpoint": rule_median_midpoint,
    "minmax_fpfn": rule_minmax_fpfn,
}
DEFAULT_RULE = "balacc_argmax"


# --- scoring -----------------------------------------------------------------

def _score_dir(model, device, root: Path, batch_size: int = 64, crop_from_native: bool = False):
    tf = build_eval_transform("clean", crop_from_native=crop_from_native)
    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type in ("cuda", "xpu")
        else torch.autocast(device_type="cpu", enabled=False)
    )
    scores, is_fake = [], []
    for cls, fake in (("real", 0), ("fake", 1)):
        files = sorted(p for p in (root / cls).glob("*") if p.suffix.lower() in IMAGE_EXTS)
        for i in range(0, len(files), batch_size):
            batch = torch.stack(
                [tf(Image.open(p).convert("RGB")) for p in files[i : i + batch_size]]
            ).to(device)
            with torch.inference_mode(), autocast:
                p = torch.sigmoid(model(batch)).float().cpu().numpy()
            scores.extend(p.tolist())
            is_fake.extend([bool(fake)] * len(p))
    return np.asarray(scores), np.asarray(is_fake)


def _report_at(name: str, scores: np.ndarray, is_fake: np.ndarray, thr: float) -> str:
    real, fake = scores[~is_fake], scores[is_fake]
    fpr = float((real >= thr).mean()) if len(real) else float("nan")
    fnr = float((fake < thr).mean()) if len(fake) else float("nan")
    auc = _roc_auc(is_fake.astype(int).tolist(), scores.tolist())
    return f"  {name:<22} AUC {auc:.4f}  FPR {fpr:.4f}  FNR {fnr:.4f}  max {max(fpr, fnr):.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--genval", required=True, help="withheld dir with real/ fake/")
    ap.add_argument("--calval", required=True, help="a *different* withheld dir")
    ap.add_argument("--also", action="append", default=[], help="name=dir to also report at the chosen threshold")
    ap.add_argument("--rule", default=DEFAULT_RULE, choices=list(RULES))
    ap.add_argument("--apply", action="store_true", help="write the chosen threshold into the checkpoint")
    args = ap.parse_args()

    device = resolve_device("auto")
    model, meta = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    print(f"checkpoint threshold in metadata: {meta.get('threshold')}")
    crop_from_native = bool(meta.get("crop_from_native", False))

    gen_s, gen_f = _score_dir(model, device, Path(args.genval), crop_from_native=crop_from_native)
    cal_s, cal_f = _score_dir(model, device, Path(args.calval), crop_from_native=crop_from_native)
    print(f"genval: {(~gen_f).sum()} real / {gen_f.sum()} fake   "
          f"calval: {(~cal_f).sum()} real / {cal_f.sum()} fake\n")

    print(f"{'rule':<18} {'thr(gen)':>9} {'calval balacc':>14} {'calval max(FPR,FNR)':>20}")
    scored = {}
    for name, fn in RULES.items():
        t = fn(gen_s, gen_f)
        pred = cal_s >= t
        balacc = 0.5 * ((pred & cal_f).sum() / max(cal_f.sum(), 1)
                        + (~pred & ~cal_f).sum() / max((~cal_f).sum(), 1))
        fpr = (cal_s[~cal_f] >= t).mean()
        fnr = (cal_s[cal_f] < t).mean()
        scored[name] = (t, balacc, max(fpr, fnr))
        print(f"{name:<18} {t:>9.3f} {balacc:>14.4f} {max(fpr, fnr):>20.4f}")

    chosen_t = scored[args.rule][0]
    print(f"\nrule = {args.rule}  ->  threshold {chosen_t:.3f} (fit on genval)")
    print(_report_at("genval", gen_s, gen_f, chosen_t))
    print(_report_at("calval", cal_s, cal_f, chosen_t))
    for spec in args.also:
        nm, d = spec.split("=", 1)
        s, f = _score_dir(model, device, Path(d), crop_from_native=crop_from_native)
        print(_report_at(nm, s, f, chosen_t))

    if args.apply:
        from .model import _plain
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        payload["metadata"] = _plain(payload["metadata"])
        payload["metadata"]["threshold"] = round(float(chosen_t), 4)
        payload["metadata"]["threshold_rule"] = args.rule
        payload["metadata"]["threshold_fit_on"] = str(args.genval)
        torch.save(payload, args.checkpoint)
        print(f"\nwrote threshold {chosen_t:.4f} ({args.rule}) into {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
