"""Score a checkpoint on the organisers' composition and report it the way the
brief does: Final Score = 0.5*AUC_clean + 0.5*AUC_robust, overall and per
generator.

    python -m detector.organiser_eval <checkpoint> <real_dir> <fake_dir> [--out report.md]

``fake_dir`` filenames must start with ``<Generator>_`` (that is how
scratchpad/wildfake_fetch.py names them) so per-generator rows can be split
out. AUC_robust is the mean AUC across the 14 transformed conditions in
detector.transforms.CONDITION_SPECS, matching detector.evaluation.

*** This runs against evaluation-only data (see EVAL_ONLY_DATASETS.md). It
only ever reads. ***
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from .evaluation import _roc_auc
from .model import load_checkpoint, resolve_device
from .transforms import EVALUATION_CONDITIONS, build_eval_transform

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def _generator_of(path: Path) -> str:
    return path.name.split("_")[0]


def _score(model, device, paths, condition, batch_size=32, crop_from_native=False):
    """Sigmoid scores for ``paths`` under one evaluation condition.

    ``crop_from_native`` must match how ``checkpoint`` was trained -- see
    ``detector.transforms._geometry``. Showing a resize-trained model
    native crops (or the reverse) silently costs real AUC.
    """
    transform = build_eval_transform(condition, crop_from_native=crop_from_native)
    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type in ("cuda", "xpu")
        else torch.autocast(device_type="cpu", enabled=False)
    )
    out: list[float] = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        batch = torch.stack(
            [transform(Image.open(p).convert("RGB")) for p in chunk]
        ).to(device)
        with torch.inference_mode(), autocast:
            probs = torch.sigmoid(model(batch)).float().cpu()
        out.extend(probs.tolist())
    return out


def run(checkpoint: str | Path, real_dir: Path, fake_dir: Path, batch_size: int = 32):
    device = resolve_device("auto")
    model, metadata = load_checkpoint(checkpoint, device=device)
    model.eval()
    threshold = float(metadata.get("threshold", 0.5))
    crop_from_native = bool(metadata.get("crop_from_native", False))

    reals = sorted(p for p in real_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    fakes = sorted(p for p in fake_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    by_generator: dict[str, list[Path]] = defaultdict(list)
    for path in fakes:
        by_generator[_generator_of(path)].append(path)

    print(
        f"checkpoint={checkpoint} threshold={threshold}\n"
        f"{len(reals)} real / {len(fakes)} fake "
        f"({len(by_generator)} generators: {', '.join(sorted(by_generator))})",
        flush=True,
    )

    per_condition: dict[str, dict] = {}
    for condition in EVALUATION_CONDITIONS:
        real_scores = _score(model, device, reals, condition, batch_size, crop_from_native)
        fake_scores = _score(model, device, fakes, condition, batch_size, crop_from_native)
        labels = [0] * len(real_scores) + [1] * len(fake_scores)
        overall_auc = _roc_auc(labels, real_scores + fake_scores)

        fpr = sum(s >= threshold for s in real_scores) / max(len(real_scores), 1)
        fnr = sum(s < threshold for s in fake_scores) / max(len(fake_scores), 1)

        generator_auc = {}
        index = 0
        offsets = {}
        for path in fakes:
            offsets[path] = index
            index += 1
        for generator, paths in by_generator.items():
            subset = [fake_scores[offsets[p]] for p in paths]
            generator_auc[generator] = _roc_auc(
                [0] * len(real_scores) + [1] * len(subset), real_scores + subset
            )

        per_condition[condition] = {
            "auc": overall_auc,
            "fpr": fpr,
            "fnr": fnr,
            "generator_auc": generator_auc,
        }
        print(
            f"  {condition:<20} AUC {overall_auc:.4f}  FPR {fpr:.4f}  FNR {fnr:.4f}",
            flush=True,
        )

    transformed = [c for c in EVALUATION_CONDITIONS if c != "clean"]
    auc_clean = per_condition["clean"]["auc"]
    auc_robust = sum(per_condition[c]["auc"] for c in transformed) / len(transformed)
    return {
        "checkpoint": str(checkpoint),
        "threshold": threshold,
        "counts": {"real": len(reals), "fake": len(fakes)},
        "final_score": 0.5 * auc_clean + 0.5 * auc_robust,
        "auc_clean": auc_clean,
        "auc_robust": auc_robust,
        "per_condition": per_condition,
        "generators": {
            generator: {
                "auc_clean": per_condition["clean"]["generator_auc"][generator],
                "auc_robust": sum(
                    per_condition[c]["generator_auc"][generator] for c in transformed
                )
                / len(transformed),
            }
            for generator in sorted(by_generator)
        },
    }


def render(result: dict) -> str:
    lines = ["# Results on the organisers' composition", ""]
    lines.append(
        f"{result['counts']['real']} authentic vs {result['counts']['fake']} AI "
        f"— {len(result['generators'])} generators, "
        f"{result['counts']['fake'] // max(len(result['generators']), 1)} each."
    )
    lines += ["", "| | |", "|---|---|"]
    lines.append(f"| **Final Score** | **{result['final_score']:.4f}** |")
    lines.append(f"| AUC_clean | {result['auc_clean']:.4f} |")
    lines.append(f"| AUC_robust | {result['auc_robust']:.4f} |")
    lines += ["", "| Generator | AUC clean | AUC robust |", "|---|---|---|"]
    for generator, values in sorted(
        result["generators"].items(), key=lambda kv: kv[1]["auc_clean"]
    ):
        lines.append(
            f"| {generator} | {values['auc_clean']:.3f} | {values['auc_robust']:.3f} |"
        )
    clean = result["per_condition"]["clean"]
    lines += [
        "",
        f"At the checkpoint's operating threshold ({result['threshold']}): "
        f"clean FPR {clean['fpr']:.4f}, clean FNR {clean['fnr']:.4f}.",
        "",
        "| Condition | AUC | FPR | FNR |",
        "|---|---|---|---|",
    ]
    for condition, values in result["per_condition"].items():
        lines.append(
            f"| {condition} | {values['auc']:.4f} | {values['fpr']:.4f} | {values['fnr']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("real_dir")
    parser.add_argument("fake_dir")
    parser.add_argument("--out")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    result = run(
        args.checkpoint, Path(args.real_dir), Path(args.fake_dir), args.batch_size
    )
    document = render(result)
    print("\n" + document)
    if args.out:
        Path(args.out).write_text(document, encoding="utf-8")
        Path(args.out).with_suffix(".json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(f"[REPORT] {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
