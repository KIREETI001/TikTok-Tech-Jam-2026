"""Compact per-iteration scoreboard: read one or more evaluate run-dirs and
print the numbers the FPR/FNR<=3% push actually turns on.

Usage:
    python -m detector.iteration_report runs/iter1 [runs/iter1_sidval ...]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def _load(run_dir: Path) -> tuple[dict, list[dict]]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return summary, rows


def _f(value: str | float | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def report(run_dir: Path) -> None:
    summary, rows = _load(run_dir)
    clean = summary["clean"]
    robust = summary["robust_mean"]
    final_score = 0.5 * _f(clean["roc_auc"]) + 0.5 * _f(robust["roc_auc"])

    print(f"\n=== {run_dir} ===")
    print(f"threshold (calibrated): {summary.get('threshold')}")
    print(
        f"Final Score  0.5*AUC_clean + 0.5*AUC_robust = {final_score:.4f}   "
        f"(clean {_f(clean['roc_auc']):.4f} / robust {_f(robust['roc_auc']):.4f})"
    )
    print(
        f"clean:  acc {_f(clean['accuracy']):.4f}  FPR {_f(clean['fpr']):.4f}  "
        f"FNR {_f(clean['fnr']):.4f}"
    )
    worst_fpr = max(rows, key=lambda r: _f(r["fpr"]))
    worst_fnr = max(rows, key=lambda r: _f(r["fnr"]))
    worst_auc = min(rows, key=lambda r: _f(r["roc_auc"]))
    print(
        f"worst FPR: {_f(worst_fpr['fpr']):.4f} @ {worst_fpr['condition']}   "
        f"worst FNR: {_f(worst_fnr['fnr']):.4f} @ {worst_fnr['condition']}   "
        f"worst AUC: {_f(worst_auc['roc_auc']):.4f} @ {worst_auc['condition']}"
    )
    within = [
        r["condition"]
        for r in rows
        if _f(r["fpr"]) <= 0.03 and _f(r["fnr"]) <= 0.03
    ]
    print(f"conditions within 3% FPR&FNR: {len(within)}/{len(rows)}")
    print(f"{'condition':<20} {'acc':>7} {'auc':>7} {'FPR':>7} {'FNR':>7}")
    for r in rows:
        flag = "" if (_f(r["fpr"]) <= 0.03 and _f(r["fnr"]) <= 0.03) else "  <-"
        print(
            f"{r['condition']:<20} {_f(r['accuracy']):>7.4f} {_f(r['roc_auc']):>7.4f} "
            f"{_f(r['fpr']):>7.4f} {_f(r['fnr']):>7.4f}{flag}"
        )


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    for path in argv:
        report(Path(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
