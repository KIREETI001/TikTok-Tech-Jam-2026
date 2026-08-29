#!/usr/bin/env bash
# Full evaluation + deliverable regeneration for ONE finished checkpoint, on
# this device. Adapted from finalize.sh, which hard-codes the teammate's
# machine (Python path, data dir, XPU device, SYCL env vars) and cannot run
# here as-is.
#
#   bash finalize_local.sh <checkpoint.pt> <tag>
#
# Produces, under runs/<tag>_*/:
#   - 15-condition robustness metrics on three held-out sets:
#       sidset      -- ../Data/sid_set_eval_test (materialized earlier this
#                       session; mixed full_synthetic + tampered, unlike the
#                       filtered training corpus -- labelled as such below)
#       dragon      -- ../iter5_data/dragon_holdout (8 DRAGON generators
#                       never in training; scripts/build_iter5_corpus.py)
#       organiser   -- runs/wildfake_matched (resolution-shortcut-free
#                       WildFake+COCO; scripts/build_matched_eval_local.py)
#   - per-generator ROC-AUC on the organiser set
#   - ERROR_ANALYSIS.md with FP/FN montages
#   - a compact scoreboard across all three
set -euo pipefail

CKPT="${1:?usage: finalize_local.sh <checkpoint> <tag>}"
TAG="${2:?usage: finalize_local.sh <checkpoint> <tag>}"
PY="./.venv/Scripts/python.exe"

declare -A SETS=(
  [sidset]="../Data/sid_set_eval_test"
  [dragon]="../iter5_data/dragon_holdout"
  [organiser]="runs/wildfake_matched"
)

report_dirs=""
for name in sidset dragon organiser; do
    echo "=== [$TAG] EVAL $name @ $(date) ==="
    out="runs/${TAG}_${name}"
    mkdir -p "$out"
    "$PY" pipeline.py evaluate --device cuda --run-dir "$out" --checkpoint "$CKPT" --data "${SETS[$name]}"
    report_dirs="$report_dirs ${name}:$out"
done

echo "=== [$TAG] per-generator (organiser) @ $(date) ==="
"$PY" -m detector.organiser_eval "$CKPT" "runs/wildfake_matched/real" "runs/wildfake_matched/fake" \
    --out "runs/${TAG}_organiser/per_generator.md"

echo "=== [$TAG] scoreboard + error analysis @ $(date) ==="
"$PY" -m detector.iteration_report $(echo "$report_dirs" | tr ' ' '\n' | cut -d: -f2)
"$PY" -m detector.error_report $report_dirs --out "ERROR_ANALYSIS.md" --montage
echo "=== [$TAG] DONE @ $(date) ==="
