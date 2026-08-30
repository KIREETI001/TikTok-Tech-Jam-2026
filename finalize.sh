#!/usr/bin/env bash
# Full evaluation + deliverable regeneration for ONE finished checkpoint.
#   bash finalize.sh <checkpoint.pt> <tag>
# Produces, under runs/<tag>_*/:
#   - 15-condition robustness metrics for every held-out set
#   - per-generator ROC-AUC on the organiser composition
#   - ERROR_ANALYSIS.md with FP/FN montages
#   - a compact scoreboard
set -euo pipefail

CKPT="${1:?usage: finalize.sh <checkpoint> <tag>}"
TAG="${2:?usage: finalize.sh <checkpoint> <tag>}"
PY="${PYTHON:-python}"
D="${TTJ_DATA:-./data}"

export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR="${SYCL_CACHE_DIR:-$HOME/.cache/sycl}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONUNBUFFERED=1
filt() { grep --line-buffered -vE "UserWarning|warnings.warn|symlinks|Developer Mode|unauthenticated|degraded" || true; }
run() { MSYS_NO_PATHCONV=1 "$PY" "$@"; }

declare -A SETS=(
  [fs]="$D/sid_val448_fs"
  [dragon_unseen]="$D/dragon_holdout_eval"
  [organiser]="$D/eval_only_organisers_matched"
)

report_dirs=""
for name in fs dragon_unseen organiser; do
    echo "=== [$TAG] EVAL $name @ $(date) ==="
    out="runs/${TAG}_${name}"; mkdir -p "$out"
    run pipeline.py evaluate --device "${DEVICE:-auto}" --run-dir "$out" --checkpoint "$CKPT" --data "${SETS[$name]}" 2>&1 | filt
    report_dirs="$report_dirs ${name}:$out"
done

echo "=== [$TAG] per-generator (organiser) @ $(date) ==="
run -m detector.organiser_eval "$CKPT" "$D/eval_only_organisers_matched/real" "$D/eval_only_organisers_matched/fake" \
    --out "runs/${TAG}_organiser/per_generator.json" 2>&1 | filt

echo "=== [$TAG] scoreboard + error analysis @ $(date) ==="
run -m detector.iteration_report $(echo "$report_dirs" | tr ' ' '\n' | cut -d: -f2) || true
run -m detector.error_report $report_dirs --out "ERROR_ANALYSIS.md" --montage 2>&1 | filt || true
echo "=== [$TAG] DONE @ $(date) ==="
