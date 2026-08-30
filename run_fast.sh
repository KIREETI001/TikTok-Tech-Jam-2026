#!/usr/bin/env bash
# Lean iteration for the final sprint: train -> calibrate -> eval only the two
# held-out sets that decide the submission (organiser composition + DRAGON
# unseen). Skips the internal/fs evals and the montage regeneration -- those
# run once at the end via run_iteration.sh on the winning checkpoint.
#   bash run_fast.sh <run-name> [extra pipeline.py train args...]
set -euo pipefail

RUN="${1:?usage: run_fast.sh <run-name>}"
shift || true
PY="${PYTHON:-python}"
D="${TTJ_DATA:-./data}"
GENVAL="$D/dragon_genval"
CALVAL="$D/dragon_calval"
EVAL_SETS="${EVAL_SETS:-organiser=$D/eval_only_organisers_matched dragon_unseen=$D/dragon_holdout_eval}"

export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR="${SYCL_CACHE_DIR:-$HOME/.cache/sycl}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTHONUNBUFFERED=1

filt() { grep --line-buffered -vE "UserWarning|warnings.warn|cache-system uses symlinks|Developer Mode|HF_TOKEN|degraded version|unauthenticated requests" || true; }
run() { MSYS_NO_PATHCONV=1 "$PY" "$@"; }

echo "=== [$RUN] TRAIN @ $(date) ==="
run pipeline.py train --device "${DEVICE:-auto}" --run-dir "runs/$RUN" "$@" 2>&1 | filt

echo "=== [$RUN] CALIBRATE @ $(date) ==="
run -m detector.calibrate "runs/$RUN/best.pt" --genval "$GENVAL" --calval "$CALVAL" \
    --rule minmax_fpfn --apply 2>&1 | filt

for spec in $EVAL_SETS; do
    name="${spec%%=*}"; path="${spec#*=}"
    echo "=== [$RUN] EVAL '$name' @ $(date) ==="
    mkdir -p "runs/${RUN}_${name}"
    run pipeline.py evaluate --device "${DEVICE:-auto}" --run-dir "runs/${RUN}_${name}" \
        --checkpoint "runs/$RUN/best.pt" --data "$path" 2>&1 | filt
done

echo "=== [$RUN] SCOREBOARD @ $(date) ==="
run -m detector.iteration_report runs/${RUN}_organiser runs/${RUN}_dragon_unseen || true
echo "=== [$RUN] DONE @ $(date) ==="
