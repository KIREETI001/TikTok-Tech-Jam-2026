#!/usr/bin/env bash
# One iteration of the train / calibrate / eval loop on this laptop (Intel Arc XPU).
#   bash run_iteration.sh <run-name> [extra pipeline.py train args...]
#
# 1. train (reads config.yaml)
# 2. calibrate the operating threshold on withheld DRAGON generators
#    (genval fits it, calval picks the rule) -- experiments.md 9h
# 3. evaluate on: internal held-out split, and each held-out set below
# 4. print detector.iteration_report + regenerate ERROR_ANALYSIS.md
set -euo pipefail

RUN="${1:?usage: run_iteration.sh <run-name> [extra args]}"
shift || true
PY="C:/Users/attil/ttj-venv26/Scripts/python.exe"
D="C:/Users/attil/ttj-data"
GENVAL="$D/dragon_genval"
CALVAL="$D/dragon_calval"
# name=path held-out eval sets (organiser set is the reporting number)
EVAL_SETS="${EVAL_SETS:-fs=$D/sid_val448_fs dragon_unseen=$D/dragon_holdout_eval organiser=$D/eval_only_organisers_matched}"

export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR="C:/Users/attil/ttj-cache/sycl26"
export HF_HOME="C:/Users/attil/ttj-cache/hf"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTHONUNBUFFERED=1

filt() { grep --line-buffered -vE "UserWarning|warnings.warn|cache-system uses symlinks|Developer Mode|HF_TOKEN|degraded version" || true; }
run() { MSYS_NO_PATHCONV=1 "$PY" "$@"; }

echo "=== [$RUN] TRAIN @ $(date) ==="
run pipeline.py train --device xpu --run-dir "runs/$RUN" "$@" 2>&1 | filt

echo "=== [$RUN] CALIBRATE threshold on withheld generators @ $(date) ==="
run -m detector.calibrate "runs/$RUN/best.pt" --genval "$GENVAL" --calval "$CALVAL" \
    --rule minmax_fpfn --apply 2>&1 | filt

echo "=== [$RUN] EVAL internal held-out split @ $(date) ==="
run pipeline.py evaluate --device xpu --run-dir "runs/$RUN" 2>&1 | filt
cp "runs/$RUN/summary.json" "runs/$RUN/summary.internal.json"
cp "runs/$RUN/metrics.csv"  "runs/$RUN/metrics.internal.csv"

report_dirs="internal:runs/$RUN"
for spec in $EVAL_SETS; do
    name="${spec%%=*}"; path="${spec#*=}"
    echo "=== [$RUN] EVAL held-out '$name' @ $(date) ==="
    mkdir -p "runs/${RUN}_${name}"
    run pipeline.py evaluate --device xpu --run-dir "runs/${RUN}_${name}" \
        --checkpoint "runs/$RUN/best.pt" --data "$path" 2>&1 | filt
    report_dirs="$report_dirs ${name}:runs/${RUN}_${name}"
done

echo "=== [$RUN] REPORT @ $(date) ==="
run -m detector.iteration_report $(echo "$report_dirs" | tr ' ' '\n' | cut -d: -f2)
run -m detector.error_report $report_dirs --out "ERROR_ANALYSIS.md" --montage
echo "=== [$RUN] DONE @ $(date) ==="
