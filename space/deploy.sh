#!/usr/bin/env bash
# Assemble a self-contained Hugging Face Docker Space and upload it.
#
#   bash space/deploy.sh <hf-username>/<space-name>
#
# Uses the huggingface_hub HTTP API (no git / git-lfs). Authenticate first
# with `hf auth login`, or set HF_TOKEN to a write-scoped token.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
REPO_ROOT="$(cd .. && pwd)"
PY="${PYTHON:-python}"

SPACE_ID="${1:-}"
[ -n "$SPACE_ID" ] || { echo "usage: bash space/deploy.sh <hf-username>/<space-name>" >&2; exit 1; }

CKPT="$REPO_ROOT/runs/iter6/best.pt"
[ -f "$CKPT" ] || { echo "missing $CKPT — fetch the weights first (run.sh -> option 2)" >&2; exit 1; }

BUILD="$HERE/build"
rm -rf "$BUILD"; mkdir -p "$BUILD"
cp Dockerfile requirements.txt README.md .dockerignore "$BUILD/"
cp -r "$REPO_ROOT/detector" "$BUILD/detector"
cp -r "$REPO_ROOT/webapp"   "$BUILD/webapp"
cp "$CKPT" "$BUILD/best.pt"
find "$BUILD" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

echo "Assembled $BUILD ($(du -sh "$BUILD" | cut -f1))"
SPACE_ID="$SPACE_ID" BUILD="$BUILD" "$PY" - <<'PYEOF'
import os
from huggingface_hub import HfApi
repo_id, build = os.environ["SPACE_ID"], os.environ["BUILD"]
api = HfApi()
print("Authenticated as", api.whoami()["name"])
api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(folder_path=build, repo_id=repo_id, repo_type="space",
                  commit_message="Deploy iter6a AI-image detector (Final Score 0.9362)")
sub = repo_id.replace("/", "-").replace("_", "-")
print(f"\nDone.\n  Space:    https://huggingface.co/spaces/{repo_id}\n  Live URL: https://{sub}.hf.space")
PYEOF
