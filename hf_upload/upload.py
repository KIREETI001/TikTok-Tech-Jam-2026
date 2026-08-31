"""Upload the trained checkpoint to the Hugging Face Hub.

Needs a write token: `hf auth login` first, or set HF_TOKEN.

    python hf_upload/upload.py --repo <user>/ttj-aigc-detector
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True, help="e.g. kireeti26/ttj-aigc-detector")
ap.add_argument("--private", action="store_true")
a = ap.parse_args()

api = HfApi()
create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)

for local, remote in [
    (ROOT / "runs/iter6/best.pt", "best.pt"),
    (ROOT / "hf_upload/model_config.json", "model_config.json"),
    (ROOT / "hf_upload/README.md", "README.md"),
]:
    api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=a.repo, repo_type="model")
    print(f"uploaded {remote}")

print(f"\ndone: https://huggingface.co/{a.repo}")
