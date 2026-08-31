"""Zero-friction entry point: point it at a folder of images, get predictions.

    python inference.py <image_folder>
    python inference.py <image_folder> <output.json>

Deliberately the dumbest possible interface. `pipeline.py predict` needs a
subcommand and an `--input` flag, and `run.sh` opens an interactive menu that
blocks on stdin -- neither survives being run hands-off by a grading script,
a CI job, or a judge who does not read the README first. This takes a bare
positional path, prompts for nothing, downloads the weights itself if they
are missing, and exits non-zero with a readable message on failure.

Everything it does is delegated to detector.evaluation.predict_folder, so the
numbers are identical to `pipeline.py predict` -- this is an adapter, not a
second implementation.

Outputs, next to each other:
    predictions.json        [{"image_path", "pred": 0|1, "probability_ai": float}]
    predictions.scores.csv  image_path, probability_ai, confidence
"""

from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

CKPT = ROOT / "runs" / "iter6" / "best.pt"
GH_RELEASE_URL = (
    "https://github.com/KIREETI001/TikTok-Tech-Jam-2026/releases/download/"
    "v1.0-iter6a/best.pt"
)

USAGE = (
    "usage: python inference.py <image_folder> [output.json]\n"
    "\n"
    "  <image_folder>  directory of images to score (searched recursively)\n"
    "  [output.json]   where to write results (default: predictions.json)\n"
)


def ensure_weights() -> None:
    """Fetch the checkpoint if it is not already on disk.

    Standard library only, so this works on a fresh clone before the optional
    huggingface_hub dependency is installed.
    """
    if CKPT.exists():
        return
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    print("[inference] weights not found; downloading ~87 MB ...", flush=True)
    try:
        with urllib.request.urlopen(GH_RELEASE_URL) as response, CKPT.open("wb") as out:
            shutil.copyfileobj(response, out)
    except Exception as exc:  # noqa: BLE001 - surface a readable message, not a traceback
        CKPT.unlink(missing_ok=True)  # never leave a truncated checkpoint behind
        raise SystemExit(
            f"[inference] could not download the model weights: {exc}\n"
            f"            fetch them manually from {GH_RELEASE_URL}\n"
            f"            and save to {CKPT}"
        ) from exc
    print(f"[inference] saved {CKPT} ({CKPT.stat().st_size / 1e6:.0f} MB)", flush=True)


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(USAGE)
        return 0 if argv else 2

    input_dir = Path(argv[0]).expanduser()
    output_json = Path(argv[1]).expanduser() if len(argv) > 1 else Path("predictions.json")

    if not input_dir.exists():
        raise SystemExit(f"[inference] no such folder: {input_dir}")
    if not input_dir.is_dir():
        raise SystemExit(f"[inference] not a folder: {input_dir}")

    ensure_weights()

    from detector.evaluation import predict_folder

    records = predict_folder(
        checkpoint=CKPT,
        input_dir=input_dir,
        output_json=output_json,
        device="auto",
    )
    n_ai = sum(1 for r in records if r["pred"] == 1)
    print(
        f"[inference] {len(records)} images -> {output_json}  "
        f"({n_ai} AI-generated, {len(records) - n_ai} authentic)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
