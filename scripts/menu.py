"""Friendly setup + run helper for the AI-generated-image detector.

Launched by ../run.bat (Windows) or ../run.sh (Linux/macOS). Also fine to run
directly:  python scripts/menu.py

No arguments. It walks you through: environment setup, getting the model,
a smoke test, predicting a folder, the web demo, and evaluating on the
WildFake benchmark (streamed -- no dataset download).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
CKPT = ROOT / "runs" / "iter7" / "best.pt"
HF_REPO_DEFAULT = "kireeti26/ttj-aigc-detector"


# ---------------------------------------------------------------- helpers
def _venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _py() -> str:
    """The interpreter to run project commands with: the venv's if it exists,
    otherwise the current one."""
    vp = _venv_python()
    return str(vp) if vp.exists() else sys.executable


def _run(args, **kw) -> int:
    print("\n$ " + " ".join(str(a) for a in args) + "\n", flush=True)
    return subprocess.call([str(a) for a in args], cwd=str(ROOT), **kw)


def _pip(*args) -> int:
    return _run([_py(), "-m", "pip", *args])


def _ask(prompt: str, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    try:
        v = input(f"{prompt}{hint}: ").strip()
    except EOFError:
        v = ""
    return v or default


def _deps_ready() -> bool:
    """True if the interpreter we'd run project commands with can import the
    core dependencies."""
    probe = "import numpy, torch, timm, PIL, yaml  # noqa"
    try:
        return subprocess.call(
            [_py(), "-c", probe],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) == 0
    except Exception:
        return False


def _require_deps() -> bool:
    if _deps_ready():
        return True
    where = ".venv" if _venv_python().exists() else f"'{sys.executable}'"
    print(
        f"! Dependencies are not installed in {where}.\n"
        "  Run option 1 (Set up environment) first, then try again."
    )
    return False


def _detect_accelerator() -> str:
    if shutil.which("nvidia-smi"):
        return "cuda"
    try:
        import torch  # noqa

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------- actions
def setup_env() -> None:
    print("\n== Environment setup ==")
    if sys.version_info < (3, 10):
        print(f"! Python {sys.version.split()[0]} is too old. Use 3.10-3.12."); return

    if not VENV.exists():
        print(f"Creating virtual environment at {VENV} ...")
        venv.EnvBuilder(with_pip=True).create(str(VENV))
    else:
        print(f"Reusing existing venv at {VENV}")

    guess = _detect_accelerator()
    print("\nWhich hardware will you run on?")
    print("  1) NVIDIA GPU (CUDA)")
    print("  2) CPU only")
    print("  3) Intel Arc iGPU (XPU) - the box this was built on")
    pick = _ask("choose 1/2/3", {"cuda": "1", "cpu": "2", "xpu": "3"}[guess])

    _pip("install", "-U", "pip", "wheel")
    if pick == "3":
        _pip("install", "torch==2.6.0+xpu", "torchvision==0.21.0+xpu",
             "--index-url", "https://download.pytorch.org/whl/xpu")
        _pip("install", "-r", str(ROOT / "requirements-xpu.txt"))
    elif pick == "1":
        _pip("install", "torch", "torchvision",
             "--index-url", "https://download.pytorch.org/whl/cu124")
        _pip("install", "-r", str(ROOT / "requirements-cuda.txt"))
    else:
        _pip("install", "torch", "torchvision",
             "--index-url", "https://download.pytorch.org/whl/cpu")
        _pip("install", "-r", str(ROOT / "requirements-cuda.txt"))

    print("\nDone. The venv is at .venv - this menu will use it automatically.")


def get_weights() -> None:
    print("\n== Model weights ==")
    if CKPT.exists():
        mb = CKPT.stat().st_size / 1e6
        print(f"Already present: {CKPT}  ({mb:.0f} MB)")
        if _ask("re-download anyway? y/N", "N").lower() != "y":
            return
    repo = _ask("HuggingFace model repo", HF_REPO_DEFAULT)
    code = (
        "from huggingface_hub import hf_hub_download; import shutil, pathlib; "
        f"p = hf_hub_download('{repo}', 'best.pt'); "
        f"dst = pathlib.Path(r'{CKPT}'); dst.parent.mkdir(parents=True, exist_ok=True); "
        "shutil.copy(p, dst); print('saved ->', dst)"
    )
    if _run([_py(), "-c", code]) != 0:
        print(
            f"\nCould not fetch from '{repo}'. If the weights are not published yet:\n"
            "  1) the model owner runs:  python hf_upload/upload.py --repo <user>/<name>\n"
            "  2) then re-run this option with that repo name."
        )


def smoke() -> None:
    print("\n== Smoke test (synthetic data, no downloads) ==")
    if not _require_deps():
        return
    _run([_py(), "pipeline.py", "smoke", "--device", "auto"])


def predict_folder() -> None:
    print("\n== Predict a folder of images ==")
    if not _require_deps():
        return
    if not CKPT.exists():
        print("! No model weights. Run option 2 first."); return
    folder = _ask("path to a folder of images", str(ROOT / "demo_images"))
    out = _ask("output JSON path", str(ROOT / "predictions.json"))
    _run([_py(), "pipeline.py", "predict", "--input", folder, "--output", out,
          "--checkpoint", str(CKPT), "--device", "auto"])
    print(f"\nWrote {out} (pred 0/1) and {Path(out).with_suffix('.scores.csv')} (probabilities).")


def demo() -> None:
    print("\n== Web demo ==")
    if not _require_deps():
        return
    if not CKPT.exists():
        print("! No model weights. Run option 2 first."); return
    env = {**os.environ, "DETECTOR_CHECKPOINT": str(CKPT), "DETECTOR_DEVICE": "auto"}
    print("Starting http://localhost:8000  -  Ctrl+C to stop.")
    _run([_py(), "-m", "uvicorn", "webapp.server:app", "--port", "8000"], env=env)


def eval_wildfake() -> None:
    print("\n== Evaluate on the WildFake benchmark ==")
    if not _require_deps():
        return
    if not CKPT.exists():
        print("! No model weights. Run option 2 first."); return
    print(
        "Streams ~2,400 images (real COCO + 5 fake generators) directly from\n"
        "WildFake's ModelScope host over HTTP range requests. No dataset download.\n"
        "Takes ~10-20 min depending on the CDN.\n"
    )
    if _ask("proceed? Y/n", "Y").lower() == "n":
        return
    n = _ask("images per generator (200 = full)", "200")
    _run([_py(), "scripts/evaluate_wildfake.py", "--checkpoint", str(CKPT),
          "--per-generator", n, "--real-count", str(int(n) * 6), "--device", "auto"])


def train() -> None:
    print("\n== Train from scratch ==")
    print(
        "This is a real commitment:\n"
        "  - ~50 GB of public training data to materialise (SID_Set, DRAGON,\n"
        "    Community-Forensics-Small) from HuggingFace\n"
        "  - a GPU (NVIDIA or Intel Arc); CPU training is impractical\n"
        "  - several hours per iteration\n\n"
        "The corpus-build scripts are with the maintainer (local_reference/).\n"
        "Once your data is materialised into one folder:\n\n"
        "  export TTJ_DATA=/path/to/that/folder   (set TTJ_DATA=... on Windows)\n"
        "  bash run_iteration.sh myrun\n\n"
        "See README.md -> 'Reproduce the training'."
    )
    d = os.environ.get("TTJ_DATA")
    if d and Path(d).exists() and _ask(f"TTJ_DATA={d} looks set. Start run_iteration.sh now? y/N", "N").lower() == "y":
        _run(["bash", "run_iteration.sh", _ask("run name", "myrun")])


MENU = [
    ("Set up environment (venv + dependencies)", setup_env),
    ("Download the model weights (best.pt from HuggingFace)", get_weights),
    ("Smoke test (synthetic data, ~2 min, no downloads)", smoke),
    ("Predict a folder of your own images", predict_folder),
    ("Launch the interactive web demo", demo),
    ("Evaluate on the WildFake benchmark (streamed, ~10-20 min)", eval_wildfake),
    ("Train from scratch (large download + GPU + hours)", train),
]


def main() -> int:
    while True:
        print("\n" + "=" * 60)
        print("  AI-Generated Image Detector  -  setup & run")
        print("  model: ViT-S + frozen CLIP-B/16  -  Final Score 0.933")
        print("=" * 60)
        if _deps_ready():
            dp = "ready" if _venv_python().exists() else "ready (system Python)"
        else:
            dp = "not installed - run option 1"
        wp = "yes" if CKPT.exists() else "no - run option 2"
        print(f"  dependencies: {dp}    weights: {wp}\n")
        for i, (label, _) in enumerate(MENU, 1):
            print(f"  [{i}] {label}")
        print("  [0] Exit")
        choice = _ask("\nchoose")
        if choice == "0" or choice.lower() in ("q", "exit", ""):
            return 0
        try:
            MENU[int(choice) - 1][1]()
        except (ValueError, IndexError):
            print("  ? not a valid option")
        except KeyboardInterrupt:
            print("\n(interrupted)")


if __name__ == "__main__":
    raise SystemExit(main())
