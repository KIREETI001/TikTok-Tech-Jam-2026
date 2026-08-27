# Robust AI-Generated Image Detection

A small pipeline for TikTok TechJam 2026 Problem Statement 5:

```text
ingest labelled images -> fine-tune one detector -> evaluate robustness -> predict a folder
```

The model is the Community Forensics ViT-S/224 detector (21,666,049
parameters, well below the 2-billion-parameter limit). `1` means AI-generated
and `0` means authentic.

## Repository

```text
model_training/
|-- pipeline.py              ingest, train, evaluate, predict, run, smoke, and
|                            materialize-sid-set commands
|-- config.yaml              paths, data source, and basic training settings
|-- experiments.md           log of every training/eval attempt: config,
|                            metrics, and what each one found
|-- detector/
|   |-- data.py              image discovery, validation, deduplication, split
|   |-- data_sources/        swappable train data sources (see below)
|   |   |-- local.py         ../Data/train on local disk
|   |   |-- sid_set_stream.py  SID_Set streamed over HTTP from the HF Hub
|   |   `-- mixed.py         local + sid_set_stream concatenated (default)
|   |-- model.py             Community Forensics model and checkpoints
|   |-- transforms.py        training transforms and brief evaluation matrix
|   |-- training.py          one BCE/AdamW fine-tuning loop
|   `-- evaluation.py        metrics, robustness gaps, errors, JSON prediction
|-- requirements.txt         runtime dependencies
`-- README.md                setup, usage, assumptions, and limitations
```

### Data sources

`config.yaml`'s `data_source` key picks how training images are ingested:

- `local` -- reads the `../Data/train` tree below only.
- `sid_set_stream` -- streams the SID_Set dataset directly from the Hugging
  Face Hub over HTTP at training time, with no local dataset copy needed.
  Useful on a machine (or CI runner) without `../Data/`. Configure shard
  counts via `sid_set_train_shards` (default 13). Needs the `pyarrow` and
  `huggingface-hub` dependencies (already in `requirements.txt`).
- `mixed` (**default**) -- concatenates `local` and `sid_set_stream`'s
  `train` shards into one training pool. Added after evaluating the
  `local`-only checkpoint against SID_Set and finding a real cross-dataset
  generalization gap (~95% accuracy on PS5's own held-out test vs. ~70% on
  SID_Set); mixing in ~8,400 SID_Set images closed most of that gap
  (83.7% on SID_Set) with no measurable cost on PS5 performance. See
  `experiments.md` sections 4-5 for the full diagnosis and numbers.

All three feed the same `train`/`run` commands; only `evaluate`'s automatic
reuse of the held-out split (via `run`) is local-path-only today -- evaluate
a `sid_set_stream`/`mixed`-trained checkpoint with
`pipeline.py evaluate --data <local-benchmark>` instead (see
`materialize-sid-set` below for turning a streamed dataset into one of
those).

## Data layout

Keep data outside Git:

```text
../Data/
|-- train/
|   |-- real/                authentic training images
|   `-- fake/                AI-generated training images
`-- test/                    validation-only held-out set (this project's is
    |-- real/                named test/; benchmark/ works identically --
    `-- fake/                any folder name is accepted via --data)
```

`authentic`, `non_aigc`, `ai`, and `aigc` are also accepted class-folder
names. Ingestion creates a deterministic, stratified train/validation split in
the manifest. Exact duplicate bytes are removed; the same bytes appearing
under both labels are rejected.

Do not point training at WildFake or another validation-only benchmark.

## Setup

PyTorch does not ship CUDA-enabled wheels for Python 3.14 yet (only CPU
builds), so this project pins **3.12** for GPU training:

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
& ".venv\Scripts\Activate.ps1"
```

Verify the GPU is actually being used before training:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If this prints `False`, the environment fell back to a CPU-only `torch`
build -- check `python --version` is 3.12.x, not 3.14, and reinstall `torch`
from the `cu126` index above.

The first pretrained run downloads the pinned Community Forensics checkpoint.
Set `local_files_only: true` in `config.yaml` when it is already cached and the
machine must remain offline.

## Checkpoints, logs, and other run artifacts

`.pth`/`.pt` checkpoints, training logs, and generated manifests are
deliberately **not** tracked in git (see `.gitignore`) -- they're large,
regenerate deterministically from `config.yaml` plus the data, and bloat
clone/checkout time for no benefit. Share a checkpoint by uploading it
somewhere durable (e.g. the Hugging Face Hub) and linking it here, not by
committing the binary.

## Commands

```powershell
# Inspect and split the labelled training data.
python pipeline.py ingest

# Ingest and train; save the best validation-F1 checkpoint.
python pipeline.py train

# Evaluate that checkpoint on the held-out validation split.
python pipeline.py evaluate

# One command: ingest -> train -> evaluate.
python pipeline.py run

# Evaluate an external, validation-only benchmark.
python pipeline.py evaluate --data ..\Data\test

# Required directory-to-JSON inference.
python pipeline.py predict --input <image-folder> --output predictions.json

# Self-contained regression check: synthetic data, no ../Data/ needed.
# Fails loudly if a GPU is present but training silently falls back to CPU
# (the failure mode this project hit once already), or if loss doesn't drop.
python pipeline.py smoke

# Fetch SID_Set shards from the HF Hub and save them locally as real/fake
# JPEGs, so `evaluate` can run its normal robustness matrix against a
# genuinely different dataset (see experiments.md section 4).
python pipeline.py materialize-sid-set --split validation --shards 5 --output ..\Data\sid_set_eval
python pipeline.py evaluate --data ..\Data\sid_set_eval --checkpoint runs\latest\best.pt
```

Use `--config`, `--run-dir`, `--checkpoint`, `--device`, and `--epochs` to
override the small set of defaults. Run `python pipeline.py <command> --help`
for details.

## Outputs

The default run directory is `runs/latest/`:

```text
manifest.csv             selected images, labels, split, and SHA-256
best.pt                  best validation-F1 model
training.csv             epoch loss and clean validation metrics
metrics.csv              clean plus every required transform/severity
summary.json             mean/worst transformed performance and gaps
errors.csv               representative false positives and false negatives
```

Prediction JSON deliberately uses only the required fields:

```json
[
  {"image_path": "nested/example.jpg", "pred": 1}
]
```

A sibling `predictions.scores.csv` records `probability_ai` and confidence,
keeping confidence evidence separate from the minimal organizer JSON.

## Evaluation matrix

- Clean baseline
- JPEG quality 90, 70, 50, and 30
- Gaussian blur sigma 0.5, 1.0, and 2.0
- Resize to 0.5x and 0.25x, then upscale
- Gaussian noise sigma 0.02, 0.05, and 0.10
- Brightness, contrast, and saturation jitter within +/-20%
- Center crop retaining 80%, then resize back

Each row reports accuracy, F1, ROC-AUC, false-positive rate,
false-negative rate, and the clean-to-transformed gap. `errors.csv` supplies
the requested representative FP/FN evidence.

## Results

Current default config (`mixed` data source, `train_augment_probability:
0.7`), full numbers and methodology in `experiments.md`:

| | PS5 `Data/test` (unseen) | SID_Set (unseen, different dataset) |
|---|---|---|
| Clean accuracy | 95.40% | 83.67% |
| Mean robust accuracy (15 conditions) | 91.18% | 83.66% |

The SID_Set column is a genuine cross-dataset generalization check, not a
held-out split of the same data -- see `experiments.md` sections 4-5 for how
a ~95%/~70% gap there was diagnosed (ruled out calibration and
over-aggressive fine-tuning first) and closed by mixing SID_Set images into
training.

## Limitations

- Results above are checked against one additional dataset (SID_Set) beyond
  PS5's own held-out test; a genuine "generalizes to any dataset" claim
  would want at least one more, still-unseen source (see `experiments.md`'s
  "Not yet attempted" section).
- Public/properly licensed training data remains the operator's responsibility.
- The benchmark is evaluation-only and must never be used for training or
  threshold selection.
- This is a hackathon prototype, not a production moderation system.
- The local PS5 deck does not define the polarity or extra-field rules for
  `pred`; this repository declares and consistently uses `0=real`, `1=AI`.

The previous tests, governance framework, release machinery, submission
documents, and legacy implementation are preserved outside the repository in
`../Repo Archive/`.
