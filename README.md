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
|-- pipeline.py              ingest, train, evaluate, predict, and run commands
|-- config.yaml              paths and basic training settings
|-- detector/
|   |-- data.py              image discovery, validation, deduplication, split
|   |-- model.py             Community Forensics model and checkpoints
|   |-- transforms.py        training transforms and brief evaluation matrix
|   |-- training.py          one BCE/AdamW fine-tuning loop
|   `-- evaluation.py        metrics, robustness gaps, errors, JSON prediction
|-- requirements.txt         runtime dependencies
`-- README.md                setup, usage, assumptions, and limitations
```

## Data layout

Keep data outside Git:

```text
../Data/
|-- train/
|   |-- real/                authentic training images
|   `-- fake/                AI-generated training images
`-- benchmark/               optional validation-only benchmark
    |-- real/
    `-- fake/
```

`authentic`, `non_aigc`, `ai`, and `aigc` are also accepted class-folder
names. Ingestion creates a deterministic, stratified train/validation split in
the manifest. Exact duplicate bytes are removed; the same bytes appearing
under both labels are rejected.

Do not point training at WildFake or another validation-only benchmark.

## Setup

```powershell
uv venv "..\Repo Archive\model_training\.venv" --python 3.14
uv pip install --python "..\Repo Archive\model_training\.venv\Scripts\python.exe" -r requirements.txt
& "..\Repo Archive\model_training\.venv\Scripts\Activate.ps1"
```

The first pretrained run downloads the pinned Community Forensics checkpoint.
Set `local_files_only: true` in `config.yaml` when it is already cached and the
machine must remain offline.

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
python pipeline.py evaluate --data ..\Data\benchmark

# Required directory-to-JSON inference.
python pipeline.py predict --input <image-folder> --output predictions.json
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

## Limitations

- `../Data/` is currently empty, so there are no real accuracy or robustness
  claims yet.
- Public/properly licensed training data remains the operator's responsibility.
- The benchmark is evaluation-only and must never be used for training or
  threshold selection.
- This is a hackathon prototype, not a production moderation system.
- The local PS5 deck does not define the polarity or extra-field rules for
  `pred`; this repository declares and consistently uses `0=real`, `1=AI`.

The previous tests, governance framework, release machinery, submission
documents, and legacy implementation are preserved outside the repository in
`../Repo Archive/`.
