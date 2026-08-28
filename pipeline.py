"""Small command-line pipeline for PS5 AI-generated image detection."""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from detector.data import ingest_training_data, load_manifest
from detector.data_sources import get_data_source
from detector.evaluation import evaluate, predict_folder
from detector.model import resolve_device
from detector.training import train_model, train_model_from_datasets


DEFAULT_CONFIG = "config.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest, train, evaluate, and run the PS5 image detector."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser, *, data: bool = False) -> None:
        command.add_argument("--config", default=DEFAULT_CONFIG)
        command.add_argument("--run-dir", help="override config output_dir")
        if data:
            command.add_argument("--data", help="override config data_dir")

    ingest = commands.add_parser("ingest", help="validate, deduplicate, and split data")
    common(ingest, data=True)

    train = commands.add_parser("train", help="ingest data and train one detector")
    common(train, data=True)
    train.add_argument("--epochs", type=int)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--no-pretrained", action="store_true")
    train.add_argument(
        "--offline",
        action="store_true",
        help=(
            "skip re-downloading the pretrained checkpoint if cached; does NOT "
            "avoid network access for the sid_set_stream/mixed data sources, "
            "which always fetch from the HF Hub -- use --config with "
            "data_source: local for a fully offline run"
        ),
    )

    evaluate_command = commands.add_parser(
        "evaluate", help="evaluate clean and transformed images"
    )
    common(evaluate_command)
    evaluate_command.add_argument(
        "--data", help="external labeled benchmark; omit for manifest validation split"
    )
    evaluate_command.add_argument("--checkpoint", help="default: <run-dir>/best.pt")
    evaluate_command.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )

    run = commands.add_parser("run", help="one command: ingest, train, and evaluate")
    common(run, data=True)
    run.add_argument("--epochs", type=int)
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run.add_argument("--no-pretrained", action="store_true")
    run.add_argument("--offline", action="store_true")

    predict = commands.add_parser("predict", help="predict every image in a folder")
    common(predict)
    predict.add_argument("--input", required=True, help="unlabeled image directory")
    predict.add_argument("--output", default="predictions.json")
    predict.add_argument("--checkpoint", help="default: <run-dir>/best.pt")
    predict.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    smoke = commands.add_parser(
        "smoke",
        help="self-contained GPU/pipeline regression check (synthetic images, no ../Data/ needed)",
    )
    smoke.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="fails if a GPU is present but training doesn't actually run on cuda",
    )
    smoke.add_argument("--epochs", type=int, default=2)

    materialize = commands.add_parser(
        "materialize-sid-set",
        help=(
            "fetch N SID_Set shards from the HF Hub and save them locally as "
            "real/fake JPEGs, so `evaluate` can run the normal robustness "
            "matrix against a genuinely different dataset"
        ),
    )
    materialize.add_argument("--split", choices=("train", "validation"), default="validation")
    materialize.add_argument("--shards", type=int, default=5)
    materialize.add_argument(
        "--output", required=True, help="destination folder (gets real/ and fake/ subfolders)"
    )

    report = commands.add_parser(
        "report",
        help=(
            "turn one or more evaluate run-dirs' metrics.csv/summary.json into a "
            "markdown report: the brief's Final Score (0.5*AUC_clean + 0.5*AUC_robust), "
            "a per-condition accuracy/AUC table, and worst-condition callouts"
        ),
    )
    report.add_argument(
        "--run-dir", action="append", required=True, help="an evaluate run-dir; repeatable"
    )
    report.add_argument(
        "--label", action="append", help="display name for each --run-dir, same order; optional"
    )
    report.add_argument("--output", help="write markdown to this file (also prints to stdout)")
    return parser


def _load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Config does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a YAML mapping.")
    return payload, source.parent


def _path(value: str | Path, *, base: Path) -> Path:
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    config, base = _load_config(args.config)
    data_value = getattr(args, "data", None) or config.get("data_dir", "../Data/train")
    run_value = getattr(args, "run_dir", None) or config.get("output_dir", "runs/latest")
    return {
        **config,
        "data_dir": _path(data_value, base=base),
        "run_dir": _path(run_value, base=base),
    }


def _ingest(settings: dict[str, Any]):
    """Ingest via the configured ``data_source`` ("local", "sid_set_stream",
    or "mixed" -- see detector/data_sources/). Returns ``(train_dataset,
    val_dataset, info)``; only the "local" source's ``info`` carries
    ``train_records``/``val_records`` (path-based ``ImageRecord``s) so
    ``evaluate``/``run`` can reuse the held-out split -- "sid_set_stream" and
    "mixed" have no local paths to offer, so ``run``'s automatic post-train
    evaluate is skipped for both (see ``main()`` below).
    """
    ingest_fn = get_data_source(str(settings.get("data_source", "local")))
    train_dataset, val_dataset, info = ingest_fn(settings)
    print(
        f"[INGEST] source={settings.get('data_source', 'local')} "
        f"train={info['train_count']} validation={info['val_count']} "
        f"manifest={info['manifest']}"
    )
    return train_dataset, val_dataset, info


def _train(
    settings: dict[str, Any],
    args: argparse.Namespace,
    train_dataset,
    val_dataset,
    info: dict[str, Any],
) -> Path:
    pretrained = bool(settings.get("pretrained", True)) and not args.no_pretrained
    checkpoint = train_model_from_datasets(
        train_dataset,
        val_dataset,
        settings["run_dir"],
        epochs=args.epochs or int(settings.get("epochs", 5)),
        batch_size=int(settings.get("batch_size", 16)),
        learning_rate=float(settings.get("learning_rate", 1e-5)),
        weight_decay=float(settings.get("weight_decay", 0.01)),
        num_workers=int(settings.get("num_workers", 0)),
        seed=int(settings.get("seed", 2026)),
        device=args.device,
        pretrained=pretrained,
        local_files_only=bool(settings.get("local_files_only", False)) or args.offline,
        threshold=float(settings.get("threshold", 0.5)),
        train_count=info["train_count"],
        val_count=info["val_count"],
        model_type=str(settings.get("model_type", "vit")),
        vit_checkpoint=settings.get("vit_checkpoint"),
    )
    saved_parameter_count = torch.load(checkpoint, map_location="cpu", weights_only=True)["metadata"][
        "parameter_count"
    ]
    print(f"[MODEL] parameters={saved_parameter_count:,} checkpoint={checkpoint}")
    return checkpoint


def _evaluate(
    settings: dict[str, Any], args: argparse.Namespace, *, records=None, checkpoint=None
) -> dict[str, object]:
    selected_checkpoint = Path(
        checkpoint or getattr(args, "checkpoint", None) or Path(settings["run_dir"]) / "best.pt"
    )
    # ``run --data`` identifies the training root. When the caller has already
    # supplied held-out records, do not reinterpret that same argument as an
    # external evaluation set.
    external_data = getattr(args, "data", None) if records is None else None
    if external_data:
        records = None
        data_root = _path(external_data, base=Path.cwd())
    else:
        data_root = None
        if records is None:
            records = load_manifest(Path(settings["run_dir"]) / "manifest.csv", split="validation")
    summary = evaluate(
        checkpoint=selected_checkpoint,
        data_root=data_root,
        records=records,
        output_dir=settings["run_dir"],
        batch_size=int(settings.get("batch_size", 16)),
        num_workers=int(settings.get("num_workers", 0)),
        device=args.device,
        threshold=float(settings.get("threshold", 0.5)),
    )
    robust = summary["robust_mean"]
    print(
        f"[EVALUATE] conditions={summary['conditions_evaluated']} "
        f"clean_accuracy={summary['clean']['accuracy']:.4f} "
        f"mean_transformed_accuracy={robust['accuracy']:.4f}"
    )
    return summary


def _make_synthetic_images(root: Path, *, per_class: int = 20, size: int = 224) -> None:
    """Writes deterministic synthetic real/fake JPEGs so `smoke` needs no
    external dataset. "real" images are solid colors; "fake" images are
    random noise -- an easy, arbitrary distinction good enough to exercise
    the full ingest -> train pipeline and confirm loss actually decreases.
    """
    rng = np.random.default_rng(2026)
    (root / "real").mkdir(parents=True, exist_ok=True)
    (root / "fake").mkdir(parents=True, exist_ok=True)
    for i in range(per_class):
        color = rng.integers(0, 256, size=3)
        solid = np.broadcast_to(color, (size, size, 3)).astype(np.uint8)
        Image.fromarray(solid, mode="RGB").save(root / "real" / f"{i:03d}.jpg", quality=90)

        noise = rng.integers(0, 256, size=(size, size, 3)).astype(np.uint8)
        Image.fromarray(noise, mode="RGB").save(root / "fake" / f"{i:03d}.jpg", quality=90)


def _cmd_smoke(args: argparse.Namespace) -> None:
    if args.device in ("cuda", "auto") and not torch.cuda.is_available():
        if args.device == "cuda":
            raise RuntimeError("smoke --device cuda requested but torch.cuda.is_available() is False.")
        print("[SMOKE] WARNING: no CUDA GPU detected; running on CPU (cannot verify GPU training).")

    with tempfile.TemporaryDirectory(prefix="pipeline_smoke_") as tmp:
        tmp_path = Path(tmp)
        data_dir = tmp_path / "data"
        run_dir = tmp_path / "run"
        _make_synthetic_images(data_dir)

        train_records, val_records = ingest_training_data(
            data_dir, run_dir / "manifest.csv", validation_fraction=0.25, seed=2026
        )
        checkpoint = train_model(
            train_records,
            val_records,
            run_dir,
            epochs=max(2, args.epochs),
            batch_size=8,
            num_workers=0,
            device=args.device,
            train_augment_probability=0.0,  # keep smoke fast and deterministic-ish
        )

        resolved = resolve_device(args.device)
        if args.device != "cpu" and resolved.type != "cuda" and torch.cuda.is_available():
            raise RuntimeError(
                f"smoke resolved to device={resolved} despite a CUDA GPU being available."
            )

        history_path = run_dir / "training.csv"
        with history_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        first_loss = float(rows[0]["train_loss"])
        last_loss = float(rows[-1]["train_loss"])
        if not last_loss < first_loss:
            raise RuntimeError(
                f"smoke train_loss did not decrease: epoch1={first_loss:.4f} -> "
                f"epoch{len(rows)}={last_loss:.4f}"
            )

        print(
            f"[SMOKE] OK: device={resolved} train_loss {first_loss:.4f} -> {last_loss:.4f} "
            f"over {len(rows)} epochs, checkpoint={checkpoint.name} (discarded)"
        )


def _sanitize_img_id(img_id: str, index: int) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(img_id))
    return f"{index:06d}_{safe}"[:150]


def _cmd_materialize_sid_set(args: argparse.Namespace) -> None:
    """Fetch args.shards SID_Set shards and save them locally as real/fake
    JPEGs under args.output, so the normal path-based `evaluate` (and its
    15-condition robustness matrix) can run against them exactly like any
    other local benchmark -- see detector/data_sources/sid_set_stream.py
    for the underlying HF-Hub streaming.
    """
    from detector.data_sources.sid_set_stream import CLASS_TO_IDX, SIDSetDataset

    idx_to_class = {idx: name.lower() for name, idx in CLASS_TO_IDX.items()}
    output = Path(args.output).expanduser().resolve()
    for name in idx_to_class.values():
        (output / name).mkdir(parents=True, exist_ok=True)

    print(f"[MATERIALIZE] fetching {args.shards} '{args.split}' shard(s) from SID_Set ...")
    dataset = SIDSetDataset(args.split, args.shards)
    print(f"[MATERIALIZE] {len(dataset)} images indexed; saving to {output}")

    counts = {name: 0 for name in idx_to_class.values()}
    for i in range(len(dataset)):
        image, label = dataset[i]
        img_id, _label = dataset.samples[i]
        class_name = idx_to_class[label]
        filename = _sanitize_img_id(img_id, i) + ".jpg"
        image.save(output / class_name / filename, quality=95)
        counts[class_name] += 1
        if (i + 1) % 500 == 0 or i + 1 == len(dataset):
            print(f"[MATERIALIZE] saved {i + 1}/{len(dataset)}")

    print(f"[MATERIALIZE] done: {counts}")


def _read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _report_section(run_dir: Path, label: str) -> str:
    summary = _read_json(run_dir / "summary.json")
    clean = summary["clean"]
    robust = summary["robust_mean"]
    worst = summary["worst_condition"]
    final_score = 0.5 * clean["roc_auc"] + 0.5 * robust["roc_auc"]

    lines = [f"## {label}", ""]
    lines.append(
        f"**Final Score** (brief formula: 0.5 x AUC_clean + 0.5 x AUC_robust): "
        f"**{final_score:.4f}**"
    )
    lines.append("")
    lines.append(f"Clean accuracy: {clean['accuracy']:.4f} | Clean ROC-AUC: {clean['roc_auc']:.4f}")
    lines.append(
        f"Robust-mean accuracy: {robust['accuracy']:.4f} | Robust-mean ROC-AUC: {robust['roc_auc']:.4f}"
    )
    lines.append("")
    lines.append("| Condition | Accuracy | ROC-AUC | FPR | FNR |")
    lines.append("|---|---|---|---|---|")
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            lines.append(
                f"| {row['condition']} | {float(row['accuracy']):.4f} | "
                f"{float(row['roc_auc']):.4f} | {float(row['fpr']):.4f} | {float(row['fnr']):.4f} |"
            )
    lines.append("")
    lines.append(
        f"Worst condition (accuracy): **{worst['accuracy']['condition']}** "
        f"({worst['accuracy']['value']:.4f})"
    )
    lines.append(
        f"Worst condition (ROC-AUC): **{worst['roc_auc']['condition']}** "
        f"({worst['roc_auc']['value']:.4f})"
    )
    lines.append("")
    return "\n".join(lines)


def _cmd_report(args: argparse.Namespace) -> None:
    run_dirs = [Path(p) for p in args.run_dir]
    labels = list(args.label or [])
    labels += [run_dirs[i].name for i in range(len(labels), len(run_dirs))]

    sections = [
        _report_section(run_dir, label) for run_dir, label in zip(run_dirs, labels, strict=True)
    ]
    document = "\n".join(["# Robustness Report", ""] + sections)

    print(document)
    if args.output:
        Path(args.output).write_text(document + "\n", encoding="utf-8")
        print(f"[REPORT] written to {Path(args.output).resolve()}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "smoke":
            _cmd_smoke(args)
            return 0
        if args.command == "materialize-sid-set":
            _cmd_materialize_sid_set(args)
            return 0
        if args.command == "report":
            _cmd_report(args)
            return 0
        settings = _settings(args)
        if args.command == "ingest":
            _ingest(settings)
        elif args.command == "train":
            train_dataset, val_dataset, info = _ingest(settings)
            _train(settings, args, train_dataset, val_dataset, info)
        elif args.command == "evaluate":
            _evaluate(settings, args)
        elif args.command == "run":
            train_dataset, val_dataset, info = _ingest(settings)
            checkpoint = _train(settings, args, train_dataset, val_dataset, info)
            # ``evaluate`` only knows how to read path-based ImageRecords
            # (detector.data's manifest.csv); that's only available for the
            # "local" data source. "sid_set_stream" and "mixed" have no
            # local paths to hand it, so skip the automatic post-train
            # evaluate rather than fail -- run `pipeline.py evaluate --data
            # <local-benchmark>` separately instead.
            held_out_records = info.get("val_records")
            if held_out_records is not None:
                _evaluate(settings, args, records=held_out_records, checkpoint=checkpoint)
            else:
                print(
                    "[EVALUATE] skipped: automatic post-train evaluation needs a "
                    "local-path data source; run `pipeline.py evaluate --data "
                    "<folder>` against a local benchmark instead."
                )
        elif args.command == "predict":
            checkpoint = Path(args.checkpoint or Path(settings["run_dir"]) / "best.pt")
            records = predict_folder(
                checkpoint=checkpoint,
                input_dir=args.input,
                output_json=args.output,
                batch_size=int(settings.get("batch_size", 16)),
                num_workers=int(settings.get("num_workers", 0)),
                device=args.device,
                threshold=float(settings.get("threshold", 0.5)),
            )
            print(f"[PREDICT] images={len(records)} output={Path(args.output).resolve()}")
        else:
            raise AssertionError(f"Unhandled command: {args.command}")
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
