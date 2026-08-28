"""Mixed data source: local PS5 data, a sample of SID_Set, and a sample of
DRAGON, combined into one training pool.

Added after evaluating the fine-tuned Community Forensics checkpoint
against SID_Set (a dataset never seen during training): clean accuracy on
SID_Set was only ~70%, versus ~95% on the PS5 held-out test set. Two
follow-up checks ruled out the cheaper explanations:
  - Threshold recalibration recovered at most ~1pt (69.5% -> 70.3%) --
    not a calibration problem.
  - A zero-shot (never PS5-fine-tuned) checkpoint scored *worse* on
    SID_Set (60.1% clean) than the fine-tuned one -- fine-tuning didn't
    narrow generalization, it modestly helped.
Both point at the same conclusion: the model simply hasn't seen enough
generator diversity. Mixing in real SID_Set training images (its "train"
shards; "validation" shards are deliberately left untouched, since they're
what the held-out cross-dataset evaluation above uses -- pulling from them
here would leak eval data into training) closed most of that gap.

DRAGON was added as a third source (see dragon.py) for further generator
diversity beyond what SID_Set alone provides -- specifically several
distilled/fast diffusion variants (LCM, SDXL Turbo/Lightning, Hyper-SD)
not represented in either PS5 or SID_Set. DRAGON is fake-only (no real
class), so it only adds to the fake side of the combined pool -- see
experiments.md section 7 for the resulting class-balance note.
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import ConcatDataset, Dataset

from . import dragon, local, sid_set_stream


def ingest(settings: dict[str, Any]) -> tuple[Dataset, Dataset, dict[str, Any]]:
    local_train, local_val, local_info = local.ingest(settings)
    sidset_train, sidset_val, sidset_info = sid_set_stream.ingest(settings)

    # DRAGON is skippable via dragon_shards: 0 -- added after it hit a
    # persistent (not transient) DNS resolution failure against the HF Hub
    # in this environment; retrying endlessly wastes GPU time for no
    # benefit, so this lets training proceed on the proven PS5+SID_Set pool
    # without it rather than blocking on a network problem outside this
    # codebase's control.
    train_parts = [local_train, sidset_train]
    val_parts = [local_val, sidset_val]
    info = {
        "local_train_count": local_info["train_count"],
        "local_val_count": local_info["val_count"],
        "sidset_train_count": sidset_info["train_count"],
        "sidset_val_count": sidset_info["val_count"],
    }
    manifest_parts = [local_info["manifest"], sidset_info["manifest"]]

    if int(settings.get("dragon_shards", 5)) > 0:
        dragon_train, dragon_val, dragon_info = dragon.ingest(settings)
        train_parts.append(dragon_train)
        val_parts.append(dragon_val)
        info["dragon_train_count"] = dragon_info["train_count"]
        info["dragon_val_count"] = dragon_info["val_count"]
        manifest_parts.append(dragon_info["manifest"])

    info["train_count"] = sum(info[k] for k in info if k.endswith("_train_count"))
    info["val_count"] = sum(info[k] for k in info if k.endswith("_val_count"))
    info["manifest"] = " + ".join(str(p) for p in manifest_parts)

    train_dataset = ConcatDataset(train_parts)
    val_dataset = ConcatDataset(val_parts)
    return train_dataset, val_dataset, info
