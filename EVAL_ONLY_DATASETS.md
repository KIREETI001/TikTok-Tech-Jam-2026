# Evaluation-only datasets — do NOT train on these

The organisers' brief designates WildFake as a **validation-only benchmark**.
Anything listed here is off-limits for training, for validation-split
selection, and for threshold calibration. It exists purely to produce an
honest held-out number.

| Local path | Contents | Source |
|---|---|---|
| `C:/Users/attil/ttj-data/eval_only_wildfake/fake/` | 1,200 AI images — 200 each from WildFake's ADM, DALLE, DDIM, DDPM, Imagen, VQDM | ModelScope `hy2628982280/WildFake`, `Images/Diffusion_based/*.zip` |
| `C:/Users/attil/ttj-data/eval_only_coco2017/real/` | 1,200 authentic photographs | COCO 2017 val (HF mirror `joortif/coco-2017-val-reduced`) |
| `C:/Users/attil/ttj-data/eval_only_organisers/` | The two above combined into one `real/` + `fake/` tree — the organisers' composition | assembled locally |

Both sides are stored at ≤448 px short edge, matching every other set in this
project so the 15-condition robustness matrix stays comparable.

## Why the `eval_only_` prefix

`config.yaml`'s `data_dir` is the only thing that decides what gets trained
on. The prefix makes an accidental pointing-at obvious in a diff, and
`detector/data.py`'s `load_labeled_root` refuses any path whose name starts
with `eval_only_` when called from the training path (see
`assert_not_eval_only`).

## How WildFake was obtained without downloading 1.3 TB

WildFake on ModelScope is ~1.3 TB and its per-generator archives are 6–26 GB
each. `scratchpad/wildfake_fetch.py` never downloads them: it resolves
ModelScope's signed CDN redirect, then serves `zipfile` a seekable file-like
object backed by HTTP `Range` requests. Reading a 6 GB archive's central
directory costs ~1.5 s and a few MB; only the ~200 sampled members per
generator are then range-fetched. Sampling is seeded (`random.Random(2026)`)
so the set is reproducible.
