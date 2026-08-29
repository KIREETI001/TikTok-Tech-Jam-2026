"""Score a checkpoint on the resolution-matched WildFake+COCO set
(runs/wildfake_matched, see build_matched_eval_local.py). Usage:
    python scripts/eval_matched.py <checkpoint> [--label NAME]
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detector.data import load_labeled_root
from detector.evaluation import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("--label", default=None)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

records = load_labeled_root("runs/wildfake_matched")
summary = evaluate(
    checkpoint=args.checkpoint, records=records,
    output_dir=f"runs/matched_eval_{args.label or Path(args.checkpoint).parent.name}",
    device=args.device, batch_size=32,
)
c, r, rg = summary["clean"], summary["robust_mean"], summary["robust_mean_grouped"]
label = args.label or args.checkpoint
print(f"\n=== {label} on resolution-matched WildFake+COCO ({len(records)} images) ===")
print(f"AUC_clean          {c['roc_auc']:.4f}")
print(f"AUC_robust (flat)  {r['roc_auc']:.4f}   Final: {0.5*c['roc_auc']+0.5*r['roc_auc']:.4f}")
print(f"AUC_robust (grouped){rg['roc_auc']:.4f}   Final: {0.5*c['roc_auc']+0.5*rg['roc_auc']:.4f}")
