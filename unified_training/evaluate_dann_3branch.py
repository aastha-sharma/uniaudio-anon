"""
Evaluate DANN 3-Branch Unified Model on a given test manifest.

Usage example:

python evaluate_dann_3branch.py \
  --model ./checkpoints/dann_3branch/best_stage3_dann.pth \
  --manifest data/cross_domain/manifest_test_singing_only.csv \
  --checkpoint-dir ./checkpoints \
  --output ./results/dann_3branch_singing.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


import argparse
import json

from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

from src.data.dataset import DeepfakeDataset
from unified_training.unified_model_3branch import UnifiedDeepfakeDetector3Branch
from unified_training.train_dann_3branch import UnifiedDeepfakeDetector3Branch_DANN


def compute_eer(labels, scores):
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(_[idx])


def make_test_loader(manifest, batch_size=32, num_workers=4):
    dataset = DeepfakeDataset(
        manifest,
        split="test",
        segment_sec=6,
        sr=16000,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataset, loader


def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_scores = []
    all_datasets = []

    with torch.no_grad():
        for batch in dataloader:
            # Compatible with both (audio, label, dataset, stem) and shorter tuples
            if len(batch) == 4:
                audio, labels, dataset_names, _ = batch
            elif len(batch) == 3:
                audio, labels, dataset_names = batch
            else:
                audio, labels = batch[:2]
                dataset_names = None

            audio = audio.to(device)

            # For evaluation we don't need domain predictions; lambda_=0
            fused_logit, _ = model(audio, lambda_=0.0, return_domain_preds=False)

            scores = torch.sigmoid(fused_logit).cpu().numpy().ravel().tolist()
            all_scores.extend(scores)
            all_labels.extend(labels.numpy().tolist())

            if dataset_names is not None:
                # dataset_names may be a list/array of strings
                all_datasets.extend(list(dataset_names))
            else:
                all_datasets.extend(["unknown"] * labels.shape[0])

    # Overall metrics
    eer, eer_thresh = compute_eer(all_labels, all_scores)
    auc = roc_auc_score(all_labels, all_scores)

    # Accuracy at EER threshold
    scores_arr = np.asarray(all_scores)
    labels_arr = np.asarray(all_labels)
    preds_eer = (scores_arr >= eer_thresh).astype(int)
    acc_eer = (preds_eer == labels_arr).mean()

    # Per-dataset metrics
    per_dataset = {}
    all_datasets = np.asarray(all_datasets)
    for name in np.unique(all_datasets):
        mask = (all_datasets == name)
        if mask.sum() < 10:
            continue
        ds_labels = labels_arr[mask]
        ds_scores = scores_arr[mask]
        ds_eer, _ = compute_eer(ds_labels, ds_scores)
        ds_auc = roc_auc_score(ds_labels, ds_scores)
        per_dataset[name] = {
            "samples": int(mask.sum()),
            "eer": float(ds_eer),
            "auc": float(ds_auc),
        }

    metrics = {
        "eer": float(eer),
        "auc": float(auc),
        "acc_at_eer": float(acc_eer),
        "eer_threshold": float(eer_thresh),
        "num_samples": int(len(all_labels)),
        "num_bonafide": int((labels_arr == 0).sum()),
        "num_spoof": int((labels_arr == 1).sum()),
        "per_dataset": per_dataset,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate DANN 3-Branch Unified Model")
    parser.add_argument("--model", required=True, help="Path to DANN checkpoint (.pth)")
    parser.add_argument("--manifest", required=True, help="Path to TEST manifest CSV")
    parser.add_argument("--checkpoint-dir", default="./checkpoints",
                        help="Dir with expert checkpoints (for base model init)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to JSON file to save metrics")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("EVALUATE DANN 3-BRANCH MODEL")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model:  {args.model}")
    print(f"Manifest: {args.manifest}")

    # 1) Build base unified model (3-branch)
    base_model = UnifiedDeepfakeDetector3Branch(checkpoint_dir=args.checkpoint_dir)

    # 2) Wrap with DANN
    model = UnifiedDeepfakeDetector3Branch_DANN(
        base_model,
        use_w2v_only=True,  # matches your training config
    ).to(device)

    # 3) Load trained DANN weights
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 4) Build test loader
    dataset, loader = make_test_loader(
        args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"\nLoaded {len(dataset)} test samples")

    # 5) Run evaluation
    metrics = evaluate(model, loader, device)

    print("\n============================================================")
    print("OVERALL RESULTS")
    print("============================================================")
    print(f"EER: {metrics['eer']*100:.2f}%")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Accuracy at EER threshold: {metrics['acc_at_eer']*100:.2f}%")
    print(f"EER threshold: {metrics['eer_threshold']:.4f}")
    print(f"Samples: {metrics['num_samples']} "
          f"(Bonafide: {metrics['num_bonafide']}, Spoof: {metrics['num_spoof']})")

    if metrics["per_dataset"]:
        print("\n============================================================")
        print("PER-DATASET RESULTS")
        print("============================================================")
        for name, m in metrics["per_dataset"].items():
            print(f"\n{name}:")
            print(f"  EER: {m['eer']*100:.2f}%")
            print(f"  AUC: {m['auc']:.4f}")
            print(f"  Samples: {m['samples']}")

    # 6) Save JSON if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": str(args.model),
            "manifest": str(args.manifest),
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print("\n============================================================")
        print(f"Results saved to: {out_path}")
        print("============================================================")


if __name__ == "__main__":
    main()
