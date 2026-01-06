"""
Evaluate the 3-Branch Unified Model on a split with corrected t-DCF.
Outputs: results/unified_3branch_<split>_<timestamp>/{metrics_overall.json, metrics_per_dataset.csv, scores.csv}
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc as pr_auc_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

from unified_model_3branch import UnifiedDeepfakeDetector3Branch
from src.data.dataset import DeepfakeDataset
from src.train.train_expert import compute_eer


def compute_auc(labels, scores):
    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def compute_pr_auc(labels, scores):
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y, s, pos_label=1)
    return float(pr_auc_fn(recall, precision))


def compute_tDCF_min(
    labels,
    scores,
    p_tar=0.9801,
    p_non=0.0099,
    p_spoof=0.01,
    C_miss_asv=1.0,
    C_fa_asv=10.0,
    C_miss_cm=1.0,
    C_fa_cm=10.0,
    P_miss_asv=0.01,
    P_fa_asv=0.01,
):
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan"), -1

    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    P_miss_cm = 1.0 - tpr  # spoof -> bonafide
    P_fa_cm = fpr          # bonafide -> spoof

    C_def = p_tar * C_miss_asv * P_miss_asv + p_non * C_fa_asv * P_fa_asv
    if C_def <= 0:
        return float("nan"), -1

    tDCF = (p_tar * C_miss_cm * P_miss_cm + p_spoof * C_fa_cm * P_fa_cm) / C_def
    idx = int(np.nanargmin(tDCF))
    return float(tDCF[idx]), float(idx)


def evaluate_unified_3branch(model, dataloader, device):
    model.eval()
    all_labels, all_scores = [], []
    per_dataset_scores = defaultdict(list)
    per_dataset_labels = defaultdict(list)
    all_gate_weights = []
    per_sample_rows = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating", ncols=100):
            if len(batch) == 4:
                audio, labels, dataset_names, file_ids = batch
            else:
                audio, labels, dataset_names = batch
                file_ids = [None] * len(labels)

            audio = audio.to(device)
            fused_logit, gate_weights = model(audio)

            scores = torch.sigmoid(fused_logit).cpu().numpy().flatten()
            labs = labels.numpy().astype(int).tolist()

            all_scores.extend(scores.tolist())
            all_labels.extend(labs)
            all_gate_weights.append(gate_weights.detach().cpu().numpy())

            for dsname, label, score, fid in zip(dataset_names, labs, scores.tolist(), file_ids):
                dsname = str(dsname)
                per_dataset_labels[dsname].append(int(label))
                per_dataset_scores[dsname].append(float(score))
                per_sample_rows.append(
                    {
                        "dataset": dsname,
                        "file": None if fid is None else str(fid),
                        "label": int(label),
                        "score": float(score),
                    }
                )

    avg_gate_weights = np.concatenate(all_gate_weights, axis=0).mean(axis=0)
    overall_eer = compute_eer(all_labels, all_scores)
    overall_auc = compute_auc(all_labels, all_scores)
    overall_pr_auc = compute_pr_auc(all_labels, all_scores)

    return (
        overall_eer,
        overall_auc,
        overall_pr_auc,
        per_dataset_labels,
        per_dataset_scores,
        avg_gate_weights,
        per_sample_rows,
        all_labels,
        all_scores,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3-Branch Unified Model")
    parser.add_argument("--manifest", default="manifest.csv")
    parser.add_argument("--checkpoint", default="checkpoints/unified_3branch/final_unified_3branch.pth")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--split", default="test")

    # t-DCF config
    parser.add_argument("--tdcf-p-tar", type=float, default=0.9801)
    parser.add_argument("--tdcf-p-non", type=float, default=0.0099)
    parser.add_argument("--tdcf-p-spoof", type=float, default=0.01)
    parser.add_argument("--tdcf-c-miss-asv", type=float, default=1.0)
    parser.add_argument("--tdcf-c-fa-asv", type=float, default=10.0)
    parser.add_argument("--tdcf-c-miss-cm", type=float, default=1.0)
    parser.add_argument("--tdcf-c-fa-cm", type=float, default=10.0)
    parser.add_argument("--tdcf-pmiss-asv", type=float, default=0.01)
    parser.add_argument("--tdcf-pfa-asv", type=float, default=0.01)

    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("3-BRANCH UNIFIED MODEL - EVALUATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Manifest: {args.manifest}")
    print(f"Split: {args.split}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 70)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.results_dir) / f"unified_3branch_{args.split}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nInitializing 3-Branch Unified Model...")
    model = UnifiedDeepfakeDetector3Branch(checkpoint_dir=args.checkpoint_dir).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    print("3-branch checkpoint loaded")

    print(f"\nLoading '{args.split}' dataset...")
    test_dataset = DeepfakeDataset(args.manifest, split=args.split)
    print(f"Found {len(test_dataset)} samples in '{args.split}' split")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    print("\n" + "=" * 70)
    print("EVALUATION IN PROGRESS")
    print("=" * 70)

    (
        overall_eer,
        overall_auc,
        overall_pr_auc,
        per_dataset_labels,
        per_dataset_scores,
        avg_gate_weights,
        per_sample_rows,
        all_labels,
        all_scores,
    ) = evaluate_unified_3branch(model, test_loader, device)

    tdcf_overall, _ = compute_tDCF_min(
        labels=all_labels,
        scores=all_scores,
        p_tar=args.tdcf_p_tar,
        p_non=args.tdcf_p_non,
        p_spoof=args.tdcf_p_spoof,
        C_miss_asv=args.tdcf_c_miss_asv,
        C_fa_asv=args.tdcf_c_fa_asv,
        C_miss_cm=args.tdcf_c_miss_cm,
        C_fa_cm=args.tdcf_c_fa_cm,
        P_miss_asv=args.tdcf_pmiss_asv,
        P_fa_asv=args.tdcf_pfa_asv,
    )

    print("\n" + "=" * 70)
    print(f"3-BRANCH UNIFIED MODEL - {args.split.upper()} RESULTS")
    print("=" * 70)
    print(f"Overall EER:     {overall_eer:.4f} ({overall_eer*100:.2f}%)")
    print(f"Overall ROC-AUC: {overall_auc:.4f}")
    print(f"Overall PR-AUC:  {overall_pr_auc:.4f}")
    print(f"Overall t-DCF:   {tdcf_overall:.4f}")

    rows = []
    print(f"\nPer-Dataset {args.split.upper()} Metrics:")
    print("-" * 70)
    for dataset_name in sorted(per_dataset_labels.keys()):
        labels = per_dataset_labels[dataset_name]
        scores = per_dataset_scores[dataset_name]

        eer = compute_eer(labels, scores)
        auc_ = compute_auc(labels, scores)
        pr_ = compute_pr_auc(labels, scores)
        tdcf, _ = compute_tDCF_min(
            labels=labels,
            scores=scores,
            p_tar=args.tdcf_p_tar,
            p_non=args.tdcf_p_non,
            p_spoof=args.tdcf_p_spoof,
            C_miss_asv=args.tdcf_c_miss_asv,
            C_fa_asv=args.tdcf_c_fa_asv,
            C_miss_cm=args.tdcf_c_miss_cm,
            C_fa_cm=args.tdcf_c_fa_cm,
            P_miss_asv=args.tdcf_pmiss_asv,
            P_fa_asv=args.tdcf_pfa_asv,
        )

        n_samples = len(labels)
        n_spoof = int(np.sum(labels))
        n_bonafide = n_samples - n_spoof

        print(
            f"  {dataset_name:20s}: "
            f"EER {eer:.4f} ({eer*100:6.2f}%) | "
            f"ROC-AUC {auc_:.4f} | PR-AUC {pr_:.4f} | "
            f"t-DCF {tdcf:.4f} | "
            f"{n_samples} samples (spoof: {n_spoof}, bonafide: {n_bonafide})"
        )

        rows.append(
            {
                "dataset": dataset_name,
                "n_samples": n_samples,
                "n_spoof": n_spoof,
                "n_bonafide": n_bonafide,
                "eer": float(eer),
                "roc_auc": float(auc_),
                "pr_auc": float(pr_),
                "tDCF_min_norm": float(tdcf),
            }
        )

    print("\nAverage Gate Weights (Expert Importance):")
    print("-" * 70)
    expert_names = ["LogMel", "MFCC", "W2V-ResNet"]
    for name, weight in zip(expert_names, avg_gate_weights):
        bar = "█" * int(max(0.0, float(weight)) * 50)
        print(f"  {name:20s}: {weight:.4f} ({weight*100:5.2f}%) {bar}")

    overall_dict = {
        "model": "3-branch (LogMel + MFCC + W2V-ResNet)",
        "split": args.split,
        "overall": {
            "eer": float(overall_eer),
            "roc_auc": float(overall_auc),
            "pr_auc": float(overall_pr_auc),
            "tDCF_min_norm": float(tdcf_overall),
        },
        "gate_weights": {n: float(w) for n, w in zip(expert_names, avg_gate_weights)},
        "tdcf_config": {
            "p_tar": args.tdcf_p_tar,
            "p_non": args.tdcf_p_non,
            "p_spoof": args.tdcf_p_spoof,
            "C_miss_asv": args.tdcf_c_miss_asv,
            "C_fa_asv": args.tdcf_c_fa_asv,
            "C_miss_cm": args.tdcf_c_miss_cm,
            "C_fa_cm": args.tdcf_c_fa_cm,
            "P_miss_asv": args.tdcf_pmiss_asv,
            "P_fa_asv": args.tdcf_pfa_asv,
        },
    }
    (out_dir / "metrics_overall.json").write_text(json.dumps(overall_dict, indent=2))
    pd.DataFrame(rows).sort_values("dataset").to_csv(out_dir / "metrics_per_dataset.csv", index=False)
    pd.DataFrame(per_sample_rows).to_csv(out_dir / "scores.csv", index=False)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Results folder: {out_dir.resolve()}")
    print("  ├─ metrics_overall.json")
    print("  ├─ metrics_per_dataset.csv")
    print("  └─ scores.csv")


if __name__ == "__main__":
    main()
