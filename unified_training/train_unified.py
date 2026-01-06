"""
 3-Branch Unified Training
Experts: LogMel + MFCC + W2V-ResNet

Features:
1. Staged entropy (0 -> 1e-4 in Stage 2) to prevent early gate collapse.
2. Per-expert aux weights (Boost W2V initially to ensure it learns).
3. Temperature annealing (1.8 -> 1.2) to sharpen expert selection over time.
4. Per-expert diagnostics to monitor contribution and performance.
"""
import sys
from pathlib import Path as PathLib
# Ensure the root directory is in python path
sys.path.insert(0, str(PathLib(__file__).parent.parent))

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import roc_curve

# Import your specific 3-branch model
from unified_model_3branch import UnifiedDeepfakeDetector3Branch
from src.data.dataset import make_loaders


# --- Loss Component Helpers ---

def entropy_regularization(gate_weights):
    """Encourages the gate to not be too confident (high entropy) initially."""
    eps = 1e-8
    entropy = -(gate_weights * torch.log(gate_weights + eps)).sum(dim=1).mean()
    return entropy


def diversity_penalty(expert_embeddings):
    """Penalizes cosine similarity between different expert embeddings."""
    sims = []
    # Compare LogMel vs MFCC, LogMel vs W2V, MFCC vs W2V
    for i in range(len(expert_embeddings)):
        for j in range(i + 1, len(expert_embeddings)):
            a = F.normalize(expert_embeddings[i], dim=1)
            b = F.normalize(expert_embeddings[j], dim=1)
            sims.append((a * b).sum(dim=1).mean())
    if not sims:
        return torch.tensor(0.0, device=expert_embeddings[0].device)
    return torch.stack(sims).mean()


class ImprovedUnifiedLoss3Branch(nn.Module):
    def __init__(self, lambda_aux=0.1, lambda_entropy=0.0, lambda_diversity=0.1, expert_aux_weights=None):
        super().__init__()
        self.lambda_aux = float(lambda_aux)
        self.lambda_entropy = float(lambda_entropy)
        self.lambda_diversity = float(lambda_diversity)
        
        # Default weights: Boost W2V-ResNet (index 2) initially because it learns slower than CNNs
        if expert_aux_weights is None:
            expert_aux_weights = [1.0, 1.0, 1.5] 
        self.expert_aux_weights = torch.tensor(expert_aux_weights, dtype=torch.float32)

    def update_entropy_lambda(self, new_lambda: float):
        self.lambda_entropy = float(new_lambda)

    def update_expert_weights(self, new_weights):
        self.expert_aux_weights = torch.tensor(list(new_weights), dtype=torch.float32)

    def forward(self, fused_logit, expert_outputs, gate_weights, labels):
        labels = labels.float().view(-1, 1)

        # 1. Main Task Loss
        main = F.binary_cross_entropy_with_logits(fused_logit, labels)

        # 2. Auxiliary Losses (Weighted per expert)
        names = ["logmel", "mfcc", "w2v_rn"]
        aux_terms = []
        for i, n in enumerate(names):
            if n in expert_outputs["logits"]:
                aux = F.binary_cross_entropy_with_logits(expert_outputs["logits"][n], labels)
                aux_terms.append(self.expert_aux_weights[i].to(aux.device) * aux)
        
        aux_mean = sum(aux_terms) / len(aux_terms) if aux_terms else torch.tensor(0.0).to(main.device)

        # 3. Regularization
        ent = entropy_regularization(gate_weights)
        div = diversity_penalty([expert_outputs["embeddings"][n] for n in names])

        # Total Loss formulation:
        # We subtract entropy because we want to MAXIMIZE entropy (make distribution wider) early on,
        # but standard optimization minimizes loss. So Min(-Entropy) = Max(Entropy).
        total = main + self.lambda_aux * aux_mean - self.lambda_entropy * ent + self.lambda_diversity * div

        return total, {
            "total": float(total.item()),
            "main": float(main.item()),
            "aux": float(aux_mean.item()),
            "entropy": float(ent.item()),
            "diversity": float(div.item()),
        }


# --- Evaluation Helpers ---

def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def compute_per_expert_metrics(model, dataloader, device):
    """Calculates EER for each expert individually to monitor their health."""
    model.eval()
    all_labels = []
    expert_scores = {k: [] for k in ["logmel", "mfcc", "w2v_rn"]}
    gate_sum = None
    n = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing per-expert metrics", leave=False):
            if len(batch) >= 2:
                audio, labels = batch[0], batch[1]
            audio = audio.to(device)

            _, gate_weights, ex = model(audio, return_expert_outputs=True)

            for k in expert_scores.keys():
                s = torch.sigmoid(ex["logits"][k]).cpu().numpy().ravel().tolist()
                expert_scores[k].extend(s)

            gw = gate_weights.detach()
            if gate_sum is None:
                gate_sum = torch.zeros_like(gw[0])
            gate_sum = gate_sum + gw.sum(dim=0)

            n += audio.size(0)
            all_labels.extend(labels.numpy().tolist())

    metrics = {f"{k}_eer": compute_eer(all_labels, v) for k, v in expert_scores.items()}
    metrics["gate_weights"] = (gate_sum / n).cpu().numpy().tolist() if n > 0 else []
    return metrics


def evaluate(model, dataloader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    all_labels, all_scores = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            if len(batch) >= 2:
                audio, labels = batch[0], batch[1]
            audio = audio.to(device)
            labels_dev = labels.to(device)

            fused_logit, gate_weights, ex = model(audio, return_expert_outputs=True)

            if criterion is not None:
                loss, _ = criterion(fused_logit, ex, gate_weights, labels_dev)
                total_loss += loss.item()

            scores = torch.sigmoid(fused_logit).cpu().numpy().ravel().tolist()
            all_scores.extend(scores)
            all_labels.extend(labels.numpy().tolist())

    eer = compute_eer(all_labels, all_scores)
    avg_loss = total_loss / max(1, len(dataloader)) if criterion else 0.0
    return eer, avg_loss


# --- Training Loop ---

def train_stage(model, train_loader, dev_loader, stage, args, device):
    print("\n" + "=" * 60)
    print(f"STAGE {stage} TRAINING (3 Branches)")
    print("=" * 60)

    # 1. Configuration per Stage
    if stage == 1:
        # Warmup: Freeze backbones, train heads + gating.
        # No entropy yet (let it pick best experts), boost W2V aux weight.
        model.freeze_experts(unfreeze_heads=True)
        lambda_ent = 0.0
        aux_w = [1.0, 1.0, 1.5] 
    elif stage == 2:
        # Alignment: Unfreeze top layers. 
        # Start temperature annealing.
        model.unfreeze_partial()
        lambda_ent = 0.0 # Will be enabled at epoch 5
        aux_w = [1.0, 1.0, 1.0] # Equalize weights
    else:
        # Refinement: Unfreeze everything.
        # Entropy active to maintain diversity.
        model.unfreeze_all()
        lambda_ent = 1e-4
        aux_w = [1.0, 1.0, 1.0]

    # 2. Loss & Optimizer
    crit = ImprovedUnifiedLoss3Branch(
        lambda_aux=0.1,
        lambda_entropy=lambda_ent,
        lambda_diversity=0.1,
        expert_aux_weights=aux_w,
    ).to(device)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    num_epochs = {1: args.stage1_epochs, 2: args.stage2_epochs, 3: args.stage3_epochs}[stage]
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))

    best_eer = 1.0
    patience = 0

    # 3. Epoch Loop
    for epoch in range(num_epochs):
        # -- Schedule Updates --
        
        # Temperature Annealing (Stage 2 only)
        if stage == 2:
            prog = epoch / max(1, num_epochs)
            new_temp = 1.8 - 0.6 * prog # Decays from 1.8 to 1.2
            model.gate.set_temperature(new_temp)

        # Enable Entropy Regularization (Stage 2, Epoch 5)
        if stage == 2 and epoch == 5:
            crit.update_entropy_lambda(1e-4)
            print("  -> Enabled entropy regularization: 1e-4")

        # Decay W2V Boost (Stage 1, Epoch 3)
        if stage == 1 and epoch == 3:
            crit.update_expert_weights([1.0, 1.0, 1.0])
            print("  -> Decayed W2V-ResNet aux boost to 1.0")

        # -- Train Step --
        model.train()
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            if len(batch) >= 2:
                audio, labels = batch[0], batch[1]
            audio = audio.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            fused_logit, gate_weights, ex = model(audio, return_expert_outputs=True)
            loss, loss_dict = crit(fused_logit, ex, gate_weights, labels)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "ent": f"{loss_dict['entropy']:.4f}"})

        # -- Eval Step --
        dev_eer, dev_loss = evaluate(model, dev_loader, device, crit)
        
        # Periodic Expert Check (Every 2 epochs)
        per_exp = compute_per_expert_metrics(model, dev_loader, device) if (epoch % 2 == 0) else None

        avg_train = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        temp_read = model.gate.get_temperature()
        
        print(f"\nEpoch {epoch+1}/{num_epochs} (Temp: {temp_read:.2f})")
        print(f"  Train Loss: {avg_train:.4f} | Dev EER: {dev_eer:.4f} ({dev_eer*100:.2f}%)")

        if per_exp:
            print("  Per-Expert EERs:")
            for n in ["logmel", "mfcc", "w2v_rn"]:
                print(f"    {n.upper()}: {per_exp[f'{n}_eer']:.4f}")
            print(f"  Gate weights (avg): {[f'{w:.2f}' for w in per_exp['gate_weights']]}")

        # -- Checkpoint --
        if dev_eer < best_eer:
            best_eer = dev_eer
            patience = 0
            out = Path(args.output_dir) / f"best_stage{stage}_3branch.pth"
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out)
            print(f"  ✓ Saved best checkpoint: {out}")
        else:
            patience += 1
            print(f"  Patience: {patience}/{args.patience}")
            if patience >= args.patience:
                print("  Early stopping.")
                break

    print(f"\nStage {stage} complete. Best EER: {best_eer:.4f}")
    return best_eer


def main():
    parser = argparse.ArgumentParser(description="Unified Training (3 branches)")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Where expert weights are")
    parser.add_argument("--output-dir", default="./checkpoints/unified_3branch")
    parser.add_argument("--stage1-epochs", type=int, default=10)
    parser.add_argument("--stage2-epochs", type=int, default=25)
    parser.add_argument("--stage3-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("UNIFIED TRAINING (3 BRANCHES)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")
    print(f"Expert Weights: {args.checkpoint_dir}")
    print(f"Stages: 1({args.stage1_epochs}) -> 2({args.stage2_epochs}) -> 3({args.stage3_epochs})")

    print("\nInitializing model...")
    # Load the 3-branch model with pre-trained expert weights
    model = UnifiedDeepfakeDetector3Branch(checkpoint_dir=args.checkpoint_dir).to(device)

    print("\nLoading data...")
    train_loader, dev_loader = make_loaders(
        args.manifest,
        batch=args.batch_size,
        workers=args.num_workers,
    )

    best = {}
    for stage in [1, 2, 3]:
        best[f"stage{stage}"] = train_stage(model, train_loader, dev_loader, stage, args, device)

    # Save Final
    final_path = Path(args.output_dir) / "final_unified_3branch.pth"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_path)

    # Save Summary
    summary = {
        "model": "3-branch (LogMel + MFCC + W2V-ResNet)",
        "best_eers": best,
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
    }
    (Path(args.output_dir) / "training_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    for k, v in best.items():
        print(f"{k} Best EER: {v:.4f} ({v*100:.2f}%)")
    print(f"\nFinal model saved to {final_path}")


if __name__ == "__main__":
    main()