import sys
from pathlib import Path as PathLib
sys.path.insert(0, str(PathLib(__file__).parent.parent))
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import roc_curve

from unified_training.unified_model_3branch import UnifiedDeepfakeDetector3Branch
from src.data.dataset import make_loaders

def entropy_regularization(gate_weights):
    eps = 1e-8
    entropy = -torch.sum(gate_weights * torch.log(gate_weights + eps), dim=1).mean()
    return entropy

def similarity_penalty(expert_embeddings):
    sims = []
    names = list(expert_embeddings.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            emb_a = expert_embeddings[names[i]]
            emb_b = expert_embeddings[names[j]]
            a = F.normalize(emb_a, dim=1)
            b = F.normalize(emb_b, dim=1)
            sims.append((a * b).sum(dim=1).mean())
    if not sims: return torch.tensor(0.0)
    return torch.stack(sims).mean()

class RobustUnifiedLoss(nn.Module):
    def __init__(self, lambda_aux=0.1, lambda_entropy=0.05, lambda_sim=0.1, expert_aux_weights=None):
        super().__init__()
        self.lambda_aux = lambda_aux
        self.lambda_entropy = lambda_entropy
        self.lambda_sim = lambda_sim
        if expert_aux_weights is None:
            expert_aux_weights = [1.2, 1.2, 1.0] 
        self.expert_aux_weights = torch.tensor(expert_aux_weights)

    def forward(self, fused_logit, expert_outputs, gate_weights, labels):
        labels = labels.float().view(-1, 1)
        
        main = F.binary_cross_entropy_with_logits(fused_logit, labels)
        
        names = ["logmel", "mfcc", "w2v_rn"]
        aux_losses = []
        for i, n in enumerate(names):
            l = F.binary_cross_entropy_with_logits(expert_outputs["logits"][n], labels)
            aux_losses.append(l * self.expert_aux_weights[i].to(l.device))
        aux_mean = sum(aux_losses) / len(aux_losses)
        
        ent = entropy_regularization(gate_weights)
        sim = similarity_penalty(expert_outputs["embeddings"])
        
        total = main + (self.lambda_aux * aux_mean) - (self.lambda_entropy * ent) + (self.lambda_sim * sim)
        
        return total, {
            "total": total.item(),
            "main": main.item(),
            "ent": ent.item(),
            "sim": sim.item()
        }

def compute_eer(labels, scores):
    if not labels: return 0.0
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)

def evaluate(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    expert_scores = {k: [] for k in ["logmel", "mfcc", "w2v_rn"]}
    gate_sum = None
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            if len(batch) == 4: audio, labels, _, _ = batch
            else: audio, labels, _ = batch
            
            audio = audio.to(device)
            fused, gates, ex = model(audio, return_expert_outputs=True)
            
            all_scores.extend(torch.sigmoid(fused).cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            
            for k in expert_scores:
                expert_scores[k].extend(torch.sigmoid(ex["logits"][k]).cpu().numpy().tolist())
            
            if gate_sum is None: gate_sum = torch.zeros_like(gates[0])
            gate_sum += gates.sum(dim=0)
            count += audio.size(0)
            
    metrics = {"eer": compute_eer(all_labels, all_scores)}
    for k in expert_scores:
        metrics[f"{k}_eer"] = compute_eer(all_labels, expert_scores[k])
    
    metrics["gate_avg"] = (gate_sum / count).cpu().numpy().tolist() if count > 0 else []
    return metrics

def train_stage(model, train_loader, dev_loader, stage, args, device):
    print(f"\n=== STAGE {stage} TRAINING ===")
    
    if stage == 1:
        model.freeze_experts(unfreeze_heads=True)
        crit = RobustUnifiedLoss(lambda_entropy=0.01, expert_aux_weights=[1.2, 1.2, 1.0]).to(device)
        epochs = args.stage1_epochs
    elif stage == 2:
        model.unfreeze_partial()
        crit = RobustUnifiedLoss(lambda_entropy=0.05, expert_aux_weights=[1.0, 1.0, 1.0]).to(device)
        epochs = args.stage2_epochs
    else:
        model.unfreeze_all()
        crit = RobustUnifiedLoss(lambda_entropy=0.1, expert_aux_weights=[1.0, 1.0, 1.0]).to(device)
        epochs = args.stage3_epochs

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    
    best_eer = 1.0
    patience = 0
    
    for epoch in range(epochs):
        if stage == 2:
            progress = epoch / max(1, epochs)
            new_temp = 1.8 - (0.8 * progress)
            model.gate.set_temperature(new_temp)
            
        model.train()
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs}")
        losses = []
        
        for batch in pbar:
            if len(batch) == 4: audio, labels, _, _ = batch
            else: audio, labels, _ = batch
            
            audio, labels = audio.to(device), labels.to(device)
            optimizer.zero_grad()
            
            fused, gates, ex = model(audio, return_expert_outputs=True)
            loss, d = crit(fused, ex, gates, labels)
            
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            
            losses.append(loss.item())
            pbar.set_postfix({"L": f"{loss.item():.2f}", "Ent": f"{d['ent']:.2f}"})
            
        metrics = evaluate(model, dev_loader, device)
        print(f"Ep {epoch+1} Results:")
        print(f"  Unified EER: {metrics['eer']:.4f}")
        print(f"  Experts: LM {metrics['logmel_eer']:.3f} | MFCC {metrics['mfcc_eer']:.3f} | W2V {metrics['w2v_rn_eer']:.3f}")
        print(f"  Gate Avg: {metrics['gate_avg']}")
        
        if metrics['eer'] < best_eer:
            best_eer = metrics['eer']
            patience = 0
            save_path = Path(args.output_dir) / f"best_stage{stage}_3branch.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  Saved Best: {save_path}")
        else:
            patience += 1
            if patience >= args.patience:
                print("  Early Stopping")
                break
                
    return best_eer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output-dir", default="checkpoints/unified_3branch_speech_only")
    parser.add_argument("--stage1-epochs", type=int, default=10)
    parser.add_argument("--stage2-epochs", type=int, default=25)
    parser.add_argument("--stage3-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("UNIFIED SPEECH BASELINE TRAINING")
    print("="*60)
    
    model = UnifiedDeepfakeDetector3Branch(checkpoint_dir=args.checkpoint_dir, min_gate_weight=0.12).to(device)
    
    train_loader, dev_loader = make_loaders(args.manifest, batch=args.batch_size, workers=args.num_workers)
    
    best = {}
    for stage in [1, 2, 3]:
        if stage == 3 and args.stage3_epochs == 0: continue
        best[f"stage{stage}"] = train_stage(model, train_loader, dev_loader, stage, args, device)
        
    print("\nTraining Complete.")
    for k, v in best.items():
        print(f"{k} Best EER: {v:.4f}")

if __name__ == "__main__":
    main()