# src/train/train_expert.py - COMPLETE WITH WARMUP/UNFREEZE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.data.dataset import make_loaders
from src.features.extractors import LogMel, MFCC


class ResNet18Embed(nn.Module):
    """ResNet-18 encoder that outputs 512-d embeddings"""
    def __init__(self, in_ch=1, out_dim=512):
        super().__init__()
        self.backbone = torchvision.models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            in_ch, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.backbone.fc = nn.Identity()
        self.proj = nn.Linear(512, out_dim)
    
    def forward(self, S):
        h = self.backbone(S)
        return self.proj(h)


class DeepfakeBranch(nn.Module):
    """Single branch in multi-branch deepfake detection system"""
    def __init__(self, kind='logmel'):
        super().__init__()
        self.kind = kind
        
        if kind == 'logmel':
            self.feat = LogMel()
        elif kind == 'mfcc':
            self.feat = MFCC()
        else:
            raise ValueError(f"Unknown kind: {kind}")
        
        self.enc = ResNet18Embed(in_ch=1, out_dim=512)
        self.head = nn.Linear(512, 1)
    
    def forward(self, x):
        S = self.feat(x)
        z = self.enc(S)
        logit = self.head(z)
        return logit.squeeze(1)


def make_model(kind, **kwargs):
    """Create model based on kind"""
    if kind in ['logmel', 'mfcc']:
        return DeepfakeBranch(kind)
        
    elif kind in ['w2v_transformer', 'w2v_tx']:
        from src.models.experts_w2v import Wav2VecTransformerExpert
        return Wav2VecTransformerExpert(
            model_name=kwargs.get('w2v_model_name', 'facebook/wav2vec2-base-960h'),
            freeze_base=False,
            d_model=768,
            nhead=8,
            num_layers=2,
            dropout=0.1
        )
        
    elif kind in ['w2v_resnet', 'w2v_rn']:
        from src.models.experts_w2v import Wav2VecResNetExpert
        return Wav2VecResNetExpert(
            model_name=kwargs.get('w2v_model_name', 'facebook/wav2vec2-base-960h'),
            freeze_base=False,
            dropout=0.0  # Changed to 0.0 for initial training
        )
        
    else:
        raise ValueError(f"Unknown model kind: {kind}")


def compute_eer(labels, scores):
    """Compute Equal Error Rate"""
    from sklearn.metrics import roc_curve
    import numpy as np
    
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return float(eer)


def mc_eval_logits(model, x, crops=5, crop_sec=4, sr=16000):
    """Multi-crop evaluation for stability"""
    T = crop_sec * sr
    logits = []
    for _ in range(crops):
        if x.size(1) > T:
            s = torch.randint(0, x.size(1) - T + 1, (1,), device=x.device).item()
            x_ = x[:, s:s+T]
        else:
            x_ = F.pad(x, (0, max(0, T - x.size(1))))
        logits.append(model(x_))
    return torch.stack(logits, 0).mean(0)


def train(manifest, kind='logmel', epochs=6, batch=32, lr=1e-3, device='cuda', w2v_model_name=None, workers=4):
    """Train a single branch model"""
    device = device if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"TRAINING {kind.upper()} BRANCH - BATTLE-TESTED VERSION")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    if 'w2v' in kind:
        batch = 8
        print(f"Using batch size {batch} for Wav2Vec2")
    
    dl_tr, dl_dev = make_loaders(manifest, batch=batch, workers=workers)
    
    model = make_model(kind, w2v_model_name=w2v_model_name or 'facebook/wav2vec2-base-960h').to(device)
    
    # Initialize bias for w2v models
    if 'w2v' in kind:
        p = 0.5
        with torch.no_grad():
            last = [m for m in model.classifier if isinstance(m, nn.Linear)][-1]
            last.bias.fill_(math.log((p + 1e-6) / (1 - p + 1e-6)))
        print(f"Initialized final layer bias with prior p={p}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  % Trainable: {100*trainable_params/total_params:.1f}%")
    
    if 'w2v' in kind:
        print("\nParam grouping preview (HEAD vs BASE):")
        head_cnt = base_cnt = 0
        for name, p in model.named_parameters():
            tag = "BASE" if name.startswith("wav2vec.wav2vec") else "HEAD"
            if p.requires_grad:
                if tag == "BASE": 
                    base_cnt += p.numel()
                else: 
                    head_cnt += p.numel()
        print(f"Trainable params -> HEAD: {head_cnt:,} | BASE: {base_cnt:,}")
        
        base_params, head_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith('wav2vec.wav2vec'):
                base_params.append(p)
            else:
                head_params.append(p)

        # HEAD FIRST, base tiny LR initially
        opt_groups = []
        opt_groups.append({'params': head_params, 'lr': lr, 'weight_decay': 1e-4})
        if base_params:
            base_lr = max(lr * 1e-2, 3e-6)  # At least 3e-6 initially
            opt_groups.append({'params': base_params, 'lr': base_lr, 'weight_decay': 1e-4})

        opt = torch.optim.AdamW(opt_groups, betas=(0.9, 0.999), eps=1e-8)

        print(f"\nOptimizer: AdamW param groups")
        for i, pg in enumerate(opt.param_groups):
            n_params = sum(p.numel() for p in pg['params'])
            print(f"  Group {i}: lr={pg['lr']:.2e}, params={n_params:,}")
        
        # ===== WARMUP/UNFREEZE CONFIGURATION =====
        WARMUP_STEPS = 0                # 0 = unfreeze immediately
        BASE_LR_AFTER_WARMUP = 1e-4     # real finetune LR for W2V BASE
        HEAD_LR = lr                    # CLI --lr (recommend 3e-4)
        step = 0
        
        print(f"\n🔥 W2V WARMUP CONFIG:")
        print(f"  WARMUP_STEPS: {WARMUP_STEPS}")
        print(f"  HEAD_LR: {HEAD_LR:.2e}")
        print(f"  BASE_LR_AFTER_WARMUP: {BASE_LR_AFTER_WARMUP:.2e}")
        
        scheduler = None
        
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = None
        print(f"Optimizer: Adam, lr={lr}")
    
    print(f"Mixed precision: False")
    
    best_eer = 1.0
    
    def eval_eer():
        model.eval()
        ys, ss = [], []
        
        with torch.no_grad():
            for x, y, _, _ in dl_dev:
                x = x.to(device)
                # Use multi-crop for stable evaluation
                logit = mc_eval_logits(model, x, crops=5)
                score = torch.sigmoid(logit)
                ys.extend(y.cpu().numpy().tolist())
                ss.extend(score.cpu().numpy().ravel().tolist())
        
        eer = compute_eer(ys, ss)
        return eer
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Batches per epoch: {len(dl_tr)}\n")
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        bad_batches = 0
        
        for batch_idx, (x, y, _, _) in enumerate(dl_tr):
            # ===== WARMUP/UNFREEZE LOGIC =====
            if 'w2v' in kind:
                step += 1
                in_warmup = step <= WARMUP_STEPS
                
                # Freeze/unfreeze BASE based on warmup
                for name, p in model.named_parameters():
                    is_base = name.startswith('wav2vec.wav2vec')
                    p.requires_grad = (not is_base) if in_warmup else True
                
                # Unfreeze and boost BASE LR after warmup
                if step == WARMUP_STEPS + 1:
                    opt.param_groups[0]['lr'] = HEAD_LR
                    if len(opt.param_groups) > 1:
                        opt.param_groups[1]['lr'] = BASE_LR_AFTER_WARMUP
                    print(f"\n🔓 > Unfrozen BASE. LRs -> HEAD:{HEAD_LR:.2e}, BASE:{BASE_LR_AFTER_WARMUP:.2e}\n")
            
            # ===== REGULAR TRAINING =====
            x = x.to(device)
            y = y.to(device).float()
            
            logit = model(x)
            
            if not torch.isfinite(logit).all():
                bad_batches += 1
                opt.zero_grad(set_to_none=True)
                continue
            
            # Add pos_weight for class imbalance
            pos = y.mean().clamp(1e-3, 1-1e-3)
            pos_weight = (1 - pos) / pos
            loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
            
            if not torch.isfinite(loss):
                bad_batches += 1
                opt.zero_grad(set_to_none=True)
                continue
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            
            if 'w2v' in kind:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            opt.step()
            
            if scheduler:
                scheduler.step()
            
            epoch_losses.append(loss.item())
            
            # ===== LOGGING =====
            if (batch_idx + 1) % 50 == 0:
                avg_loss = sum(epoch_losses[-50:]) / len(epoch_losses[-50:])
                lrs = ', '.join(f"{pg['lr']:.2e}" for pg in opt.param_groups)
                print(f"  Epoch {epoch} [{batch_idx+1:5d}/{len(dl_tr)}] "
                      f"Loss: {avg_loss:.4f}, LRs: {lrs}")
            
            if (batch_idx + 1) % 200 == 0:
                with torch.no_grad():
                    p = torch.sigmoid(logit)
                    print(f"    logits μ={logit.mean().item():+.3f} σ={logit.std().item():.3f}  "
                          f"probs μ={p.mean().item():.3f}  y μ={y.mean().item():.3f}")
                    
                    # ===== TRAINABLE PARAMS CHECK =====
                    if 'w2v' in kind:
                        base_cnt = sum(p.numel() for n, p in model.named_parameters()
                                      if p.requires_grad and n.startswith('wav2vec.wav2vec'))
                        head_cnt = sum(p.numel() for n, p in model.named_parameters()
                                      if p.requires_grad and not n.startswith('wav2vec.wav2vec'))
                        print(f"    trainable -> HEAD:{head_cnt:,}  BASE:{base_cnt:,}")
            
            if (batch_idx + 1) % 500 == 0 and bad_batches:
                print(f"  Note: skipped {bad_batches} bad batches so far")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        eer = eval_eer()
        
        print(f"\n[{kind}] Epoch {epoch}: Loss={avg_loss:.4f}, Dev EER={eer:.4f} (multi-crop)")
        if bad_batches:
            print(f"  Skipped {bad_batches} bad batches this epoch")
        
        if eer < best_eer:
            improvement = best_eer - eer
            best_eer = eer
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint_path = f'checkpoints/{kind}_pretrained.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✅ Best model saved! Improved by {improvement:.4f}")
        else:
            print(f"  No improvement (best: {best_eer:.4f})")
        
        print()
    
    print(f"{'='*60}")
    print(f"Training complete!")
    print(f"Best dev EER ({kind}): {best_eer:.4f} ({best_eer*100:.2f}%)")
    print(f"{'='*60}\n")
    
    return best_eer


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(description='Train individual deepfake detection branches')
    ap.add_argument("--manifest", required=True, help="Path to manifest.csv")
    ap.add_argument("--kind", 
                    choices=["logmel", "mfcc", "w2v_tx", "w2v_transformer", "w2v_resnet", "w2v_rn"],
                    default="logmel",
                    help="Branch type to train")
    ap.add_argument("--epochs", type=int, default=8, help="Number of epochs")
    ap.add_argument("--batch", type=int, default=32, help="Batch size")
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    ap.add_argument("--w2v_model", type=str, default="facebook/wav2vec2-base-960h",
                    help="Wav2Vec2 model checkpoint")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    
    args = ap.parse_args()
    
    train(
        manifest=args.manifest,
        kind=args.kind,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        w2v_model_name=args.w2v_model,
        workers=args.workers
    )