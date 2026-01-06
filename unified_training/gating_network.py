"""
DANN Training for 3-Branch Model
Trains with domain adversarial loss to improve cross-domain transfer
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from unified_model_3branch import UnifiedDeepfakeDetector3Branch
from src.data.dataset import DeepfakeDataset
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import roc_curve


def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class DomainClassifier(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.classifier(x)


class UnifiedDeepfakeDetector3Branch_DANN(nn.Module):
    def __init__(self, base_model, use_w2v_only=True):
        super().__init__()
        self.base_model = base_model
        self.use_w2v_only = use_w2v_only
        
        # Domain classifiers
        if use_w2v_only:
            self.domain_classifier_w2v = DomainClassifier(input_dim=512)
        else:
            self.domain_classifier_logmel = DomainClassifier(input_dim=512)
            self.domain_classifier_mfcc = DomainClassifier(input_dim=512)
            self.domain_classifier_w2v = DomainClassifier(input_dim=512)
    
    def forward(self, x, lambda_=0.0, return_domain_preds=False):
        # Get expert outputs from base model
        fused_logit, gate_weights, expert_outputs = self.base_model(
            x, return_expert_outputs=True
        )
        
        # Apply gradient reversal and get domain predictions
        domain_preds = {}
        
        if self.use_w2v_only:
            # Only apply DANN to Wav2Vec2 branch
            w2v_emb = expert_outputs['embeddings']['w2v_rn']
            w2v_emb_reversed = GradientReversalLayer.apply(w2v_emb, lambda_)
            domain_preds['w2v_rn'] = self.domain_classifier_w2v(w2v_emb_reversed)
        else:
            # Apply to all branches
            for name in ['logmel', 'mfcc', 'w2v_rn']:
                emb = expert_outputs['embeddings'][name]
                emb_reversed = GradientReversalLayer.apply(emb, lambda_)
                classifier = getattr(self, f'domain_classifier_{name}')
                domain_preds[name] = classifier(emb_reversed)
        
        if return_domain_preds:
            return fused_logit, gate_weights, expert_outputs, domain_preds
        return fused_logit, gate_weights


class DANNLoss(nn.Module):
    def __init__(self, lambda_aux=0.1, lambda_domain=0.3):
        super().__init__()
        self.lambda_aux = lambda_aux
        self.lambda_domain = lambda_domain
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, fused_logit, expert_outputs, domain_preds, 
                fake_labels, domain_labels):
        # Main fake detection loss
        fake_labels = fake_labels.float().view(-1, 1)
        main_loss = self.bce(fused_logit, fake_labels)
        
        # Auxiliary losses for experts
        aux_losses = []
        for name in ['logmel', 'mfcc', 'w2v_rn']:
            aux_loss = self.bce(expert_outputs['logits'][name], fake_labels)
            aux_losses.append(aux_loss)
        aux_loss_mean = sum(aux_losses) / len(aux_losses)
        
        # Domain classification loss
        domain_labels = domain_labels.float().view(-1, 1)
        domain_losses = []
        for name, pred in domain_preds.items():
            domain_loss = self.bce(pred, domain_labels)
            domain_losses.append(domain_loss)
        domain_loss_mean = sum(domain_losses) / len(domain_losses)
        
        # Total loss
        total = main_loss + self.lambda_aux * aux_loss_mean + \
                self.lambda_domain * domain_loss_mean
        
        return total, {
            'main': main_loss.item(),
            'aux': aux_loss_mean.item(),
            'domain': domain_loss_mean.item(),
            'total': total.item()
        }


def create_cross_domain_loader(manifest_path, speech_ratio=0.7, batch_size=32, 
                               num_workers=4, split='train'):
    """Create dataloader mixing speech and singing data"""
    import pandas as pd
    import tempfile
    import os
    
    df = pd.read_csv(manifest_path)
    df_split = df[df['split'] == split].copy()
    
    speech_datasets = ['asvspoof2019', 'asvspoof2021', 'wavefake', 'in_the_wild']
    singing_datasets = ['svdd', 'singfake', 'ctrsvdd']
    
    df_speech = df_split[df_split['dataset'].isin(speech_datasets)]
    df_singing = df_split[df_split['dataset'].isin(singing_datasets)]
    
    # Calculate samples per domain
    if speech_ratio < 1.0:
        total_samples = int(min(len(df_speech), len(df_singing) / (1 - speech_ratio) * speech_ratio))
        n_speech = int(total_samples * speech_ratio)
        n_singing = int(total_samples * (1 - speech_ratio))
        
        df_speech_sampled = df_speech.sample(n=min(n_speech, len(df_speech)), random_state=42)
        df_singing_sampled = df_singing.sample(n=min(n_singing, len(df_singing)), random_state=42)
    else:
        # 100% speech
        df_speech_sampled = df_speech
        df_singing_sampled = pd.DataFrame()
    
    # Create temporary manifests
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        df_speech_sampled.to_csv(f.name, index=False)
        speech_manifest = f.name
    
    # Create datasets
    speech_dataset = DeepfakeDataset(speech_manifest, split=split)
    
    # Add domain labels
    class DomainDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, domain_label):
            self.dataset = dataset
            self.domain_label = domain_label
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            batch = self.dataset[idx]
            audio = batch[0]
            label = batch[1]
            return audio, label, self.domain_label
    
    speech_domain = DomainDataset(speech_dataset, domain_label=0)
    
    if len(df_singing_sampled) > 0:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            df_singing_sampled.to_csv(f.name, index=False)
            singing_manifest = f.name
        
        singing_dataset = DeepfakeDataset(singing_manifest, split=split)
        singing_domain = DomainDataset(singing_dataset, domain_label=1)
        combined = ConcatDataset([speech_domain, singing_domain])
        
        os.unlink(singing_manifest)
        print(f"Created cross-domain loader: {len(df_speech_sampled)} speech + {len(df_singing_sampled)} singing")
    else:
        combined = speech_domain
        print(f"Created speech-only loader: {len(df_speech_sampled)} speech")
    
    loader = DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    os.unlink(speech_manifest)
    
    return loader


def evaluate_dann(model, dataloader, device, lambda_=0.0):
    """Evaluate DANN model"""
    model.eval()
    all_labels = []
    all_scores = []
    all_domain_labels = []
    all_domain_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            audio, labels, domain_labels = batch
            audio = audio.to(device)
            
            fused_logit, _, _, domain_preds = model(
                audio, lambda_=lambda_, return_domain_preds=True
            )
            
            scores = torch.sigmoid(fused_logit).cpu().numpy().ravel()
            all_scores.extend(scores.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_domain_labels.extend(domain_labels.numpy().tolist())
            
            # Average domain predictions
            domain_pred_avg = torch.stack(list(domain_preds.values())).mean(dim=0)
            domain_pred_probs = torch.sigmoid(domain_pred_avg).cpu().numpy().ravel()
            all_domain_preds.extend(domain_pred_probs.tolist())
    
    eer = compute_eer(all_labels, all_scores)
    
    # Domain classification accuracy
    domain_preds_binary = (np.array(all_domain_preds) > 0.5).astype(int)
    domain_acc = (domain_preds_binary == np.array(all_domain_labels)).mean()
    
    return eer, domain_acc


def train_stage_dann(model, train_loader, dev_loader, stage, args, device):
    print("\n" + "="*60)
    print(f"DANN STAGE {stage} TRAINING")
    print("="*60)
    
    # Configure stage
    if stage == 1:
        # Warmup: no DANN, freeze experts except heads and gate
        lambda_max = 0.0
        model.base_model.freeze_experts(unfreeze_heads=True)
        # Unfreeze domain classifiers
        for name, param in model.named_parameters():
            if 'domain_classifier' in name:
                param.requires_grad = True
        num_epochs = args.stage1_epochs
        
    elif stage == 2:
        # Progressive DANN with partial unfreezing
        lambda_max = 1.0
        model.base_model.unfreeze_partial()
        # Ensure domain classifiers are trainable
        for name, param in model.named_parameters():
            if 'domain_classifier' in name:
                param.requires_grad = True
        num_epochs = args.stage2_epochs
        
    else:  # stage 3
        # Full DANN
        lambda_max = 1.0
        model.base_model.unfreeze_all()
        for param in model.parameters():
            param.requires_grad = True
        num_epochs = args.stage3_epochs
    
    criterion = DANNLoss(lambda_aux=0.1, lambda_domain=args.lambda_domain)
    
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))
    
    best_eer = 1.0
    patience = 0
    
    for epoch in range(num_epochs):
        # Compute lambda schedule
        p = epoch / max(1, num_epochs)
        lambda_p = lambda_max * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)
        
        model.train()
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            audio, labels, domain_labels = batch
            audio = audio.to(device)
            labels = labels.to(device)
            domain_labels = domain_labels.to(device)
            
            optimizer.zero_grad()
            
            fused_logit, _, expert_outputs, domain_preds = model(
                audio, lambda_=lambda_p, return_domain_preds=True
            )
            
            loss, loss_dict = criterion(
                fused_logit, expert_outputs, domain_preds,
                labels, domain_labels
            )
            
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.3f}", 'lambda': f"{lambda_p:.3f}"})
        
        # Evaluate
        dev_eer, domain_acc = evaluate_dann(model, dev_loader, device, lambda_=lambda_p)
        
        avg_loss = np.mean(epoch_losses)
        print(f"\nEpoch {epoch+1}/{num_epochs} (lambda: {lambda_p:.3f})")
        print(f"  Train Loss: {avg_loss:.4f} | Dev EER: {dev_eer:.4f} ({dev_eer*100:.2f}%)")
        print(f"  Domain Accuracy: {domain_acc:.4f} ({domain_acc*100:.2f}%)")
        print(f"  Target: ~50% domain acc (domain confusion)")
        
        # Save best
        if dev_eer < best_eer:
            best_eer = dev_eer
            patience = 0
            save_path = Path(args.output_dir) / f"best_stage{stage}_dann.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best checkpoint: {save_path}")
        else:
            patience += 1
            print(f"  Patience: {patience}/{args.patience}")
            if patience >= args.patience:
                print("  Early stopping")
                break
    
    print(f"\nStage {stage} complete. Best EER: {best_eer:.4f}")
    return best_eer


def main():
    parser = argparse.ArgumentParser(description="DANN Training for 3-Branch Model")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-dir", default="../checkpoints")
    parser.add_argument("--output-dir", default="../checkpoints/dann_3branch")
    parser.add_argument("--base-model", default=None, 
                       help="Path to baseline model (if None, starts from pretrained experts)")
    parser.add_argument("--stage1-epochs", type=int, default=10)
    parser.add_argument("--stage2-epochs", type=int, default=25)
    parser.add_argument("--stage3-epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lambda-domain", type=float, default=0.3)
    parser.add_argument("--use-w2v-only", action='store_true',
                       help="Apply DANN only to Wav2Vec2 branch")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("DANN TRAINING FOR 3-BRANCH MODEL")
    print("="*60)
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")
    print(f"DANN on Wav2Vec2 only: {args.use_w2v_only}")
    
    # Initialize base model
    print("\nInitializing base model...")
    base_model = UnifiedDeepfakeDetector3Branch(checkpoint_dir=args.checkpoint_dir)
    
    if args.base_model:
        print(f"Loading baseline weights from {args.base_model}")
        state_dict = torch.load(args.base_model, map_location=device)
        base_model.load_state_dict(state_dict)
    
    # Wrap in DANN
    model = UnifiedDeepfakeDetector3Branch_DANN(
        base_model, 
        use_w2v_only=args.use_w2v_only
    ).to(device)
    
    # Create cross-domain dataloaders
    print("\nCreating cross-domain dataloaders...")
    
    # Stage 1: 100% speech
    train_loader_s1 = create_cross_domain_loader(
        args.manifest, speech_ratio=1.0, batch_size=args.batch_size,
        num_workers=args.num_workers, split='train'
    )
    dev_loader_s1 = create_cross_domain_loader(
        args.manifest, speech_ratio=1.0, batch_size=args.batch_size,
        num_workers=args.num_workers, split='dev'
    )
    
    # Stage 2: 70% speech, 30% singing
    train_loader_s2 = create_cross_domain_loader(
        args.manifest, speech_ratio=0.7, batch_size=args.batch_size,
        num_workers=args.num_workers, split='train'
    )
    dev_loader_s2 = create_cross_domain_loader(
        args.manifest, speech_ratio=0.7, batch_size=args.batch_size,
        num_workers=args.num_workers, split='dev'
    )
    
    # Stage 3: 60% speech, 40% singing
    train_loader_s3 = create_cross_domain_loader(
        args.manifest, speech_ratio=0.6, batch_size=args.batch_size,
        num_workers=args.num_workers, split='train'
    )
    dev_loader_s3 = create_cross_domain_loader(
        args.manifest, speech_ratio=0.6, batch_size=args.batch_size,
        num_workers=args.num_workers, split='dev'
    )
    
    # Train stages
    best_eers = {}
    
    loaders = [
        (train_loader_s1, dev_loader_s1),
        (train_loader_s2, dev_loader_s2),
        (train_loader_s3, dev_loader_s3)
    ]
    
    for stage in [1, 2, 3]:
        train_loader, dev_loader = loaders[stage - 1]
        best_eers[f"stage{stage}"] = train_stage_dann(
            model, train_loader, dev_loader, stage, args, device
        )
    
    # Save final model
    final_path = Path(args.output_dir) / "final_dann_3branch.pth"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_path)
    
    # Save summary
    summary = {
        "model": "3-branch DANN",
        "best_eers": best_eers,
        "timestamp": datetime.now().isoformat(),
        "args": vars(args)
    }
    (Path(args.output_dir) / "training_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    
    print("\n" + "="*60)
    print("DANN TRAINING COMPLETE")
    print("="*60)
    for k, v in best_eers.items():
        print(f"{k} Best EER: {v:.4f} ({v*100:.2f}%)")
    print(f"\nFinal model: {final_path}")


if __name__ == "__main__":
    main()