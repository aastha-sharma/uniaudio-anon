"""
DANN Training for 3-Branch Model (Entropy Reg + Metric Updates)
FEW-SHOT SUPERVISED ADAPTATION VERSION:
1. Limits Singing Data to 10% (Simulating Few-Shot scenario).
2. Unmasks Loss: Enables supervised training on both Speech and Singing.
3. Keeps optional DANN architecture for experimental comparison.
"""
import sys
import os
import atexit
import shutil
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import random
import numpy as np
import pandas as pd
import tempfile
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import roc_curve

# --- LOCAL IMPORTS ---
from unified_training.unified_model_3branch import UnifiedDeepfakeDetector3Branch
from src.data.dataset import DeepfakeDataset

# --- GLOBAL TEMP FILE TRACKER (Fix #1: Memory Leak) ---
temp_files = []

def cleanup_temp_files():
    """Deletes temporary manifest files on exit."""
    for f in temp_files:
        if os.path.exists(f):
            try:
                os.unlink(f)
            except OSError:
                pass

# Register cleanup to run automatically when script exits
atexit.register(cleanup_temp_files)


def compute_eer(labels, scores):
    """Computes Equal Error Rate (EER)."""
    if len(labels) == 0: return 0.0
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


class GradientReversalLayer(torch.autograd.Function):
    """
    Reverses the gradient during backward pass.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class DomainClassifier(nn.Module):
    """
    Simple classifier to distinguish Source (Speech) from Target (Singing).
    """
    def __init__(self, input_dim=512, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1) # Binary: 0=Source, 1=Target
        )
    
    def forward(self, x):
        return self.classifier(x)


class UnifiedDeepfakeDetector3Branch_DANN(nn.Module):
    """
    Wraps the 3-Branch model. Can optionally add Domain Adaptation heads.
    """
    def __init__(self, base_model, use_w2v_only=False):
        super().__init__()
        self.base_model = base_model
        self.use_w2v_only = use_w2v_only
        
        if use_w2v_only:
            self.domain_classifier_w2v_rn = DomainClassifier(input_dim=512)
        else:
            self.domain_classifier_logmel = DomainClassifier(input_dim=512)
            self.domain_classifier_mfcc = DomainClassifier(input_dim=512)
            self.domain_classifier_w2v_rn = DomainClassifier(input_dim=512)
    
    def forward(self, x, lambda_=0.0, return_domain_preds=False):
        # 1. Get features from the base model
        fused_logit, gate_weights, expert_outputs = self.base_model(
            x, return_expert_outputs=True
        )
        
        domain_preds = {}
        
        # 2. Apply Gradient Reversal and Domain Classification (Optional if lambda > 0)
        if self.use_w2v_only:
            w2v_emb = expert_outputs['embeddings']['w2v_rn']
            w2v_emb_reversed = GradientReversalLayer.apply(w2v_emb, lambda_)
            domain_preds['w2v_rn'] = self.domain_classifier_w2v_rn(w2v_emb_reversed)
        else:
            for name in ['logmel', 'mfcc', 'w2v_rn']:
                emb = expert_outputs['embeddings'][name]
                # Apply GRL
                emb_reversed = GradientReversalLayer.apply(emb, lambda_)
                # Predict Domain
                classifier = getattr(self, f'domain_classifier_{name}')
                domain_preds[name] = classifier(emb_reversed)
        
        if return_domain_preds:
            return fused_logit, gate_weights, expert_outputs, domain_preds
        return fused_logit, gate_weights, expert_outputs


class DANNLoss_UDA(nn.Module):
    """
    Loss = Main_Task_Loss + (Lambda * Domain_Loss) + Entropy_Reg
    Modified for Few-Shot Supervised Learning.
    """
    def __init__(self, lambda_aux=0.1, lambda_domain=0.3, lambda_ent=0.01):
        super().__init__()
        self.lambda_aux = lambda_aux
        self.lambda_domain = lambda_domain
        self.lambda_ent = lambda_ent 
        self.bce_none = nn.BCEWithLogitsLoss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, fused_logit, expert_outputs, domain_preds, 
                fake_labels, domain_labels, gate_weights):
        
        fake_labels = fake_labels.float().view(-1, 1)
        domain_labels = domain_labels.float().view(-1, 1)
        
        # --- FIX #4: Variable Naming (source_mask -> supervised_mask) ---
        # We train on ALL provided labels (Speech + 10% Singing).
        supervised_mask = torch.ones_like(domain_labels)
        # -------------------------------------
        
        # 1. Main Task Loss (Supervised on ALL data)
        pixel_loss = self.bce_none(fused_logit, fake_labels)
        main_loss = (pixel_loss * supervised_mask).sum() / (supervised_mask.sum() + 1e-8)
        
        # 2. Auxiliary Losses for Experts
        aux_losses = []
        for name in ['logmel', 'mfcc', 'w2v_rn']:
            aux_loss_raw = self.bce_none(expert_outputs['logits'][name], fake_labels)
            masked_aux_loss = (aux_loss_raw * supervised_mask).sum() / (supervised_mask.sum() + 1e-8)
            aux_losses.append(masked_aux_loss)
        aux_loss_mean = sum(aux_losses) / len(aux_losses)
        
        # 3. Domain Loss (Optional DANN component)
        domain_losses = []
        for name, pred in domain_preds.items():
            domain_loss = self.bce(pred, domain_labels)
            domain_losses.append(domain_loss)
        domain_loss_mean = sum(domain_losses) / len(domain_losses)
        
        # 4. Entropy Regularization (Fix #9: Numerical Stability)
        # Clamp gate weights to avoid log(0)
        gate_weights_safe = torch.clamp(gate_weights, min=1e-8)
        entropy = -torch.sum(gate_weights_safe * torch.log(gate_weights_safe), dim=1).mean()
        entropy_loss = -self.lambda_ent * entropy
        
        # Total Loss
        total = main_loss + \
                self.lambda_aux * aux_loss_mean + \
                self.lambda_domain * domain_loss_mean + \
                entropy_loss
        
        return total, {
            'main': main_loss.item(),
            'aux': aux_loss_mean.item(),
            'domain': domain_loss_mean.item(),
            'ent': entropy_loss.item(),
            'n_supervised': supervised_mask.sum().item()
        }


def create_uda_loader(manifest_path, speech_ratio=0.7, batch_size=32, num_workers=4, split='train'):
    """
    Creates a DataLoader.
    FEW-SHOT IMPLEMENTATION: Uses only 10% of singing data for training.
    """
    df = pd.read_csv(manifest_path, dtype={'used_stem': str}, low_memory=False)
    df_split = df[df['split'] == split].copy()
    
    # Fix #10: Check if split is empty
    if len(df_split) == 0:
        raise ValueError(f"No data found for split={split}. Check manifest.")
    
    # Define Datasets
    speech_datasets = ['asvspoof2019', 'asvspoof2021', 'wavefake', 'in_the_wild']
    singing_datasets = ['svdd', 'singfake', 'ctrsvdd']
    
    df_speech = df_split[df_split['dataset'].isin(speech_datasets)]
    df_singing = df_split[df_split['dataset'].isin(singing_datasets)]
    
    # --- FEW-SHOT LOGIC: DISABLED (Using 100% Data) ---
    if split == 'train' and len(df_singing) > 0:
        print(f"Original Singing Samples: {len(df_singing)}")
        # Use ALL singing data
        df_singing = df_singing 
        print(f"Using ALL Singing Samples (100%): {len(df_singing)}")
    # -------------------------------------------------

    # --- SAMPLING/BALANCING LOGIC ---
    if len(df_singing) > 0:
        if speech_ratio < 1.0:
            n_speech = int(len(df_singing) * speech_ratio / (1.0 - speech_ratio))
            n_speech = min(n_speech, len(df_speech))
        else:
            n_speech = len(df_speech)

        if n_speech > 0 and len(df_speech) > 0:
            df_speech_sampled = df_speech.sample(n=n_speech, random_state=42)
        else:
            df_speech_sampled = df_speech # Fallback
            
        df_singing_sampled = df_singing 
        
    else:
        df_speech_sampled = df_speech
        df_singing_sampled = pd.DataFrame() 
    
    # --- Create Temporary Manifests ---
    def create_temp_ds(dataframe, domain_idx):
        if dataframe is None or len(dataframe) == 0: return None
        # Fix #1: Track temp file for deletion
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            dataframe.to_csv(f.name, index=False)
            temp_files.append(f.name)
            tmp_path = f.name
        
        ds = DeepfakeDataset(tmp_path, split=split)
        
        class DomainWrapper(torch.utils.data.Dataset):
            def __init__(self, dataset, domain_lbl):
                self.dataset = dataset
                self.domain_lbl = domain_lbl
            def __len__(self): return len(self.dataset)
            def __getitem__(self, idx):
                # Robust unpacking
                batch = self.dataset[idx]
                audio = batch[0]
                label = batch[1]
                if audio.dim() > 1 and audio.shape[0] == 1:
                    audio = audio.squeeze(0)
                return audio, label, self.domain_lbl
        
        wrapper = DomainWrapper(ds, domain_idx)
        return wrapper

    speech_ds = create_temp_ds(df_speech_sampled, 0) # Source = 0
    singing_ds = create_temp_ds(df_singing_sampled, 1) # Target = 1
    
    # --- ROBUST CONCATENATION ---
    datasets = []
    if speech_ds is not None: datasets.append(speech_ds)
    if singing_ds is not None: datasets.append(singing_ds)
    
    if not datasets:
        raise ValueError(f"No data found for split={split}. Check manifest.")
        
    if len(datasets) > 1:
        combined = ConcatDataset(datasets)
    else:
        combined = datasets[0]
        
    # shuffle=True ensures mixing without needing custom Sampler
    loader = DataLoader(combined, batch_size=batch_size, shuffle=True, 
                        num_workers=num_workers, pin_memory=True, 
                        drop_last=True) 
    
    return loader


def evaluate_dann(model, dataloader, device, lambda_=0.0):
    model.eval()
    src_labels, src_scores = [], []
    tgt_labels, tgt_scores = [], []
    all_domain_labels, all_domain_preds = [], []
    
    with torch.no_grad():
        # Added progress bar for evaluation
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            audio, labels, domain_labels = batch
            audio = audio.to(device)
            
            fused_logit, _, _, domain_preds = model(
                audio, lambda_=lambda_, return_domain_preds=True
            )
            
            scores = torch.sigmoid(fused_logit).cpu().numpy().ravel()
            labels_np = labels.numpy()
            domain_np = domain_labels.numpy()
            
            # --- Separate Metrics for Source (Speech) and Target (Singing) ---
            src_mask = (domain_np == 0)
            tgt_mask = (domain_np == 1)
            
            if src_mask.any():
                src_labels.extend(labels_np[src_mask].tolist())
                src_scores.extend(scores[src_mask].tolist())
            if tgt_mask.any():
                tgt_labels.extend(labels_np[tgt_mask].tolist())
                tgt_scores.extend(scores[tgt_mask].tolist())

            # --- Domain Confusion Metrics ---
            all_domain_labels.extend(domain_np.tolist())
            domain_pred_avg = torch.stack(list(domain_preds.values())).mean(dim=0)
            domain_pred_probs = torch.sigmoid(domain_pred_avg).cpu().numpy().ravel()
            all_domain_preds.extend(domain_pred_probs.tolist())
    
    metrics = {
        'src_eer': compute_eer(src_labels, src_scores),
        'tgt_eer': compute_eer(tgt_labels, tgt_scores)
    }
    
    if len(all_domain_labels) > 0:
        domain_preds_binary = (np.array(all_domain_preds) > 0.5).astype(int)
        domain_acc = (domain_preds_binary == np.array(all_domain_labels)).mean()
    else:
        domain_acc = 0.0
        
    metrics['domain_acc'] = domain_acc
    metrics['domain_confusion'] = 1.0 - domain_acc
    
    return metrics


def train_stage_dann(model, train_loader, dev_loader, stage, args, device):
    print("\n" + "="*60)
    print(f"FEW-SHOT ADAPTATION STAGE {stage} TRAINING")
    print("="*60)
    
    # --- CONFIGURATION ---
    if stage == 1:
        # Warmup: Initialize heads
        lambda_max = 0.0 # No domain loss initially
        model.base_model.freeze_experts(unfreeze_heads=True)
        num_epochs = args.stage1_epochs
        current_lr = args.lr
        
    elif stage == 2:
        # Adaptation: Unfreeze partially
        lambda_max = args.lambda_domain # User defined strength
        model.base_model.unfreeze_partial() 
        num_epochs = args.stage2_epochs
        current_lr = args.lr 
        
    else:
        # Full Fine-tuning
        lambda_max = args.lambda_domain
        model.base_model.unfreeze_all()
        num_epochs = args.stage3_epochs
        current_lr = args.lr
    
    # Fix #8: Added configurable lambda_aux
    criterion = DANNLoss_UDA(
        lambda_aux=args.lambda_aux, 
        lambda_domain=args.lambda_domain,
        lambda_ent=args.lambda_ent
    )
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=current_lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))
    
    best_tgt_eer = 1.0
    patience = 0
    
    # --- TRAINING LOOP ---
    for epoch in range(num_epochs):
        p = epoch / max(1, num_epochs)
        
        # Fix #3: Negative Lambda from Scheduling
        # Clipping at 0.0 to prevent negative lambda values
        raw_lambda = lambda_max * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)
        lambda_p = max(0.0, raw_lambda)
        
        model.train()
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            audio, labels, domain_labels = batch
            audio = audio.to(device)
            labels = labels.to(device)
            domain_labels = domain_labels.to(device)
            
            optimizer.zero_grad()
            
            fused_logit, gate_weights, expert_outputs, domain_preds = model(
                audio, lambda_=lambda_p, return_domain_preds=True
            )
            
            loss, loss_dict = criterion(
                fused_logit, expert_outputs, domain_preds,
                labels, domain_labels, gate_weights
            )
            
            loss.backward()
            
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({
                'loss': f"{loss.item():.2f}", 
                'src': int(loss_dict['n_supervised']),
            })
        
        # --- EVALUATION ---
        metrics = evaluate_dann(model, dev_loader, device, lambda_=lambda_p)
        avg_loss = np.mean(epoch_losses)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss:       {avg_loss:.4f}")
        print(f"  [SOURCE] EER:     {metrics['src_eer']*100:.2f}%")
        print(f"  [TARGET] EER:     {metrics['tgt_eer']*100:.2f}%")
        
        # Save Best
        if metrics['tgt_eer'] < best_tgt_eer:
            best_tgt_eer = metrics['tgt_eer']
            patience = 0
            save_path = Path(args.output_dir) / f"best_stage{stage}_fewshot.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
                'epoch': epoch
            }, save_path)
            print(f"  ✓ Saved best checkpoint: {save_path}")
        else:
            patience += 1
            print(f"  Patience: {patience}/{args.patience}")
            if patience >= args.patience:
                print("  Early stopping")
                break
    
    return best_tgt_eer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-dir", default="../checkpoints")
    parser.add_argument("--output-dir", default="../checkpoints/few_shot_experiment")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--stage1-epochs", type=int, default=0)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--stage3-epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lambda-aux", type=float, default=0.1)
    parser.add_argument("--lambda-domain", type=float, default=0.1)
    parser.add_argument("--lambda-ent", type=float, default=0.0)
    parser.add_argument("--speech-ratio-s1", type=float, default=1.0)
    parser.add_argument("--speech-ratio-s2", type=float, default=0.5)
    parser.add_argument("--speech-ratio-s3", type=float, default=0.5) 
    parser.add_argument("--use-w2v-only", action='store_true')
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("FEW-SHOT SUPERVISED ADAPTATION (UNFROZEN)")
    print("="*60)
    
    base_model = UnifiedDeepfakeDetector3Branch(checkpoint_dir=args.checkpoint_dir)
    
    if args.base_model:
        print(f"Loading baseline weights from {args.base_model}")
        # Fix #5: weights_only=False and safe loading
        ckpt = torch.load(args.base_model, map_location=device, weights_only=False)
        if 'model_state_dict' in ckpt:
            try:
                base_model.load_state_dict(ckpt['model_state_dict'], strict=False)
            except KeyError:
                base_model.load_state_dict(ckpt['base_model_state_dict'], strict=False)
        else:
            base_model.load_state_dict(ckpt, strict=False)
    
    model = UnifiedDeepfakeDetector3Branch_DANN(
        base_model, 
        use_w2v_only=args.use_w2v_only
    ).to(device)
    
    # --- STAGE EXECUTION LOGIC ---
    
    # STAGE 1 (Frozen Backbone, Train Heads Only)
    if args.stage1_epochs > 0:
        print("\nPreparing Stage 1 (Frozen Backbone)...")
        loader_train = create_uda_loader(args.manifest, speech_ratio=args.speech_ratio_s1, 
                                         batch_size=args.batch_size, split='train')
        loader_dev = create_uda_loader(args.manifest, speech_ratio=args.speech_ratio_s1, 
                                       batch_size=args.batch_size, split='dev')
        train_stage_dann(model, loader_train, loader_dev, 1, args, device)

    # STAGE 2 (Partial Unfreeze)
    if args.stage2_epochs > 0:
        print("\nPreparing Stage 2...")
        loader_train = create_uda_loader(args.manifest, speech_ratio=args.speech_ratio_s2, 
                                         batch_size=args.batch_size, split='train')
        loader_dev = create_uda_loader(args.manifest, speech_ratio=args.speech_ratio_s2, 
                                       batch_size=args.batch_size, split='dev')
        train_stage_dann(model, loader_train, loader_dev, 2, args, device)
        
    # STAGE 3 (Full Unfreeze)
    if args.stage3_epochs > 0:
        print("\nPreparing Stage 3...")
        loader_train = create_uda_loader(args.manifest, speech_ratio=args.speech_ratio_s3, 
                                         batch_size=args.batch_size, split='train')
        loader_dev = create_uda_loader(args.manifest, speech_ratio=args.speech_ratio_s3, 
                                       batch_size=args.batch_size, split='dev')
        train_stage_dann(model, loader_train, loader_dev, 3, args, device)
    
    print("\nTraining Complete.")
    
    # Fix #3: Removed redundant cleanup call; atexit handles it.

if __name__ == "__main__":
    main()