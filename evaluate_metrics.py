import argparse
import sys
import os
import torch
import torchaudio
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate

# Ensure python can find project modules
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

try:
    from unified_training.unified_model_3branch import UnifiedDeepfakeDetector3Branch
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)

# --- CUSTOM EVAL DATASET ---
class DeepfakeEvalDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, target_sr=16000, max_len=64000, split='test'):
        super().__init__()
        try:
            df_all = pd.read_csv(manifest_path)
            if 'split' in df_all.columns:
                self.df = df_all[df_all['split'] == split].copy()
                print(f"Loaded {len(self.df)} samples from split='{split}'")
            else:
                self.df = df_all
                print("WARNING: Using ALL data (no split column found)")
        except Exception as e:
            print(f"Error reading manifest {manifest_path}: {e}")
            sys.exit(1)
            
        self.target_sr = target_sr
        self.max_len = max_len 
        
        self.label_map = {
            'bonafide': 0, 'real': 0, '0': 0, 0: 0,
            'spoof': 1, 'fake': 1, 'deepfake': 1, '1': 1, 1: 1
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['path']
        raw_label = row['label']
        
        if isinstance(raw_label, str):
            label = self.label_map.get(raw_label.lower().strip(), 1)
        else:
            label = int(raw_label)

        ds_name = row['dataset'] if 'dataset' in row else "unknown"

        # --- ROBUST AUDIO LOADING ---
        waveform = None
        try:
            if os.path.exists(path):
                waveform, sr = torchaudio.load(path)
            elif 'proc_path' in row and os.path.exists(row['proc_path']):
                waveform, sr = torchaudio.load(row['proc_path'])
            else:
                return None, label, ds_name
        except Exception as e:
            return None, label, ds_name

        # --- PREPROCESSING ---
        try:
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if waveform.numel() == 0:
                return None, label, ds_name

            length = waveform.shape[1]
            if length > self.max_len:
                waveform = waveform[:, :self.max_len]
            elif length < self.max_len:
                pad_amount = self.max_len - length
                waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

            waveform = waveform.squeeze(0)
            return waveform, label, ds_name
            
        except Exception as e:
            return None, label, ds_name

def compute_metrics(labels, scores):
    if len(labels) == 0:
        return {k: 0.0 for k in ["EER", "AUC", "Accuracy", "Precision", "Recall", "F1", "Threshold"]}

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    best_threshold = thresholds[eer_index]

    try:
        auc_score = auc(fpr, tpr)
    except:
        auc_score = 0.0

    preds = [1 if s >= best_threshold else 0 for s in scores]
    
    return {
        "EER": float(eer),
        "AUC": float(auc_score),
        "Accuracy": float(accuracy_score(labels, preds)),
        "Precision": float(precision_score(labels, preds, zero_division=0)),
        "Recall": float(recall_score(labels, preds, zero_division=0)),
        "F1": float(f1_score(labels, preds, zero_division=0)),
        "Threshold": float(best_threshold)
    }

def collate_fn_skip_corrupt(batch):
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

def fix_state_dict_keys(state_dict):
    """
    Strips the 'base_model.' prefix added by the DANN wrapper during training.
    """
    new_dict = {}
    for k, v in state_dict.items():
        if k.startswith('base_model.'):
            new_key = k.replace('base_model.', '')
            new_dict[new_key] = v
        elif 'domain_classifier' in k:
            continue # Skip domain heads not needed for inference
        else:
            new_dict[k] = v
    return new_dict

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*60)
    print(f"EVALUATING MODEL: {Path(args.model).name}")
    print(f"SPLIT: {args.split.upper()}")
    print("="*60)
    
    abs_checkpoint_dir = Path(args.checkpoint_dir).resolve()
    model = UnifiedDeepfakeDetector3Branch(checkpoint_dir=str(abs_checkpoint_dir))
    
    try:
        ckpt = torch.load(args.model, map_location=device, weights_only=False)
        
        # Determine which dict to use
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
            
        # CRITICAL FIX: Strip prefixes
        clean_state_dict = fix_state_dict_keys(state_dict)
        
        # Load with strict=False to ignore missing classifier bias if any, but notify mismatch
        missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
        print("✓ Weights loaded.")
        if missing:
            print(f"  Warning: Missing keys (might be okay if heads differ): {len(missing)}")
        if unexpected:
            print(f"  Warning: Unexpected keys ignored: {len(unexpected)}")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    model.to(device)
    model.eval()

    dataset = DeepfakeEvalDataset(args.manifest, split=args.split)
    
    if len(dataset) == 0:
        print(f"No data found for split '{args.split}'. Check your manifest.")
        return

    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        shuffle=False,
        collate_fn=collate_fn_skip_corrupt
    )

    results = []
    print(f"Processing samples (corrupt files will be skipped)...")
    
    with torch.no_grad():
        for batch in tqdm(loader):
            if batch is None: continue
            
            audio, label, ds_name = batch
            audio = audio.to(device)
            
            try:
                out = model(audio)
                fused_logit = out[0] if isinstance(out, tuple) else out
                scores = torch.sigmoid(fused_logit).cpu().numpy().ravel()
                
                for l, s, d in zip(label.numpy(), scores, ds_name):
                    results.append({"label": int(l), "score": float(s), "dataset": str(d)})
            except Exception as e:
                continue

    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    if len(df) == 0:
        print("No valid samples processed.")
        return

    overall = compute_metrics(df['label'], df['score'])
    print(f"\n[OVERALL] ({len(df)} samples processed)")
    print(f"  EER:        {overall['EER']:.4f} ({overall['EER']*100:.2f}%)")
    print(f"  AUC:        {overall['AUC']:.4f}")
    
    print(f"\n[PER-DATASET BREAKDOWN]")
    per_dataset_metrics = {}
    for ds in sorted(df['dataset'].unique()):
        sub = df[df['dataset'] == ds]
        if len(sub['label'].unique()) < 2: 
            print(f"  {ds:<15}: Skipped (Only one class found)")
            continue
            
        m = compute_metrics(sub['label'], sub['score'])
        print(f"  {ds:<15}: EER={m['EER']*100:.2f}% | AUC={m['AUC']:.4f}")
        per_dataset_metrics[ds] = m

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "model_path": str(args.model),
            "manifest_path": str(args.manifest),
            "split": args.split,
            "total_samples": len(df),
            "overall": overall,
            "per_dataset": per_dataset_metrics
        }
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nFull metrics saved to: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--output", default="results/eval_results.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--split", type=str, default="test", help="Which split to evaluate (train/dev/test)")
    args = parser.parse_args()
    
    evaluate(args)