#!/usr/bin/env python3
import argparse
import torch
import torchaudio
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor

@torch.no_grad()
def extract(path, model, processor, device):
    try:
        wav, sr = torchaudio.load(path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav = resampler(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        input_values = processor(wav.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values
        input_values = input_values.to(device)
        
        # [Batch, Time, 768]
        outputs = model(input_values)
        return outputs.last_hidden_state.cpu()
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Wav2Vec2 model on {device}...")
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    model.eval()

    df = pd.read_csv(args.manifest)
    output_root = Path(args.output_dir)

    print(f"Extracting features for {len(df)} files...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        proc_path = Path(row['proc_path'])
        if not proc_path.exists(): continue

        # Robust relative path extraction:
        # If path is .../derived_16k_mono/singing/train/file.wav -> singing/train/file.pt
        try:
            parts = proc_path.parts
            if "derived_16k_mono" in parts:
                idx = parts.index("derived_16k_mono")
                rel_path = Path(*parts[idx+1:])
            else:
                rel_path = proc_path.name
        except ValueError:
            rel_path = proc_path.name

        save_path = output_root / Path(rel_path).with_suffix(".pt")
        
        if save_path.exists(): continue
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        feats = extract(proc_path, model, processor, device)
        if feats is not None:
            torch.save(feats, save_path)

if __name__ == "__main__":
    main()
