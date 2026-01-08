#!/usr/bin/env python3
"""
prepare_ctrsvdd.py

Organizes the CtrSVDD (SVDD Challenge 2024) dataset.
CRITICAL: Handles the segmentation of restricted bonafide data from external corpora 
(JVS-Music, Kiritan, etc.) if provided, as the official CtrSVDD download is incomplete.

Mapping:
  - CtrSVDD train.txt -> 'train' split
  - CtrSVDD dev.txt   -> 'test' split (treated as evaluation in this pipeline)

Usage:
    python tools/prepare_ctrsvdd.py \
        --ctrsvdd_root data/SVDD_Challenge_2024 \
        --output_dir   data/CtrSVDD \
        --external_dir data/external_corpora \
        --timestamps   data/SVDD_Challenge_2024/timestamps
"""

import argparse
import os
import shutil
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm

def segment_audio(src_path: Path, start_time: float, end_time: float, out_path: Path, target_sr=16000) -> bool:
    """Segments audio from external corpus and saves as PCM16 FLAC."""
    try:
        if not src_path.exists():
            return False
            
        if end_time <= start_time:
            return False

        duration = end_time - start_time
        if duration <= 0:
            return False

        # Load specific segment
        X, sr = librosa.load(str(src_path), sr=target_sr, offset=start_time, duration=duration)
        
        if len(X) == 0:
            return False

        # Ensure output directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write with explicit PCM_16 subtype for consistency
        sf.write(str(out_path), X, sr, format='FLAC', subtype='PCM_16')
        return True
    except Exception as e:
        print(f"Error segmenting {src_path.name}: {e}")
        return False

def process_timestamps(timestamp_file: Path, external_root: Path, output_dir: Path) -> int:
    """Reads CtrSVDD timestamp files and generates missing bonafide samples."""
    if not timestamp_file.exists():
        print(f"Skipping timestamps (file not found): {timestamp_file}")
        return 0

    print(f"Processing external segments from: {timestamp_file.name}")
    
    # Robust split detection based on filename
    fname = timestamp_file.name.lower()
    if "_dev" in fname:
        split_dir = "test"  # Map dev timestamps to test split
    elif "_train" in fname:
        split_dir = "train"
    else:
        print(f"Warning: Could not determine split from filename {fname}. Skipping.")
        return 0
    
    count = 0
    with open(timestamp_file, 'r') as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc=f"Gen {split_dir}"):
        parts = line.strip().split('\t')
        if len(parts) < 4: continue
        
        rel_path, new_name, start_str, end_str = parts
        
        # Security: Prevent path traversal
        try:
            # Resolve absolute path and ensure it's inside external_root
            src_path = (external_root / rel_path).resolve()
            if not str(src_path).startswith(str(external_root.resolve())):
                print(f"Security warning: Path traversal attempt detected {rel_path}. Skipping.")
                continue
        except Exception:
            continue

        # Bonafide is implied for these external datasets
        out_path = output_dir / split_dir / "bonafide" / f"{new_name}.flac"
        
        if out_path.exists(): 
            continue
            
        try:
            start, end = float(start_str), float(end_str)
            if segment_audio(src_path, start, end, out_path):
                count += 1
        except ValueError:
            continue
            
    return count

def build_file_index(root_dir: Path):
    """Recursively indexes all FLAC files to handle nested directory structures."""
    print(f"Indexing files in {root_dir}...")
    return {p.stem: p for p in root_dir.rglob("*.flac")}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctrsvdd_root", required=True, type=Path, help="Path to unzipped CtrSVDD data")
    parser.add_argument("--output_dir", required=True, type=Path, help="Target structured directory")
    parser.add_argument("--external_dir", default=None, type=Path, help="Root containing external corpora")
    parser.add_argument("--timestamps", default=None, type=Path, help="Directory containing timestamp txt files")
    args = parser.parse_args()

    root = args.ctrsvdd_root
    dest = args.output_dir

    # 1. Organize Official Provided Data
    # Mapping official metadata files to our pipeline splits
    splits = {
        "train": root / "train.txt",
        "test": root / "dev.txt"  # Official 'dev' becomes 'test'
    }

    print("Step 1: Organizing official CtrSVDD files...")
    
    # Pre-index source files to handle flat or nested structures safely
    # Note: Official download usually splits audio into train_set and dev_set folders
    src_indices = {}
    for folder in ["train_set", "dev_set"]:
        if (root / folder).exists():
            src_indices[folder] = build_file_index(root / folder)

    for split_name, list_file in splits.items():
        if not list_file.exists():
            print(f"Warning: Metadata file not found: {list_file}")
            continue

        with open(list_file, "r") as f:
            lines = f.readlines()

        # Determine which source index to use
        src_key = "train_set" if split_name == "train" else "dev_set"
        current_index = src_indices.get(src_key, {})
        
        missing_in_official = 0
        
        for line in tqdm(lines, desc=f"Moving {split_name}"):
            parts = line.strip().split()
            if len(parts) < 6: continue
            
            # Format: m4singer CtrSVDD_0110 CtrSVDD_0110_D_0015416 ...
            file_id = parts[2]
            label = "bonafide" if parts[5] == "bonafide" else "spoof"
            
            # Lookup file in index
            src = current_index.get(file_id)
            
            if not src:
                missing_in_official += 1
                # Expected for files that need generation from external data
                continue
            
            dst_dir = dest / split_name / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy instead of move to preserve original download
            target_file = dst_dir / src.name
            if not target_file.exists():
                shutil.copy2(src, target_file)
        
        if missing_in_official > 0:
            print(f"  {missing_in_official} files missing from official {src_key} (will attempt generation in Step 2)")

    # 2. Generate Missing Bonafide Data (if external dir provided)
    if args.external_dir and args.timestamps:
        print("\nStep 2: Generating missing bonafide samples from external corpora...")
        
        if not args.timestamps.exists():
             print(f"Timestamp directory not found: {args.timestamps}")
        else:
            generated = 0
            ts_files = list(args.timestamps.glob("*_timestamps_*.txt"))
            
            if not ts_files:
                print("No timestamp files found in provided directory.")
            
            for ts_file in ts_files:
                generated += process_timestamps(ts_file, args.external_dir, args.output_dir)
                
            print(f"Generated {generated} additional bonafide samples.")
    else:
        print("\nSkipping Step 2 (External Generation). Provide --external_dir and --timestamps to generate full CtrSVDD dataset.")

if __name__ == "__main__":
    main()
