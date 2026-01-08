#!/usr/bin/env python3
import argparse
import subprocess
import csv
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

FAILURE_LOG = "preprocess_failures.csv"

def convert_to_16k_mono(args):
    input_path, output_path = args
    output_path = Path(output_path)
    
    if output_path.exists():
        return (True, None)
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.wav")
    
    try:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(input_path),
            "-ac", "1", "-ar", "16000",
            "-sample_fmt", "s16",
            str(tmp_path)
        ]
        subprocess.run(cmd, check=True)
        tmp_path.replace(output_path)
        return (True, None)
    except Exception as e:
        if tmp_path.exists(): tmp_path.unlink()
        return (False, str(e))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to manifest.csv")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    # Filter for 'mix' stem type; vocals are handled by extract_vocals.py
    tasks = [(row['path'], row['proc_path']) for _, row in df.iterrows() if row['used_stem'] != 'vocals']

    print(f"Processing {len(tasks)} files (16k mono conversion)...")
    
    failures = []
    with Pool(args.workers) as p:
        for i, result in tqdm(enumerate(p.imap(convert_to_16k_mono, tasks)), total=len(tasks)):
            success, err = result
            if not success:
                failures.append((tasks[i][0], err))

    if failures:
        print(f"\nWARNING: {len(failures)} files failed. Logged to {FAILURE_LOG}")
        with open(FAILURE_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["input_path", "error"])
            writer.writerows(failures)
    else:
        print("\nAll files processed successfully.")

if __name__ == "__main__":
    main()
