#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import subprocess
import re
from pathlib import Path
from tqdm import tqdm

def safe_filename(name):
    # Remove non-alphanumeric chars to prevent OS issues
    return re.sub(r'[^\w\-_]', '_', name)

def download_file(url, output_path):
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        cmd = [
            'yt-dlp',
            '--extract-audio',
            '--audio-format', 'wav',
            '--postprocessor-args', '-ar 16000 -ac 1',
            '--output', str(output_path),
            url
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output_dir", default="data/SingFake")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    base_dir = Path(args.output_dir)

    print(f"Downloading {len(df)} files for SingFake...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Map raw split names to standard dev/test scheme if needed
        split = row['Set']
        if split.lower() in ['validation', 'val']: split = 'dev'
        
        label = "bonafide" if row['Bonafide Or Spoof'].lower() == 'bonafide' else "spoof"
        
        clean_singer = safe_filename(str(row['Singer']))
        clean_title = safe_filename(str(row['Title']))
        filename = f"{clean_singer}_{clean_title}.wav"
        
        save_path = base_dir / split / label / filename
        download_file(row['Url'], save_path)

if __name__ == "__main__":
    main()
