#!/usr/bin/env python3
import argparse
import subprocess
import pandas as pd
import shutil
import tempfile
from pathlib import Path
from tqdm import tqdm

def process_file(input_path: str, output_path: str, device="cuda"):
    output_path = Path(output_path)
    if output_path.exists(): return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try Demucs extraction
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cmd = [
                "demucs", "-n", "htdemucs", "-d", device,
                "--two-stems=vocals", "-o", str(temp_path),
                str(input_path)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            # Locate vocals output
            vocals_src = next(temp_path.rglob("vocals.wav"))
            
            # Convert to 16k mono
            cmd_conv = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(vocals_src),
                "-ac", "1", "-ar", "16000",
                str(output_path)
            ]
            subprocess.run(cmd_conv, check=True)
            return
            
    except Exception as e:
        # Fallback: Convert original mix to 16k mono if separation fails
        try:
            cmd_fallback = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(input_path),
                "-ac", "1", "-ar", "16000",
                str(output_path)
            ]
            subprocess.run(cmd_fallback, check=True)
        except Exception as e2:
            pass # Silent fail if both fail

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    df_vocals = df[df['used_stem'] == 'vocals']

    print(f"Extracting vocals (with fallback) for {len(df_vocals)} files...")
    # Tqdm loop (Demucs is GPU heavy, so we avoid multiprocessing here)
    for _, row in tqdm(df_vocals.iterrows(), total=len(df_vocals)):
        process_file(row['path'], row['proc_path'], device=args.device)

if __name__ == "__main__":
    main()
