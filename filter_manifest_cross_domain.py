"""
Filter Manifest for Cross-Domain Baseline Testing 
"""
import argparse
import pandas as pd
from pathlib import Path
import json


def filter_manifest_cross_domain(manifest_csv, output_dir):
    """
    Create filtered manifests for cross-domain baseline testing
    
    Creates:
    1. manifest_speech_only.csv - Train+Dev speech (splits preserved)
    2. manifest_test_singing_only.csv - Test singing only
    3. manifest_test_speech_only.csv - Test speech only (for comparison)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("FILTERING MANIFEST FOR CROSS-DOMAIN BASELINE")
    print("="*60)
    
    # Load full manifest
    df = pd.read_csv(manifest_csv, dtype={'used_stem': str})
    print(f"\nTotal samples in manifest: {len(df):,}")
    
    # Define speech and singing datasets
    speech_datasets = ['asvspoof2019', 'asvspoof2021', 'wavefake', 'in_the_wild']
    singing_datasets = ['svdd', 'singfake', 'ctrsvdd']
    
    print("\n" + "="*60)
    print("ORIGINAL DATASET DISTRIBUTION")
    print("="*60)
    print("\nBy Dataset and Split:")
    dataset_counts = df.groupby(['dataset', 'split']).size().unstack(fill_value=0)
    print(dataset_counts)
    
    # 1. SPEECH-ONLY MANIFEST (Train + Dev, splits preserved)
    print("\n" + "="*60)
    print("CREATING SPEECH-ONLY MANIFEST (Train + Dev)")
    print("="*60)
    
    # Filter for Speech datasets AND (Train or Dev) splits
    df_speech = df[
        (df['dataset'].isin(speech_datasets)) & 
        (df['split'].isin(['train', 'dev']))
    ].copy()
    
    print(f"\nTotal speech samples: {len(df_speech):,}")
    print("\nBreakdown by split (CRITICAL CHECK):")
    print(df_speech.groupby('split').size())
    print("\nBreakdown by dataset and split:")
    print(df_speech.groupby(['dataset', 'split']).size())
    
    # Save the file - 'split' column is strictly preserved
    speech_manifest_path = output_dir / "manifest_speech_only.csv"
    df_speech.to_csv(speech_manifest_path, index=False)
    print(f"\n✓ Saved: {speech_manifest_path}")
    
    # 2. SINGING TEST MANIFEST
    print("\n" + "="*60)
    print("CREATING SINGING-ONLY TEST MANIFEST")
    print("="*60)
    
    df_test_singing = df[
        (df['split'] == 'test') & 
        (df['dataset'].isin(singing_datasets))
    ].copy()
    
    print(f"\nSinging test samples: {len(df_test_singing):,}")
    
    singing_test_path = output_dir / "manifest_test_singing_only.csv"
    df_test_singing.to_csv(singing_test_path, index=False)
    print(f"\n✓ Saved: {singing_test_path}")
    
    # 3. SPEECH TEST MANIFEST (for comparison)
    print("\n" + "="*60)
    print("CREATING SPEECH-ONLY TEST MANIFEST")
    print("="*60)
    
    df_test_speech = df[
        (df['split'] == 'test') & 
        (df['dataset'].isin(speech_datasets))
    ].copy()
    
    print(f"\nSpeech test samples: {len(df_test_speech):,}")
    
    speech_test_path = output_dir / "manifest_test_speech_only.csv"
    df_test_speech.to_csv(speech_test_path, index=False)
    print(f"\n✓ Saved: {speech_test_path}")
    
    # SUMMARY
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    train_count = len(df_speech[df_speech['split'] == 'train'])
    dev_count = len(df_speech[df_speech['split'] == 'dev'])
    
    print("\n1. Training Manifest (Speech only, splits preserved):")
    print(f"   File: {speech_manifest_path}")
    print(f"   Train samples: {train_count:,}")
    print(f"   Dev samples:   {dev_count:,}")
    
    print("\n2. Singing Test Manifest:")
    print(f"   File: {singing_test_path}")
    print(f"   Samples: {len(df_test_singing):,}")
    
    # Save metadata
    info = {
        'manifests': {
            'speech_train': str(speech_manifest_path),
            'test_singing': str(singing_test_path),
            'test_speech': str(speech_test_path)
        }
    }
    
    info_path = output_dir / "cross_domain_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"\n✓ Saved metadata: {info_path}")
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Filter manifest for cross-domain baseline testing")
    parser.add_argument("--manifest", required=True, help="Path to original manifest.csv")
    parser.add_argument("--output-dir", default="./data/cross_domain", help="Output directory")
    args = parser.parse_args()
    
    filter_manifest_cross_domain(args.manifest, args.output_dir)


if __name__ == "__main__":
    main()