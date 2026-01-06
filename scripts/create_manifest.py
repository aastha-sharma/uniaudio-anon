# create_manifest.py - FINAL COMPLETE VERSION
import os
import pandas as pd
from pathlib import Path
import librosa
from tqdm import tqdm

DATASET_PATHS = {
    'asvspoof2019': '/data/aastha/datasets/asvspoof2019_la/LA',
    'asvspoof2021': '/data/aastha/datasets/asvspoof2021',
    'svdd': '/data/aastha/datasets/svdd_singing',
    'singfake': '/data/aastha/datasets/singfake',
    'wavefake': '/data/aastha/datasets/wavefake'
}

OUTPUT_DIR = '/data/aastha/derived_16k_mono'
ASVSPOOF2019_PROTOCOL_DIR = '/data/aastha/datasets/asvspoof2019_la/ASVspoof2019_LA_cm_protocols'
MANIFEST_OUTPUT_PATH = '/data/aastha/manifest.csv'


def extract_entity_id(dataset_name, audio_path):
    """
    Extract singer/speaker ID for disjoint splits
    
    SVDD: CtrSVDD_0052_T_0010761.wav → "0052"
    SingFake: Angela_Chang_xxx.wav → "Angela_Chang"
    WaveFake: LJ001-0001.wav → "LJ001"
              BASIC5000_0001_gen.wav → "BASIC5000"
    """
    filename = audio_path.stem
    
    if dataset_name == 'svdd':
        parts = filename.split('_')
        if len(parts) >= 2 and parts[0] == 'CtrSVDD':
            return parts[1]
        return filename.split('_')[0] if '_' in filename else filename
    
    elif dataset_name == 'singfake':
        parts = filename.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return parts[0] if parts else filename
    
    elif dataset_name == 'wavefake':
        if '-' in filename:
            return filename.split('-')[0]
        elif '_' in filename:
            return filename.split('_')[0]
        return filename
    
    return "na"


def grouped_split(entity_ids, train_ratio=0.70, dev_ratio=0.15, test_ratio=0.15):
    """
    Create disjoint train/dev/test splits based on entity IDs
    No entity appears in multiple splits
    """
    unique_entities = sorted(set(entity_ids))
    n_entities = len(unique_entities)
    
    n_train = int(n_entities * train_ratio)
    n_dev = int(n_entities * dev_ratio)
    
    entity_to_split = {}
    for idx, entity in enumerate(unique_entities):
        if idx < n_train:
            entity_to_split[entity] = 'train'
        elif idx < n_train + n_dev:
            entity_to_split[entity] = 'dev'
        else:
            entity_to_split[entity] = 'test'
    
    return entity_to_split


def parse_asvspoof_protocol(protocol_path):
    """Parse ASVspoof protocol file"""
    file_to_label = {}
    
    if not os.path.exists(protocol_path):
        return None
    
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                filename = parts[1]
                label = parts[-1]
                file_to_label[filename] = label
    
    return file_to_label


def load_asvspoof2019_splits():
    """Load ASVspoof 2019 official splits"""
    protocol_files = {
        'train': 'ASVspoof2019.LA.cm.train.trn.txt',
        'dev': 'ASVspoof2019.LA.cm.dev.trl.txt',
        'eval': 'ASVspoof2019.LA.cm.eval.trl.txt'
    }
    
    file_info = {}
    
    for split_name, protocol_file in protocol_files.items():
        protocol_path = os.path.join(ASVSPOOF2019_PROTOCOL_DIR, protocol_file)
        
        print(f"  Loading: {protocol_file}")
        file_to_label = parse_asvspoof_protocol(protocol_path)
        
        if file_to_label is None:
            continue
        
        our_split = 'test' if split_name == 'eval' else split_name
        
        for filename, label in file_to_label.items():
            file_info[filename] = {
                'split': our_split,
                'label': label
            }
        
        print(f"    ✅ Loaded {len(file_to_label)} files")
    
    print(f"  ✅ Total: {len(file_info)} files")
    return file_info


def get_label_from_path(file_path):
    """Extract label from directory structure"""
    path_str = str(file_path).lower()
    if 'bonafide' in path_str:
        return 'bonafide'
    elif 'spoof' in path_str:
        return 'spoof'
    return 'unknown'


def get_duration(file_path):
    """Get audio duration in seconds"""
    try:
        return librosa.get_duration(path=str(file_path))
    except:
        return None


def create_manifest():
    """Main function to create manifest with proper disjoint splits"""
    manifest_data = []
    
    for dataset_name, dataset_path in DATASET_PATHS.items():
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name}...")
        print(f"{'='*60}")
        
        if not os.path.exists(dataset_path):
            print(f"⚠️ Path not found: {dataset_path}")
            continue
        
        # Load ASVspoof 2019 official splits
        asvspoof2019_splits = None
        if dataset_name == 'asvspoof2019':
            asvspoof2019_splits = load_asvspoof2019_splits()
        
        # Get all audio files
        audio_files = []
        for ext in ['.flac', '.wav']:
            audio_files.extend(Path(dataset_path).rglob(f'*{ext}'))
        
        audio_files = sorted(audio_files)
        print(f"Found {len(audio_files)} audio files")
        
        # For datasets requiring disjoint splits
        entity_to_split = None
        if dataset_name in ['svdd', 'singfake', 'wavefake']:
            print(f"  Creating singer/speaker-disjoint splits...")
            
            # Extract entity IDs
            entity_ids = [extract_entity_id(dataset_name, p) for p in audio_files]
            
            # Create disjoint splits
            entity_to_split = grouped_split(entity_ids)
            
            # Statistics
            unique_entities = set(entity_ids)
            split_counts = {}
            for entity in unique_entities:
                split = entity_to_split[entity]
                split_counts[split] = split_counts.get(split, 0) + 1
            
            print(f"  ✅ Total unique entities: {len(unique_entities)}")
            print(f"     Train: {split_counts.get('train', 0)} entities")
            print(f"     Dev: {split_counts.get('dev', 0)} entities")
            print(f"     Test: {split_counts.get('test', 0)} entities")
            
            # Show sample entity IDs
            sample_entities = sorted(unique_entities)[:5]
            print(f"  Sample entity IDs: {sample_entities}")
        
        # Track messages
        asvspoof2021_msg_shown = False
        
        # Process each file
        for idx, audio_path in enumerate(tqdm(audio_files, desc=f"{dataset_name}")):
            filename_no_ext = audio_path.stem
            
            # Determine split and label
            if dataset_name == 'asvspoof2019':
                if asvspoof2019_splits and filename_no_ext in asvspoof2019_splits:
                    split = asvspoof2019_splits[filename_no_ext]['split']
                    label = asvspoof2019_splits[filename_no_ext]['label']
                else:
                    continue
            
            elif dataset_name == 'asvspoof2021':
                split = 'test'
                label = get_label_from_path(audio_path)
                
                if not asvspoof2021_msg_shown:
                    print(f"\n  ℹ️  All files assigned to TEST set")
                    asvspoof2021_msg_shown = True
            
            elif dataset_name in ['svdd', 'singfake', 'wavefake']:
                # Use disjoint splits
                entity_id = extract_entity_id(dataset_name, audio_path)
                split = entity_to_split[entity_id]
                label = get_label_from_path(audio_path)
            
            else:
                continue
            
            if label == 'unknown':
                continue
            
            # Get duration
            duration = get_duration(audio_path)
            if duration is None or duration < 1.0:
                continue
            
            # Output path
            relative_path = audio_path.relative_to(dataset_path)
            proc_path = Path(OUTPUT_DIR) / dataset_name / relative_path.with_suffix('.wav')
            
            # Vocals stem flag
            used_stem = 'vocals' if dataset_name in ['svdd', 'singfake'] else None
            
            manifest_data.append({
                'path': str(audio_path),
                'label': label,
                'dataset': dataset_name,
                'duration': round(duration, 2),
                'split': split,
                'proc_path': str(proc_path),
                'used_stem': used_stem
            })
    
    # Create DataFrame
    df = pd.DataFrame(manifest_data)
    
    # Statistics
    print(f"\n{'='*60}")
    print("MANIFEST STATISTICS")
    print(f"{'='*60}")
    print(f"\n📊 Total files: {len(df)}")
    
    print(f"\n📊 By dataset and split:")
    print(df.groupby(['dataset', 'split']).size())
    
    print(f"\n📊 Training data:")
    train_df = df[df['split'] == 'train']
    if len(train_df) > 0:
        print(train_df.groupby(['dataset', 'label']).size())
        print(f"Total: {len(train_df)}")
    
    print(f"\n📊 Testing data:")
    test_df = df[df['split'] == 'test']
    if len(test_df) > 0:
        print(test_df.groupby(['dataset', 'label']).size())
        print(f"Total: {len(test_df)}")
    
    # Important notes
    print(f"\n{'='*60}")
    print("CRITICAL: DISJOINT SPLITS ENFORCED")
    print(f"{'='*60}")
    print("✅ SVDD: Split by singer ID (CtrSVDD_XXXX)")
    print("✅ SingFake: Split by singer name (Angela_Chang, Bella_Yao, etc.)")
    print("✅ WaveFake: Split by speaker ID (LJ001, BASIC5000, etc.)")
    print("✅ No singer/speaker appears in both train and test!")
    print("\n✅ ASVspoof 2019: Official protocol splits")
    print("✅ ASVspoof 2021: All test set (standard practice)")
    
    # Save to data folder
    df.to_csv(MANIFEST_OUTPUT_PATH, index=False)
    print(f"\n✅ Manifest saved to: {MANIFEST_OUTPUT_PATH}")
    
    return df


if __name__ == "__main__":
    print("="*60)
    print("Creating Manifest with Disjoint Splits")
    print("="*60)
    df = create_manifest()
    print("\n✅ Ready for: Audio preprocessing")
