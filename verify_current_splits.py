# verify_current_splits.py
import pandas as pd

df = pd.read_csv('manifest.csv')

print("="*70)
print("CURRENT SPLIT COMPOSITION")
print("="*70)

# Overall stats
print(f"\nTotal samples: {len(df):,}")
print(f"  Train: {len(df[df['split']=='train']):,}")
print(f"  Dev:   {len(df[df['split']=='dev']):,}")
print(f"  Test:  {len(df[df['split']=='test']):,}")

# Per dataset-label breakdown for TRAIN split
print("\n" + "="*70)
print("TRAIN SPLIT: (dataset, label) composition")
print("="*70)
train_composition = df[df['split']=='train'].groupby(['dataset','label']).size().sort_index()
print(train_composition)

# Check for missing buckets
print("\n" + "="*70)
print("MISSING BUCKETS ANALYSIS")
print("="*70)

all_datasets = ['asvspoof2019', 'asvspoof2021', 'svdd', 'singfake', 'wavefake']
all_labels = ['bonafide', 'spoof']

missing_train = []
for dataset in all_datasets:
    for label in all_labels:
        count = len(df[(df['split']=='train') & (df['dataset']==dataset) & (df['label']==label)])
        if count == 0 and dataset != 'asvspoof2021':  # ASVspoof2021 is test-only by design
            missing_train.append((dataset, label))
            print(f"❌ MISSING in train: ({dataset}, {label})")

if not missing_train:
    print("✅ All dataset-label combinations present in train!")
else:
    print(f"\n⚠️  Found {len(missing_train)} missing buckets in training set")

# Per-dataset split percentages
print("\n" + "="*70)
print("SPLIT PERCENTAGES PER DATASET")
print("="*70)

for dataset in all_datasets:
    subset = df[df['dataset']==dataset]
    total = len(subset)
    if total == 0:
        continue
    
    n_train = len(subset[subset['split']=='train'])
    n_dev = len(subset[subset['split']=='dev'])
    n_test = len(subset[subset['split']=='test'])
    
    print(f"\n{dataset}:")
    print(f"  Total:  {total:,}")
    print(f"  Train:  {n_train:,} ({n_train/total*100:.1f}%)")
    print(f"  Dev:    {n_dev:,} ({n_dev/total*100:.1f}%)")
    print(f"  Test:   {n_test:,} ({n_test/total*100:.1f}%)")
    
    # Check label balance within each split
    for split_name in ['train', 'dev', 'test']:
        split_subset = subset[subset['split']==split_name]
        if len(split_subset) > 0:
            bonafide = len(split_subset[split_subset['label']=='bonafide'])
            spoof = len(split_subset[split_subset['label']=='spoof'])
            print(f"    {split_name}: bonafide={bonafide:,}, spoof={spoof:,}")
