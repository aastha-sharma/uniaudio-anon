# validate_manifest.py - Enhanced with safety checks
import pandas as pd
import os
from pathlib import Path

print("Reading manifest from:", os.path.realpath('manifest.csv'))
print("=" * 60)

df = pd.read_csv('manifest.csv', dtype={'used_stem': str})

# ============================================================
# SAFETY CHECK 1: Required Columns
# ============================================================
REQUIRED = ["path", "proc_path", "label", "dataset", "split", "duration", "used_stem"]
missing_cols = [c for c in REQUIRED if c not in df.columns]
if missing_cols:
    raise ValueError(f"❌ Manifest missing columns: {missing_cols}")
print("✅ All required columns present")

# ============================================================
# SAFETY CHECK 2: Label Validity
# ============================================================
ok_labels = {"bonafide", "spoof"}
bad_labels = set(df["label"].unique()) - ok_labels
if bad_labels:
    print(f"⚠️  Unexpected labels: {bad_labels}")
else:
    print("✅ All labels valid (bonafide/spoof)")

# ============================================================
# SAFETY CHECK 3: ASVspoof 2021 Should Be Test Only
# ============================================================
bad_21 = df[(df.dataset == "asvspoof2021") & (~df.split.isin(["test", "eval"]))]
if len(bad_21):
    print(f"⚠️  asvspoof2021 has {len(bad_21)} non-test/eval rows")
else:
    print("✅ ASVspoof2021 correctly in test/eval only")

# ============================================================
# SAFETY CHECK 4: Duplicates
# ============================================================
dup_path = df.duplicated(["path"]).sum()
dup_proc = df.duplicated(["proc_path"]).sum()
if dup_path or dup_proc:
    print(f"⚠️  Duplicates — path:{dup_path}, proc_path:{dup_proc}")
else:
    print("✅ No duplicate paths")

# ============================================================
# SAFETY CHECK 5: Short Durations
# ============================================================
short = (df["duration"] < 1.0).sum()
if short:
    print(f"⚠️  Files <1s: {short}")
else:
    print("✅ No files shorter than 1 second")

# ============================================================
# SAFETY CHECK 6: Non-ASCII Paths
# ============================================================
has_nonascii = any(any(ord(ch) > 127 for ch in str(p)) for p in df["path"].head(50))
if has_nonascii:
    print("⚠️  Some paths contain non-ASCII characters")
else:
    print("✅ Paths are ASCII-clean (sample)")

print("\n" + "=" * 60)

# ============================================================
# STATISTICS
# ============================================================
print(f"\n📊 Total samples: {len(df):,}\n")

print("=== Label Distribution ===")
print(df['label'].value_counts())
print()

print("=== Dataset Distribution ===")
print(df['dataset'].value_counts())
print()

print("=== Split Distribution ===")
print(df['split'].value_counts())
print()

print("=== used_stem Distribution ===")
print(df['used_stem'].fillna('mix').value_counts())
print()

# ============================================================
# SAMPLE PATHS
# ============================================================
print("=== Sample Paths ===")
print("First raw path:", df['path'].iloc[0])
print("First proc_path:", df['proc_path'].iloc[0])
print()

# ============================================================
# FILE EXISTENCE CHECK (Configurable)
# ============================================================
SAMPLE_N = int(os.environ.get("VM_SAMPLE", "100"))
print(f"🔍 Checking file existence (sampling {SAMPLE_N} files)...")
sample_df = df.sample(min(SAMPLE_N, len(df)), random_state=42)

missing_raw = 0
missing_proc = 0
first_missing = None

for idx, row in sample_df.iterrows():
    if not Path(row['path']).exists():
        missing_raw += 1
        if first_missing is None:
            first_missing = row['path']
    if not Path(row['proc_path']).exists():
        missing_proc += 1
        if first_missing is None:
            first_missing = row['proc_path']

print(f"Sample check ({SAMPLE_N} files):")
print(f"  Raw files missing: {missing_raw}/{SAMPLE_N}")
print(f"  Processed files missing: {missing_proc}/{SAMPLE_N}")

if missing_proc == 0 and missing_raw == 0:
    print("  ✅ All sampled files found!")
else:
    print(f"  ⚠️  {missing_raw + missing_proc} files missing in sample")
    if first_missing:
        print(f"  First missing: {first_missing}")

# ============================================================
# DURATION STATISTICS
# ============================================================
if 'duration' in df.columns:
    print("\n=== Duration Stats (seconds) ===")
    stats = df['duration'].describe()
    print(f"Count:  {stats['count']:,.0f}")
    print(f"Mean:   {stats['mean']:.2f}s")
    print(f"Std:    {stats['std']:.2f}s")
    print(f"Min:    {stats['min']:.2f}s")
    print(f"25%:    {stats['25%']:.2f}s")
    print(f"50%:    {stats['50%']:.2f}s")
    print(f"75%:    {stats['75%']:.2f}s")
    print(f"Max:    {stats['max']:.2f}s")

# ============================================================
# FINAL STATUS
# ============================================================
print("\n" + "=" * 60)
print("✅ VALIDATION COMPLETE!")
print("=" * 60)

# Exit with error if critical issues found
critical_issues = []
if missing_cols:
    critical_issues.append("Missing columns")
if missing_proc > SAMPLE_N * 0.1:  # More than 10% missing
    critical_issues.append(f"{missing_proc} processed files missing")
if bad_labels:
    critical_issues.append("Invalid labels")

if critical_issues:
    print("\n❌ CRITICAL ISSUES FOUND:")
    for issue in critical_issues:
        print(f"   - {issue}")
    print("\n⚠️  Fix these before training!")
    exit(1)
else:
    print("\n✅ Ready for training!")
    print("\nNext steps:")
    print("  python test_dataloader.py")
    print("  python test_features.py")
