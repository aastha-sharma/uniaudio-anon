#!/usr/bin/env python3
"""
validate_manifest.py

Manifest sanity checks:
- Required columns
- Label validity
- Dataset/split constraints (ASVspoof2021 is test-only)
- Duplicates
- Duration filter sanity
- Optional file-existence sampling check (off by default)
- Optional path hygiene checks

Environment variables:
  VM_SAMPLE=100            Number of rows to sample for existence checks
  VM_CHECK_EXISTS=0        Set to 1 to enable file existence checks
  VM_STRICT=0              Set to 1 to exit non-zero on warnings
  VM_ASCII_CHECK_ALL=0     Set to 1 to check all paths for non-ASCII
"""

import os
import sys
from pathlib import Path
import pandas as pd


REQUIRED = ["path", "proc_path", "label", "dataset", "split", "duration", "used_stem"]
OK_LABELS = {"bonafide", "spoof"}


def fail(msg: str, code: int = 1):
    print(f"ERROR: {msg}")
    sys.exit(code)


def warn(msg: str, strict: bool):
    print(f"WARNING: {msg}")
    if strict:
        sys.exit(1)


def main():
    strict = os.environ.get("VM_STRICT", "0") == "1"
    check_exists = os.environ.get("VM_CHECK_EXISTS", "0") == "1"
    sample_n = int(os.environ.get("VM_SAMPLE", "100"))
    ascii_check_all = os.environ.get("VM_ASCII_CHECK_ALL", "0") == "1"

    manifest_path = Path("manifest.csv").resolve()
    print(f"Reading manifest: {manifest_path}")
    print("=" * 60)

    df = pd.read_csv(manifest_path, dtype={"used_stem": str})

    # SAFETY CHECK 1: Required Columns
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        fail(f"Manifest missing columns: {missing_cols}")
    print("All required columns present")

    # SAFETY CHECK 2: Label Validity
    bad_labels = set(df["label"].dropna().unique()) - OK_LABELS
    if bad_labels:
        warn(f"Unexpected labels found: {sorted(bad_labels)}", strict)
    else:
        print("All labels valid (bonafide/spoof)")

    # SAFETY CHECK 3: ASVspoof2021 Should Be Test Only
    bad_21 = df[(df["dataset"] == "asvspoof2021") & (df["split"] != "test")]
    if len(bad_21) > 0:
        warn(f"asvspoof2021 has {len(bad_21)} rows outside split='test'", strict)
    else:
        print("ASVspoof2021 correctly assigned to test-only")

    # SAFETY CHECK 4: Duplicates
    dup_path = int(df.duplicated(["path"]).sum())
    dup_proc = int(df.duplicated(["proc_path"]).sum())
    if dup_path or dup_proc:
        warn(f"Duplicates found: path={dup_path}, proc_path={dup_proc}", strict)
    else:
        print("No duplicate paths")

    # SAFETY CHECK 5: Short Durations
    if "duration" in df.columns:
        short = int((df["duration"] < 1.0).sum())
        if short > 0:
            warn(f"Found {short} files with duration < 1.0s", strict)
        else:
            print("No files shorter than 1.0 second")

    # SAFETY CHECK 6: Non-ASCII Paths
    paths_to_check = df["path"].astype(str)
    if not ascii_check_all:
        paths_to_check = paths_to_check.head(min(500, len(df)))  # explicit sample
    non_ascii = [p for p in paths_to_check if any(ord(ch) > 127 for ch in p)]
    if non_ascii:
        warn(f"Found non-ASCII characters in paths (showing up to 3): {non_ascii[:3]}", strict)
    else:
        print("Path ASCII check passed (sampled)" if not ascii_check_all else "Path ASCII check passed (all rows)")

    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)

    print(f"\nTotal samples: {len(df):,}\n")

    print("Label distribution:")
    print(df["label"].value_counts(dropna=False))
    print()

    print("Dataset distribution:")
    print(df["dataset"].value_counts(dropna=False))
    print()

    print("Split distribution:")
    print(df["split"].value_counts(dropna=False))
    print()

    print("used_stem distribution:")
    print(df["used_stem"].fillna("mix").value_counts(dropna=False))
    print()

    # Optional existence checks
    if check_exists:
        print("=" * 60)
        print(f"File existence check (sampling {min(sample_n, len(df))} rows)")
        print("=" * 60)

        sample_df = df.sample(min(sample_n, len(df)), random_state=42)
        missing_raw = 0
        missing_proc = 0
        first_missing = None

        for _, row in sample_df.iterrows():
            raw_p = Path(str(row["path"]))
            proc_p = Path(str(row["proc_path"]))

            if not raw_p.exists():
                missing_raw += 1
                first_missing = first_missing or str(raw_p)
            if not proc_p.exists():
                missing_proc += 1
                first_missing = first_missing or str(proc_p)

        print(f"Raw files missing: {missing_raw}/{len(sample_df)}")
        print(f"Processed files missing: {missing_proc}/{len(sample_df)}")

        # Only treat as fatal in strict mode
        if missing_raw or missing_proc:
            warn(f"Missing files detected in sample. First missing: {first_missing}", strict)
        else:
            print("All sampled files exist")

    # Duration stats
    if "duration" in df.columns:
        print("\n" + "=" * 60)
        print("DURATION STATS (seconds)")
        print("=" * 60)
        stats = df["duration"].describe()
        for k in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
            v = stats[k]
            if k == "count":
                print(f"{k:>5}: {v:,.0f}")
            else:
                print(f"{k:>5}: {v:.2f}")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)

    # Final exit policy
    if strict:
        # strict mode would have exited already on warnings
        pass

    print("Status: OK (non-strict mode).")


if __name__ == "__main__":
    main()
