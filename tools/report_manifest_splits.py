#!/usr/bin/env python3
"""
verify_manifest.py

Sanity checks for manifest.csv:
- Split sizes
- Dataset/label coverage
- Split percentages per dataset
- Label balance per split
- Guards against common leakage mistakes
"""

import pandas as pd


def main():
    df = pd.read_csv("manifest.csv")

    print("=" * 70)
    print("CURRENT SPLIT COMPOSITION")
    print("=" * 70)

    # Overall stats
    print(f"\nTotal samples: {len(df):,}")
    print(f"  Train: {len(df[df['split'] == 'train']):,}")
    print(f"  Dev:   {len(df[df['split'] == 'dev']):,}")
    print(f"  Test:  {len(df[df['split'] == 'test']):,}")

    # Per dataset-label breakdown for TRAIN split
    print("\n" + "=" * 70)
    print("TRAIN SPLIT: (dataset, label) composition")
    print("=" * 70)

    train_comp = (
        df[df["split"] == "train"]
        .groupby(["dataset", "label"])
        .size()
        .sort_index()
    )
    print(train_comp)

    # Missing buckets analysis
    print("\n" + "=" * 70)
    print("MISSING BUCKETS ANALYSIS")
    print("=" * 70)

    all_datasets = sorted(df["dataset"].unique())
    all_labels = sorted(df["label"].unique())

    missing = []
    for dataset in all_datasets:
        for label in all_labels:
            count = len(
                df[
                    (df["split"] == "train")
                    & (df["dataset"] == dataset)
                    & (df["label"] == label)
                ]
            )

            # ASVspoof2021 is test-only by design
            if count == 0 and dataset != "asvspoof2021":
                missing.append((dataset, label))
                print(f"Missing in train: ({dataset}, {label})")

    if not missing:
        print("All dataset-label combinations present in train split.")
    else:
        print(f"\nFound {len(missing)} missing dataset-label buckets.")

    # Per-dataset split percentages
    print("\n" + "=" * 70)
    print("SPLIT PERCENTAGES PER DATASET")
    print("=" * 70)

    for dataset in all_datasets:
        subset = df[df["dataset"] == dataset]
        total = len(subset)
        if total == 0:
            continue

        n_train = len(subset[subset["split"] == "train"])
        n_dev = len(subset[subset["split"] == "dev"])
        n_test = len(subset[subset["split"] == "test"])

        print(f"\n{dataset}:")
        print(f"  Total:  {total:,}")
        print(f"  Train:  {n_train:,} ({n_train / total * 100:.1f}%)")
        print(f"  Dev:    {n_dev:,} ({n_dev / total * 100:.1f}%)")
        print(f"  Test:   {n_test:,} ({n_test / total * 100:.1f}%)")

        # Label balance within each split
        for split_name in ["train", "dev", "test"]:
            split_subset = subset[subset["split"] == split_name]
            if len(split_subset) > 0:
                bonafide = len(split_subset[split_subset["label"] == "bonafide"])
                spoof = len(split_subset[split_subset["label"] == "spoof"])
                print(
                    f"    {split_name}: bonafide={bonafide:,}, spoof={spoof:,}"
                )

    # Critical sanity checks
    print("\n" + "=" * 70)
    print("CRITICAL SANITY CHECKS")
    print("=" * 70)

    # ASVspoof2021 should be test-only
    leaked = df[
        (df["dataset"] == "asvspoof2021") & (df["split"] != "test")
    ]
    if len(leaked) > 0:
        print("ERROR: ASVspoof2021 samples found outside test split.")
        print(leaked[["path", "split"]].head())
    else:
        print("ASVspoof2021 split usage is correct (test-only).")

    print("\nVerification complete.")


if __name__ == "__main__":
    main()
