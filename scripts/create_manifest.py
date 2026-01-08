import os
import random
import argparse
from pathlib import Path

import pandas as pd
import librosa
from tqdm import tqdm


LABELS = ["bonafide", "spoof"]


def default_path(env_key: str, placeholder: str) -> str:
    return os.environ.get(env_key, placeholder)


def get_label_from_path(file_path: Path) -> str:
    s = str(file_path).lower()
    if "bonafide" in s:
        return "bonafide"
    if "spoof" in s:
        return "spoof"
    return "unknown"


def get_duration_seconds(file_path: Path):
    try:
        return float(librosa.get_duration(path=str(file_path)))
    except Exception:
        return None


def extract_entity_id(dataset_name: str, audio_path: Path) -> str:
    filename = audio_path.stem

    if dataset_name == "singing":
        parts = filename.split("_")
        if len(parts) >= 2 and parts[0] == "CtrSVDD":
            return parts[1]
        return parts[0]

    if dataset_name == "singfake":
        parts = filename.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return parts[0] if parts else filename

    if dataset_name == "wavefake":
        if "-" in filename:
            return filename.split("-")[0]
        if "_" in filename:
            return filename.split("_")[0]
        return filename

    return "na"


def grouped_split(entity_ids, train_ratio=0.70, dev_ratio=0.15, test_ratio=0.15, seed=1337):
    unique_entities = sorted(set(entity_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_entities)

    n = len(unique_entities)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)

    entity_to_split = {}
    for i, ent in enumerate(unique_entities):
        if i < n_train:
            entity_to_split[ent] = "train"
        elif i < n_train + n_dev:
            entity_to_split[ent] = "dev"
        else:
            entity_to_split[ent] = "test"
    return entity_to_split


def parse_asvspoof_protocol(protocol_path: str):
    if not os.path.exists(protocol_path):
        return None

    file_to_label = {}
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                file_id = parts[1]
                label = parts[-1]
                file_to_label[file_id] = label
    return file_to_label


def load_asvspoof2019_splits(protocol_dir: str):
    protocol_files = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev":   "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval":  "ASVspoof2019.LA.cm.eval.trl.txt",
    }

    file_info = {}
    for split_name, fname in protocol_files.items():
        path = os.path.join(protocol_dir, fname)
        file_to_label = parse_asvspoof_protocol(path)
        if file_to_label is None:
            continue

        our_split = "test" if split_name == "eval" else split_name
        for file_id, label in file_to_label.items():
            file_info[file_id] = {"split": our_split, "label": label}

    return file_info


def detect_official_split_dirs(root: Path):
    """
    Returns a dict mapping split_name -> directory if an official directory layout exists,
    otherwise returns None.

    Supported layouts:
      - train/dev/test
      - train/val/test
      - train_set/dev_set (SVDD-style): train_set -> train, dev_set -> test
    """
    # Layout 1: train/dev/test
    if (root / "train").exists() and (root / "dev").exists() and (root / "test").exists():
        return {"train": root / "train", "dev": root / "dev", "test": root / "test"}

    # Layout 2: train/val/test (map val -> dev)
    if (root / "train").exists() and (root / "val").exists() and (root / "test").exists():
        return {"train": root / "train", "dev": root / "val", "test": root / "test"}

    # Layout 3: SVDD-style: train_set + dev_set
    if (root / "train_set").exists() and (root / "dev_set").exists():
        # many SVDD repos treat dev_set as evaluation
        return {"train": root / "train_set", "test": root / "dev_set"}

    return None


def iter_audio_files(root: Path):
    audio_files = []
    for ext in [".flac", ".wav"]:
        audio_files.extend(root.rglob(f"*{ext}"))
    return sorted(audio_files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="manifest.csv")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--min_dur", type=float, default=1.0)

    ap.add_argument("--derived_root", default=default_path("DERIVED_ROOT", "/path/to/derived_16k_mono"))

    ap.add_argument("--asvspoof2019", default=default_path("ASV2019_ROOT", "/path/to/ASVspoof2019_LA/LA"))
    ap.add_argument("--asvspoof2019_protocols", default=default_path("ASV2019_PROTOCOLS", "/path/to/ASVspoof2019_LA_cm_protocols"))

    ap.add_argument("--asvspoof2021", default=default_path("ASV2021_ROOT", "/path/to/ASVspoof2021"))
    ap.add_argument("--singing", default=default_path("SINGING_ROOT", "/path/to/CtrSVDD_or_SVDD_singing"))
    ap.add_argument("--singfake", default=default_path("SINGFAKE_ROOT", "/path/to/SingFake"))
    ap.add_argument("--wavefake", default=default_path("WAVEFAKE_ROOT", "/path/to/WaveFake"))

    args = ap.parse_args()

    dataset_paths = {
        "asvspoof2019": args.asvspoof2019,
        "asvspoof2021": args.asvspoof2021,
        "singing": args.singing,
        "singfake": args.singfake,
        "wavefake": args.wavefake,
    }

    rows = []
    asv19_info = None

    for ds, root_str in dataset_paths.items():
        print("=" * 60)
        print(f"Processing {ds}")
        print("=" * 60)

        if root_str.startswith("/path/to/"):
            print(f"Skipping {ds}: set the path via CLI or environment variable.")
            continue

        root = Path(root_str)
        if not root.exists():
            print(f"Skipping {ds} (path not found): {root}")
            continue

        if ds == "asvspoof2019":
            prot = args.asvspoof2019_protocols
            if prot.startswith("/path/to/") or not os.path.exists(prot):
                print("Skipping ASVspoof2019: protocol directory not set or not found.")
                continue
            asv19_info = load_asvspoof2019_splits(prot)
            print(f"Loaded ASVspoof2019 protocol entries: {len(asv19_info)}")

        # Detect official splits for singing datasets
        official_split_dirs = None
        if ds in ["singing", "singfake", "wavefake"]:
            official_split_dirs = detect_official_split_dirs(root)
            if official_split_dirs is not None:
                print(f"{ds}: using official split directories: {list(official_split_dirs.keys())}")

        # Gather files
        if official_split_dirs is not None:
            # collect files per split root
            per_split_files = []
            for split_name, split_root in official_split_dirs.items():
                for p in iter_audio_files(split_root):
                    per_split_files.append((split_name, p))
            print(f"Found {len(per_split_files)} audio files across official splits")
        else:
            all_files = iter_audio_files(root)
            print(f"Found {len(all_files)} audio files")

            entity_to_split = None
            if ds in ["singing", "singfake", "wavefake"]:
                entity_ids = [extract_entity_id(ds, p) for p in all_files]
                entity_to_split = grouped_split(entity_ids, seed=args.seed)

        shown_asv21_msg = False

        # Process
        if official_split_dirs is not None:
            iterator = tqdm(per_split_files, desc=ds)
            for split, p in iterator:
                label = get_label_from_path(p)
                if label == "unknown":
                    continue

                dur = get_duration_seconds(p)
                if dur is None or dur < args.min_dur:
                    continue

                # rel path relative to dataset root for stable proc_path
                rel = p.relative_to(root)
                proc_path = Path(args.derived_root) / ds / rel.with_suffix(".wav")
                used_stem = "vocals" if ds in ["singing", "singfake"] else "mix"

                rows.append({
                    "path": str(p),
                    "label": label,
                    "dataset": ds,
                    "duration": round(dur, 2),
                    "split": split,
                    "proc_path": str(proc_path),
                    "used_stem": used_stem,
                })
        else:
            iterator = tqdm(all_files, desc=ds)
            for p in iterator:
                file_id = p.stem

                if ds == "asvspoof2019":
                    if asv19_info is None or file_id not in asv19_info:
                        continue
                    split = asv19_info[file_id]["split"]
                    label = asv19_info[file_id]["label"]

                elif ds == "asvspoof2021":
                    split = "test"
                    label = get_label_from_path(p)
                    if not shown_asv21_msg:
                        print("ASVspoof2021: assigning all files to test split (eval-only). Ensure not used for training.")
                        shown_asv21_msg = True

                else:
                    ent = extract_entity_id(ds, p)
                    split = entity_to_split[ent]
                    label = get_label_from_path(p)

                if label == "unknown":
                    continue

                dur = get_duration_seconds(p)
                if dur is None or dur < args.min_dur:
                    continue

                rel = p.relative_to(root)
                proc_path = Path(args.derived_root) / ds / rel.with_suffix(".wav")
                used_stem = "vocals" if ds in ["singing", "singfake"] else "mix"

                rows.append({
                    "path": str(p),
                    "label": label,
                    "dataset": ds,
                    "duration": round(dur, 2),
                    "split": split,
                    "proc_path": str(proc_path),
                    "used_stem": used_stem,
                })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("=" * 60)
    print("MANIFEST WRITTEN")
    print("=" * 60)
    print(f"Output: {args.out}")
    print(f"Total rows: {len(df)}")
    if len(df) > 0:
        print(df.groupby(["dataset", "split"]).size())
    print("=" * 60)


if __name__ == "__main__":
    main()
