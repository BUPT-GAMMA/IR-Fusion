import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from utils.features import (
    build_feature_stack,
    compute_channel_minmax,
    normalize_feature_file,
    write_channel_minmax,
    write_feature_names,
)


FEATURE_SUFFIXES = ("current", "eff_dist", "pdn_density", "ir_drop")
TARGET_SIZE = (256, 256)
LAYERS = (1, 4, 7, 8, 9)
SPICE_SUFFIX = ".powerrush.sp"
NODE_TXT_SUFFIX = ".ir_drop.txt"
NODE_VALUE_COLUMN = "ir_drop_V"
CURRENT_MAP_MODE = "branch-current"


def numeric_design_key(stem: str) -> Tuple[str, int]:
    match = re.search(r"(\d+)$", stem)
    prefix = stem[: match.start()] if match else stem
    number = int(match.group(1)) if match else -1
    return prefix, number


def sample_is_complete(sample_dir: Path, stem: str, spice_suffix: str = ".sp") -> bool:
    required = [sample_dir / f"{stem}{spice_suffix}"] + [sample_dir / f"{stem}_{suffix}.csv" for suffix in FEATURE_SUFFIXES]
    return all(path.exists() for path in required)


def discover_sample_dirs(raw_dir: Path, spice_suffix: str = ".sp") -> Dict[str, Path]:
    sample_dirs: Dict[str, Path] = {}
    label_files = list(raw_dir.glob("*_ir_drop.csv")) + list(raw_dir.glob("*/*_ir_drop.csv"))
    for label_file in sorted(label_files):
        stem = label_file.stem[: -len("_ir_drop")]
        sample_dir = label_file.parent
        if sample_is_complete(sample_dir, stem, spice_suffix=spice_suffix):
            sample_dirs[stem] = sample_dir
    return sample_dirs


def discover_stems(raw_dir: Path, spice_suffix: str = ".sp") -> List[str]:
    return sorted(discover_sample_dirs(raw_dir, spice_suffix=spice_suffix), key=numeric_design_key)


def read_stems_file(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    with open(path, "r") as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


def split_stems(
    stems: Sequence[str],
    seed: int,
    val_ratio: float,
    test_ratio: float,
    test_real_count: int,
    explicit_test: Optional[Sequence[str]] = None,
) -> Dict[str, List[str]]:
    stems = list(stems)
    explicit_test = set(explicit_test or [])
    rng = random.Random(seed)

    real = sorted([stem for stem in stems if stem.startswith("realdesign")], key=numeric_design_key)
    fake = sorted([stem for stem in stems if stem not in set(real)], key=numeric_design_key)

    if real:
        test = [stem for stem in real if stem in explicit_test]
        remaining_real = [stem for stem in real if stem not in explicit_test]
        if not explicit_test and test_real_count > 0:
            selected = remaining_real[-test_real_count:]
            test.extend(selected)
            remaining_real = [stem for stem in remaining_real if stem not in set(selected)]

        rng.shuffle(remaining_real)
        if not explicit_test and test_real_count <= 0 and test_ratio > 0:
            test_count = int(round(len(remaining_real) * test_ratio))
            test.extend(remaining_real[:test_count])
            remaining_real = remaining_real[test_count:]

        val_count = int(round(len(remaining_real) * val_ratio))
        val = sorted(remaining_real[:val_count], key=numeric_design_key)
        train_real = sorted(remaining_real[val_count:], key=numeric_design_key)
        train = sorted(fake + train_real, key=numeric_design_key)
        return {"train": train, "val": val, "test": sorted(test, key=numeric_design_key)}

    test = [stem for stem in stems if stem in explicit_test]
    remaining = [stem for stem in stems if stem not in explicit_test]
    rng.shuffle(remaining)
    if not explicit_test and test_ratio > 0:
        test_count = int(round(len(remaining) * test_ratio))
        test.extend(remaining[:test_count])
        remaining = remaining[test_count:]
    val_count = int(round(len(remaining) * val_ratio))
    val = sorted(remaining[:val_count], key=numeric_design_key)
    train = sorted(remaining[val_count:], key=numeric_design_key)
    return {"train": train, "val": val, "test": sorted(test, key=numeric_design_key)}


def resolve_sample_dirs(raw_dir: Path, stems: Sequence[str], spice_suffix: str) -> Dict[str, Path]:
    discovered = discover_sample_dirs(raw_dir, spice_suffix=spice_suffix)
    sample_dirs: Dict[str, Path] = {}
    for stem in stems:
        if stem in discovered:
            sample_dirs[stem] = discovered[stem]
        elif sample_is_complete(raw_dir, stem, spice_suffix=spice_suffix):
            sample_dirs[stem] = raw_dir
        elif sample_is_complete(raw_dir / stem, stem, spice_suffix=spice_suffix):
            sample_dirs[stem] = raw_dir / stem
    return sample_dirs


def relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_split_csv(path: Path, stems: Iterable[str], feature_dir: Path, label_dir: Path, root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "label", "name", "is_real", "difficulty"])
        writer.writeheader()
        for stem in stems:
            is_real = stem.startswith("realdesign")
            writer.writerow(
                {
                    "feature": relative_to(feature_dir / f"{stem}.npy", root),
                    "label": relative_to(label_dir / f"{stem}.npy", root),
                    "name": stem,
                    "is_real": int(is_real),
                    "difficulty": 1.0 if is_real else 0.0,
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the IR-Fusion feature dataset.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", default="output/processed/full120")
    parser.add_argument("--node-txt-dir", default=None)
    parser.add_argument("--test-stems-file", default="configs/real11_20_stems.txt")
    parser.add_argument("--force", action="store_true", help="Overwrite existing .npy files.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    feature_dir = out_dir / "features"
    label_dir = out_dir / "labels"
    split_dir = out_dir / "splits"
    feature_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = discover_sample_dirs(raw_dir, spice_suffix=SPICE_SUFFIX)
    stems = list(sample_dirs)
    stems = sorted(stems, key=numeric_design_key)
    if not stems:
        raise RuntimeError(f"no complete samples found in {raw_dir}")

    split = split_stems(
        stems,
        seed=0,
        val_ratio=0.0,
        test_ratio=0.0,
        test_real_count=10,
        explicit_test=read_stems_file(args.test_stems_file),
    )
    split["train"] = sorted(set(split["train"] + split["test"]), key=numeric_design_key)

    feature_names = None
    metadata = {
        "target_size": list(TARGET_SIZE),
        "layers": list(LAYERS),
        "normalize": "channel-wise",
        "include_shortest_path": True,
        "spice_suffix": SPICE_SUFFIX,
        "node_txt_suffix": NODE_TXT_SUFFIX,
        "node_txt_value_column": NODE_VALUE_COLUMN,
        "current_map_mode": CURRENT_MAP_MODE,
        "split_strategy": "fake_train_real_split" if any(stem.startswith("realdesign") for stem in stems) else "all_samples_random_split",
        "samples": {},
        "split": split,
    }

    for stem in tqdm(stems, desc="preprocess", ncols=100):
        feature_path = feature_dir / f"{stem}.npy"
        label_path = label_dir / f"{stem}.npy"
        if feature_path.exists() and label_path.exists() and not args.force:
            continue

        sample_dir = sample_dirs[stem]
        node_txt = None
        node_candidates = []
        if args.node_txt_dir:
            node_root = Path(args.node_txt_dir)
            node_candidates.extend([node_root / f"{stem}{NODE_TXT_SUFFIX}", node_root / stem / f"{stem}{NODE_TXT_SUFFIX}"])
        node_candidates.append(sample_dir / f"{stem}{NODE_TXT_SUFFIX}")
        for candidate in node_candidates:
            if candidate.exists():
                node_txt = str(candidate)
                break

        features, label, names = build_feature_stack(
            stem=stem,
            raw_dir=str(sample_dir),
            target_size=TARGET_SIZE,
            layers=LAYERS,
            node_txt_path=node_txt,
            node_txt_value_column=NODE_VALUE_COLUMN,
        )
        if feature_names is None:
            feature_names = names
            write_feature_names(str(out_dir / "feature_names.json"), feature_names)
        elif feature_names != names:
            raise RuntimeError(f"feature name mismatch for {stem}")

        np.save(feature_path, features.astype(np.float32))
        np.save(label_path, label.astype(np.float32))
        metadata["samples"][stem] = {
            "feature_shape": list(features.shape),
            "label_shape": list(label.shape),
            "is_real": stem.startswith("realdesign"),
        }

    if feature_names is None:
        feature_names_path = out_dir / "feature_names.json"
        if not feature_names_path.exists():
            raise RuntimeError("feature names are unavailable")
        with open(feature_names_path, "r") as handle:
            feature_names = json.load(handle)
    train_feature_paths = [feature_dir / f"{stem}.npy" for stem in split["train"]]
    stats = compute_channel_minmax(train_feature_paths, feature_names=feature_names)
    stats["train_stems"] = list(split["train"])
    stats_path = out_dir / "normalization.json"
    write_channel_minmax(stats_path, stats)
    for stem in tqdm(stems, desc="normalize", ncols=100):
        normalize_feature_file(feature_dir / f"{stem}.npy", stats)
    metadata["normalization_stats"] = relative_to(stats_path, out_dir)

    for split_name, split_stems_list in split.items():
        write_split_csv(split_dir / f"{split_name}.csv", split_stems_list, feature_dir, label_dir, out_dir)

    with open(out_dir / "metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"processed={len(stems)} output={out_dir}")
    print(f"splits: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")


if __name__ == "__main__":
    main()
