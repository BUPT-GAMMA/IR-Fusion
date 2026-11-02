import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SampleInfo:
    feature_path: str
    label_path: Optional[str]
    name: str
    is_real: bool
    difficulty: float


def _resolve_path(dataroot: Optional[str], path: str) -> str:
    path = os.path.expanduser(path)
    if os.path.isabs(path) or dataroot is None:
        return path
    return os.path.join(dataroot, path)


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "real"}


def _load_npy(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        data = data[data.files[0]]
    return np.asarray(data, dtype=np.float32)


def to_chw(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, :, :]
    elif array.ndim == 3:
        channels_first = array.shape[0] <= 64 and array.shape[1] > 16 and array.shape[2] > 16
        if not channels_first:
            array = np.transpose(array, (2, 0, 1))
    else:
        raise ValueError(f"expected 2D or 3D array, got shape={array.shape}")
    return np.ascontiguousarray(array)


class IRFusionMapDataset(Dataset):
    def __init__(
        self,
        ann_file: str,
        dataroot: Optional[str] = None,
        require_label: bool = True,
        augment_rotations: bool = False,
        oversample_fake: int = 1,
        oversample_real: int = 1,
    ) -> None:
        self.ann_file = ann_file
        self.dataroot = dataroot
        self.require_label = require_label
        self.samples = self._load_annotations()
        self.items: List[Tuple[int, int]] = []
        rotations = [0, 1, 2, 3] if augment_rotations else [0]
        for idx, sample in enumerate(self.samples):
            repeats = max(1, int(oversample_real if sample.is_real else oversample_fake))
            for _ in range(repeats):
                for rotation in rotations:
                    self.items.append((idx, rotation))

    def _load_annotations(self) -> List[SampleInfo]:
        samples: List[SampleInfo] = []
        with open(self.ann_file, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"empty annotation file: {self.ann_file}")
            required = {"feature", "label", "name", "is_real", "difficulty"}
            missing = required.difference(set(reader.fieldnames))
            if missing:
                raise ValueError(f"{self.ann_file} missing columns: {sorted(missing)}")
            for row in reader:
                label = row.get("label", "").strip()
                if self.require_label and not label:
                    raise ValueError(f"missing label for row: {row}")
                samples.append(
                    SampleInfo(
                        feature_path=_resolve_path(self.dataroot, row["feature"]),
                        label_path=_resolve_path(self.dataroot, label) if label else None,
                        name=row["name"],
                        is_real=_to_bool(row["is_real"]),
                        difficulty=float(row["difficulty"]),
                    )
                )
        if not samples:
            raise ValueError(f"no samples found in {self.ann_file}")
        return samples

    def __len__(self) -> int:
        return len(self.items)

    def difficulty_at(self, index: int) -> float:
        sample_idx, _ = self.items[index]
        return self.samples[sample_idx].difficulty

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample_idx, rotation = self.items[index]
        sample = self.samples[sample_idx]
        feature = to_chw(_load_npy(sample.feature_path))
        label = to_chw(_load_npy(sample.label_path)) if sample.label_path else None

        if rotation:
            feature = np.ascontiguousarray(np.rot90(feature, k=rotation, axes=(-2, -1)))
            if label is not None:
                label = np.ascontiguousarray(np.rot90(label, k=rotation, axes=(-2, -1)))

        item: Dict[str, object] = {
            "feature": torch.from_numpy(feature),
            "name": sample.name,
            "is_real": sample.is_real,
            "difficulty": sample.difficulty,
            "rotation": rotation,
            "feature_path": sample.feature_path,
        }
        if label is not None:
            item["label"] = torch.from_numpy(label)
            item["label_path"] = sample.label_path
        return item


def infer_channels(dataset: IRFusionMapDataset) -> Tuple[int, int]:
    sample = dataset[0]
    in_channels = int(sample["feature"].shape[0])
    out_channels = int(sample["label"].shape[0]) if "label" in sample else 1
    return in_channels, out_channels
