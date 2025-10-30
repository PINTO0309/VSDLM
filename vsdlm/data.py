from __future__ import annotations

import logging
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]

_PITCH_SUFFIX = re.compile(r"_pitch[mp]\d{3}")
_YAW_SUFFIX = re.compile(r"_yaw[mp]\d{3}")
_MOUTH_SUFFIX = re.compile(r"_mouth$")


@dataclass(frozen=True)
class Annotation:
    video_name: str
    label: int


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    video_name: str
    base_frame: str


def _load_annotations(data_root: Path, logger: Optional[logging.Logger] = None) -> Dict[str, Annotation]:
    logger = logger or logging.getLogger(__name__)
    annotations: Dict[str, Annotation] = {}
    csv_files = sorted(data_root.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV annotation files found under {data_root}.")

    for csv_file in csv_files:
        df = pd.read_csv(
            csv_file,
            usecols=["video_name", "still_image", "class_id"],
        )
        df = df[df["class_id"].isin([1, 2])]
        if df.empty:
            logger.warning("No usable annotations in %s (after dropping class_id == 0).", csv_file.name)
            continue
        for row in df.itertuples(index=False):
            still_image = str(row.still_image).replace("\\", "/")
            label = 0 if row.class_id == 1 else 1
            annotations[still_image] = Annotation(video_name=row.video_name, label=label)
    if not annotations:
        raise RuntimeError(f"No annotations with class_id in {{1,2}} found under {data_root}.")
    logger.info("Loaded %d annotated frames across %d CSV files.", len(annotations), len(csv_files))
    return annotations


def _base_key_from_augmented(relative_path: Path) -> Optional[str]:
    stem = relative_path.stem
    stem = _PITCH_SUFFIX.sub("", stem)
    stem = _YAW_SUFFIX.sub("", stem)
    if _MOUTH_SUFFIX.search(stem):
        stem = _MOUTH_SUFFIX.sub("", stem)
    candidate = relative_path.with_name(f"{stem}{relative_path.suffix}")
    return candidate.as_posix()


def collect_samples(
    data_root: Path,
    logger: Optional[logging.Logger] = None,
) -> List[Sample]:
    logger = logger or logging.getLogger(__name__)
    annotations = _load_annotations(data_root, logger=logger)

    samples: List[Sample] = []
    matched_frames: set[str] = set()
    image_files = sorted((p for p in data_root.glob("*/*.png") if p.is_file()))
    if not image_files:
        raise FileNotFoundError(f"No PNG images found under {data_root}.")

    for image_path in image_files:
        relative = image_path.relative_to(data_root)
        rel_key = relative.as_posix()

        annotation = annotations.get(rel_key)
        base_key = rel_key
        if annotation is None:
            base_candidate = _base_key_from_augmented(relative)
            if base_candidate is not None:
                annotation = annotations.get(base_candidate)
                base_key = base_candidate

        if annotation is None:
            logger.debug("Skipping image without annotation: %s", rel_key)
            continue

        samples.append(
            Sample(
                path=image_path,
                label=annotation.label,
                video_name=annotation.video_name,
                base_frame=base_key,
            )
        )
        matched_frames.add(base_key)

    if not samples:
        raise RuntimeError("No annotated images matched the available PNG files.")

    unmatched = len(annotations) - len(matched_frames)
    if unmatched > 0:
        logger.warning("Skipped %d annotated frames without corresponding images.", unmatched)

    logger.info(
        "Prepared %d samples across %d unique base frames and %d videos.",
        len(samples),
        len(set(s.base_frame for s in samples)),
        len(set(s.video_name for s in samples)),
    )
    return samples


def split_samples(
    samples: Sequence[Sample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, List[Sample]]:
    logger = logger or logging.getLogger(__name__)
    total_ratio = train_ratio + val_ratio + test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-3):
        raise ValueError("Train/val/test ratios must sum to 1.0.")

    groups: Dict[str, List[Sample]] = {}
    for sample in samples:
        groups.setdefault(sample.video_name, []).append(sample)

    group_names = list(groups.keys())
    if not group_names:
        raise RuntimeError("No groups found for splitting.")

    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }

    raw_counts = {split: ratios[split] * len(group_names) for split in ratios}
    counts = {split: (0 if ratios[split] == 0 else int(math.floor(raw_counts[split]))) for split in ratios}
    remaining = len(group_names) - sum(counts.values())

    remainders = sorted(
        ((raw_counts[split] - counts[split], split) for split in ratios if ratios[split] > 0),
        reverse=True,
    )
    for idx in range(remaining):
        if not remainders:
            break
        _, split = remainders[idx % len(remainders)]
        counts[split] += 1

    for split, count in counts.items():
        if ratios[split] > 0 and count == 0:
            logger.warning(
                "Split '%s' requested ratio %.2f but only %d groups available; the split will be empty.",
                split,
                ratios[split],
                len(group_names),
            )

    rng = random.Random(seed)
    group_stats: List[Tuple[str, int, int]] = []
    global_pos = 0
    global_total = 0
    for name, samples_in_group in groups.items():
        pos = sum(s.label for s in samples_in_group)
        total = len(samples_in_group)
        group_stats.append((name, pos, total))
        global_pos += pos
        global_total += total

    global_ratio = (global_pos / global_total) if global_total > 0 else 0.0
    stat_lookup = {name: (pos, total) for name, pos, total in group_stats}

    def score_assignment(assignment: Dict[str, List[str]]) -> float:
        score = 0.0
        for split, names in assignment.items():
            desired_count = counts.get(split, 0)
            if desired_count == 0:
                continue
            split_pos = 0
            split_total = 0
            for n in names:
                pos, total = stat_lookup[n]
                split_pos += pos
                split_total += total
            if split_total == 0:
                continue
            ratio = split_pos / split_total
            score += abs(ratio - global_ratio) * split_total
            score += abs(split_total - (global_total * ratios[split])) * 0.01
        return score

    best_assignment: Dict[str, List[str]] = {split: [] for split in ratios}
    best_score = float("inf")
    attempts = max(200, 20 * len(group_names))
    for _ in range(attempts):
        shuffled = group_names[:]
        rng.shuffle(shuffled)
        idx = 0
        trial = {split: [] for split in ratios}
        for split in ratios:
            count = counts.get(split, 0)
            if count > 0:
                trial[split] = shuffled[idx : idx + count]
                idx += count
        if idx != len(group_names):
            continue
        current_score = score_assignment(trial)
        if current_score < best_score:
            best_score = current_score
            best_assignment = {split: names[:] for split, names in trial.items()}

    assignments: Dict[str, List[Sample]] = {split: [] for split in ratios}
    for split, names in best_assignment.items():
        for name in names:
            assignments[split].extend(groups[name])

    if not assignments["train"]:
        raise ValueError("The training split is empty. Adjust the split ratios or provide more data.")

    for split, subset in assignments.items():
        logger.info(
            "Split '%s': %d samples (%d groups, %.1f%% labelled 'open').",
            split,
            len(subset),
            len({s.video_name for s in subset}),
            100.0 * (sum(s.label for s in subset) / len(subset)) if subset else 0.0,
        )
    return assignments


class VSDLMDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform=None) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(float(sample.label), dtype=torch.float32)
        return {
            "image": image,
            "label": label,
            "video_name": sample.video_name,
            "path": str(sample.path),
            "base_frame": sample.base_frame,
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 4,
    sampler: Optional[WeightedRandomSampler] = None,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_weighted_sampler(samples: Sequence[Sample]) -> WeightedRandomSampler:
    labels = [sample.label for sample in samples]
    counts = Counter(labels)
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
