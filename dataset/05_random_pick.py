import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


SPLIT_COUNT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly copy a fixed number of folders per talker from the "
            "'output_grid_audio_visual_speech_corpus_still_image' dataset into "
            "ten splits (balanced as evenly as possible) under "
            "'output_grid_audio_visual_speech_corpus_still_image_partial'."
        )
    )

    default_root = Path(__file__).resolve().parent
    parser.add_argument(
        "--src",
        type=Path,
        default=default_root / "output_grid_audio_visual_speech_corpus_still_image",
        help="Source directory that contains folders grouped by talker.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=default_root / "output_grid_audio_visual_speech_corpus_still_image_partial",
        help="Destination directory to store the sampled folders.",
    )
    parser.add_argument(
        "--per-talker",
        type=int,
        default=10,
        help="Number of folders to copy per talker.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20240528,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--clear-dst",
        action="store_true",
        help="Remove the destination directory before copying.",
    )
    return parser.parse_args()


def iter_source_folders(src_dir: Path) -> Iterable[Path]:
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")

    for path in sorted(src_dir.iterdir()):
        if path.is_dir():
            yield path


def extract_talker_key(folder_name: str) -> str:
    """
    Folder names follow the pattern:
        <session>_<talker>_<camera>_<frame>
    The second chunk (<talker>) distinguishes talkers; we keep the session prefix
    to stay consistent with existing naming (e.g. `002_0001`).
    """
    parts = folder_name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Folder name does not contain a talker identifier: {folder_name}")
    return "_".join(parts[:2])


def group_by_talker(folders: Iterable[Path]) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = defaultdict(list)
    for folder in folders:
        talker = extract_talker_key(folder.name)
        grouped[talker].append(folder)
    return grouped


def select_folders_per_talker(
    grouped: Dict[str, Sequence[Path]],
    sample_size: int,
    rng: random.Random,
) -> Dict[str, List[Path]]:
    selections: Dict[str, List[Path]] = {}
    for talker, candidates in grouped.items():
        ordered = sorted(candidates, key=lambda p: p.name)
        if len(ordered) <= sample_size:
            selections[talker] = ordered
            continue
        sampled = rng.sample(ordered, sample_size)
        selections[talker] = sorted(sampled, key=lambda p: p.name)
    return selections


def destinations_from_base(base: Path, count: int = SPLIT_COUNT) -> List[Path]:
    return [base.parent / f"{base.name}{idx}" for idx in range(1, count + 1)]


def prepare_destinations(dest_dirs: Sequence[Path], clear: bool) -> None:
    for dest_dir in dest_dirs:
        if clear and dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)


def copy_folders(
    selections: Dict[str, Sequence[Path]],
    dest_dirs: Sequence[Path],
) -> int:
    ordered_entries: List[tuple[str, Path]] = []
    for talker in sorted(selections):
        folders = sorted(selections[talker], key=lambda p: p.name)
        for folder in folders:
            ordered_entries.append((talker, folder))

    total = len(ordered_entries)
    if total == 0 or not dest_dirs:
        for dest_dir in dest_dirs:
            print(f"{dest_dir}: total 0 folders")
        return 0

    talker_counts: Dict[str, int] = {}
    dest_counts: Dict[Path, int] = {dest: 0 for dest in dest_dirs}
    dest_order = list(dest_dirs)
    dest_base = total // len(dest_order)
    dest_remainder = total % len(dest_order)

    for idx, (talker, folder) in enumerate(ordered_entries):
        talker_counts[talker] = talker_counts.get(talker, 0) + 1
        dest_dir = dest_order[idx % len(dest_order)]
        dest_path = dest_dir / folder.name
        if dest_path.exists():
            shutil.rmtree(dest_path)
        shutil.copytree(folder, dest_path)
        dest_counts[dest_dir] += 1

    for talker in sorted(talker_counts):
        print(f"{talker}: copied {talker_counts[talker]} folders")
    for pos, dest_dir in enumerate(dest_order):
        quota = dest_base + (1 if pos < dest_remainder else 0)
        actual = dest_counts[dest_dir]
        print(f"{dest_dir}: total {actual} folders (target {quota})")

    if dest_counts:
        min_count = min(dest_counts.values())
        max_count = max(dest_counts.values())
        if max_count - min_count > 1:
            print(
                "Warning: distribution difference exceeds 1 folder between splits."
            )

    return total


def main() -> None:
    args = parse_args()

    rng = random.Random(args.seed)
    folders = list(iter_source_folders(args.src))
    grouped = group_by_talker(folders)

    if not grouped:
        raise RuntimeError(f"No talker folders found in {args.src}")

    selections = select_folders_per_talker(grouped, args.per_talker, rng)
    dest_dirs = destinations_from_base(args.dst)
    prepare_destinations(dest_dirs, args.clear_dst)
    total_copied = copy_folders(selections, dest_dirs)

    print(
        f"Copied {total_copied} folders into {len(dest_dirs)} destinations derived from base {args.dst}"
    )


if __name__ == "__main__":
    main()
