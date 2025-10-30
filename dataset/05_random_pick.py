import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly copy a fixed number of folders per talker from the "
            "'output_grid_audio_visual_speech_corpus_still_image' dataset into "
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
        selections[talker] = rng.sample(ordered, sample_size)
    return selections


def prepare_destination(dest_dir: Path, clear: bool) -> None:
    if clear and dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)


def copy_folders(selections: Dict[str, Sequence[Path]], dest_dir: Path) -> None:
    for talker in sorted(selections):
        folders = selections[talker]
        for folder in folders:
            dest_path = dest_dir / folder.name
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(folder, dest_path)
        print(f"{talker}: copied {len(folders)} folders")


def main() -> None:
    args = parse_args()

    rng = random.Random(args.seed)
    folders = list(iter_source_folders(args.src))
    grouped = group_by_talker(folders)

    if not grouped:
        raise RuntimeError(f"No talker folders found in {args.src}")

    selections = select_folders_per_talker(grouped, args.per_talker, rng)
    prepare_destination(args.dst, args.clear_dst)
    copy_folders(selections, args.dst)

    total = sum(len(paths) for paths in selections.values())
    print(f"Copied {total} folders into {args.dst}")


if __name__ == "__main__":
    main()
