#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
from pathlib import Path
from typing import List, Tuple, Optional, Iterable, Dict, Any

import matplotlib.pyplot as plt

import cv2
import numpy as np
import pandas as pd

# === MediaPipe Face Mesh ===
import mediapipe as mp

"""
Overview:
- Recursively scan the specified directory for .mp4/.mpg/.mov files
- For each video:
  * Use MediaPipe FaceMesh to extract mouth landmarks for every frame
  * Compute MAR (Mouth Aspect Ratio) and assign labels 0/1/2 (unknown/closed/open) using the threshold
  * Output <basename>_c_xxxxxx_o_xxxxxx_unk_xxxxxx.csv (frame_index, mouth_label, MAR)
  * Output <basename>_output_c_xxxxxx_o_xxxxxx_unk_xxxxxx.mp4 (renders captions "mouth open"/"mouth closed")

Note: MAR definition based on MediaPipe FaceMesh landmark indices
    - Horizontal: 61 (right mouth corner) and 291 (left mouth corner)
    - Vertical: 13 (upper inner lip center) and 14 (lower inner lip center)
    MAR = ||13-14|| / ||61-291||

    The ratio is normalized to remain resolution-independent.
"""

VIDEO_EXTS = {".mp4", ".mpg", ".mov"}

def iter_videos(src_dir: Path, recursive: bool = True) -> Iterable[Path]:
    if recursive:
        yield from (p for p in src_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    else:
        yield from (p for p in src_dir.glob("*") if p.suffix.lower() in VIDEO_EXTS)

def norm2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

class FaceMeshMouth:
    def __init__(self):
        self._mp_face_mesh = mp.solutions.face_mesh
        # refine_landmarks=True improves capture accuracy around the lips
        self._fm = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # MediaPipe Face Mesh landmark indices
        self.idx_h_left = 291  # left mouth corner
        self.idx_h_right = 61  # right mouth corner
        self.idx_v_top = 13    # upper inner lip center
        self.idx_v_bottom = 14 # lower inner lip center

    def mar(self, frame_bgr: np.ndarray) -> Optional[float]:
        """Return MAR; returns None when no face is detected."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._fm.process(rgb)
        if not res.multi_face_landmarks:
            return None

        lm = res.multi_face_landmarks[0].landmark
        h, w = frame_bgr.shape[:2]

        # Convert landmarks to pixel coordinates
        p_left = np.array([lm[self.idx_h_left].x * w, lm[self.idx_h_left].y * h], dtype=np.float32)
        p_right = np.array([lm[self.idx_h_right].x * w, lm[self.idx_h_right].y * h], dtype=np.float32)
        p_top = np.array([lm[self.idx_v_top].x * w, lm[self.idx_v_top].y * h], dtype=np.float32)
        p_bottom = np.array([lm[self.idx_v_bottom].x * w, lm[self.idx_v_bottom].y * h], dtype=np.float32)

        horiz = norm2(p_left, p_right)
        vert = norm2(p_top, p_bottom)

        if horiz <= 1e-6:
            return None
        return vert / horiz

def moving_average(x: List[Optional[float]], win: int) -> List[Optional[float]]:
    """Moving average that ignores None values; returns None when the window has no valid value."""
    if win <= 1:
        return x
    out: List[Optional[float]] = [None] * len(x)
    vals = []
    idxs = []
    s = 0.0
    n = 0
    for i, v in enumerate(x):
        vals.append(v)
        idxs.append(i)
        if v is not None:
            s += v
            n += 1
        # Maintain sliding window
        if len(vals) > win:
            vpop = vals.pop(0)
            idxs.pop(0)
            if vpop is not None:
                s -= vpop
                n -= 1
        out[i] = (s / n) if n > 0 else None
    return out

def pick_codec_for_mp4() -> int:
    """
    Choose a FOURCC that works well for mp4 across Linux/Windows/Mac.
    - 'mp4v' is broadly supported
    - 'avc1' often fails depending on the environment
    """
    return cv2.VideoWriter_fourcc(*"mp4v")

def draw_caption(frame: np.ndarray, text: str, color=(255, 255, 255)) -> None:
    # Outline text to improve readability
    org = (20, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.1
    thickness = 2
    # Draw black outline
    cv2.putText(frame, text, org, font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    # Draw text body
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)

def process_video(
    video_path: Path,
    threshold: float = 0.6,
    smooth_win: int = 5,
    downscale: Optional[int] = None,
    output_fps: Optional[float] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path, Dict[str, int]]:
    """
    Process a single video and return (csv_path, output_video_path, counts).
    - threshold: MAR > threshold yields mouth_label=2 (open)
    - smooth_win: moving average window size in frames (1 disables smoothing)
    - downscale: constrain the longer edge to this size (e.g., 960). None keeps the original size.
    - output_fps: output video FPS (None keeps the source FPS)
    """
    print(f"[INFO] Processing: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {video_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1

    # Determine output size
    if downscale is not None and max(orig_w, orig_h) > downscale:
        if orig_w >= orig_h:
            out_w = downscale
            out_h = int(orig_h * (downscale / orig_w))
        else:
            out_h = downscale
            out_w = int(orig_w * (downscale / orig_h))
    else:
        out_w, out_h = orig_w, orig_h

    out_fps = output_fps or fps_in

    # Resolve output directory
    output_dir = output_dir or video_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fourcc = pick_codec_for_mp4()
    base_name = video_path.stem

    fm = FaceMeshMouth()

    frame_mars: List[Optional[float]] = []
    frames_bgr: List[np.ndarray] = []

    # First pass: compute MAR and store frames
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if (out_w, out_h) != (orig_w, orig_h):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        mar = fm.mar(frame)
        frame_mars.append(mar)
        frames_bgr.append(frame)

        idx += 1
        if total > 0 and idx % 200 == 0:
            print(f"  progress {idx}/{total}")

    cap.release()

    # Smooth MAR series
    smoothed = moving_average(frame_mars, smooth_win)

    # Decide labels and export CSV
    rows = []
    labels = []  # 0/1/2 (unknown/closed/open)
    for i, mar in enumerate(smoothed):
        if mar is None:
            label = 0  # unknown
        else:
            label = 2 if mar > threshold else 1
        labels.append(label)
        rows.append(
            {
                "frame_index": i,
                "mouth_label": label,
                "MAR": frame_mars[i] if frame_mars[i] is not None else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    unknown_count = sum(1 for label in labels if label == 0)
    closed_count = sum(1 for label in labels if label == 1)
    open_count = sum(1 for label in labels if label == 2)
    suffix = f"_c_{closed_count:06d}_o_{open_count:06d}_unk_{unknown_count:06d}"
    csv_path = output_dir / f"{base_name}{suffix}.csv"
    out_video_path = output_dir / f"{base_name}_output{suffix}.mp4"

    # VideoWriter (initialized with filename containing counts)
    writer = cv2.VideoWriter(str(out_video_path), fourcc, out_fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_video_path}")

    df.to_csv(csv_path, index=False)

    # Second pass: draw captions and write frames
    for i, frame in enumerate(frames_bgr):
        mouth_label = labels[i]
        if mouth_label == 2:
            text = "mouth open"
            color = (60, 220, 60)
        elif mouth_label == 1:
            text = "mouth closed"
            color = (60, 160, 255)
        else:
            text = "unknown"
            color = (200, 200, 200)
        draw_caption(frame, text, color=color)
        writer.write(frame)

    writer.release()
    counts = {
        "unknown": unknown_count,
        "closed": closed_count,
        "open": open_count,
        "total_frames": len(labels),
    }
    print(f"  -> CSV:  {csv_path.name}")
    print(f"  -> Video:{out_video_path.name}")
    return csv_path, out_video_path, counts

def main():
    ap = argparse.ArgumentParser(description="Batch mouth open/closed labeling (MediaPipe + MAR)")
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--src_dir", type=str, help="Input directory (search recursively)")
    src_group.add_argument("--src_file", type=str, help="Process single video file")
    ap.add_argument("--no_recursive", action="store_true", help="Disable recursive search")
    ap.add_argument("--threshold", type=float, default=0.05, help="MAR threshold for 'open'")
    ap.add_argument("--smooth_win", type=int, default=1, help="Moving average window on MAR (frames). 1=off")
    ap.add_argument("--downscale_long_edge", type=int, default=None, help="Downscale long edge (e.g., 960). None=original")
    ap.add_argument("--output_fps", type=float, default=None, help="Override output video FPS. None=source FPS")
    ap.add_argument("--output_dir", type=str, default="output", help="Directory to store output CSV and videos")
    args = ap.parse_args()

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.src_file:
        video_path = Path(args.src_file).expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Not found: {video_path}")
        if video_path.suffix.lower() not in VIDEO_EXTS:
            raise ValueError(f"Unsupported file type: {video_path.suffix}")
        videos = [(video_path, output_root)]
        print(f"[INFO] Processing single video: {video_path.name}")
    else:
        src = Path(args.src_dir).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Not found: {src}")
        collected = list(iter_videos(src, recursive=(not args.no_recursive)))
        if not collected:
            print("[WARN] No videos found (.mp4/.mpg/.mov).")
            return
        videos = []
        for vp in collected:
            try:
                rel_parent = vp.parent.relative_to(src)
            except ValueError:
                rel_parent = Path()
            target_dir = output_root / rel_parent
            target_dir.mkdir(parents=True, exist_ok=True)
            videos.append((vp, target_dir))
        print(f"[INFO] Found {len(videos)} videos.")

    processed_results: List[Dict[str, Any]] = []
    for vp, out_dir in videos:
        try:
            _, _, counts = process_video(
                vp,
                threshold=args.threshold,
                smooth_win=args.smooth_win,
                downscale=args.downscale_long_edge,
                output_fps=args.output_fps,
                output_dir=out_dir,
            )
            processed_results.append({"video": str(vp), "counts": counts})
        except Exception as e:
            print(f"[ERROR] {vp}: {e}")

    if not processed_results:
        print("[WARN] No videos processed successfully.")
        return

    videos_total = len(processed_results)
    videos_with_unknown = sum(1 for r in processed_results if r["counts"]["unknown"] > 0)
    videos_without_unknown = videos_total - videos_with_unknown

    total_unknown = sum(r["counts"]["unknown"] for r in processed_results)
    total_closed = sum(r["counts"]["closed"] for r in processed_results)
    total_open = sum(r["counts"]["open"] for r in processed_results)
    total_frames = total_unknown + total_closed + total_open

    total_counts = {
        "unknown": total_unknown,
        "closed": total_closed,
        "open": total_open,
    }
    ratios = {
        "unknown": (total_unknown / total_frames) if total_frames else 0.0,
        "closed": (total_closed / total_frames) if total_frames else 0.0,
        "open": (total_open / total_frames) if total_frames else 0.0,
    }

    summary_primary_rows = [
        ("1", "Videos without unknown frames", str(videos_without_unknown)),
        ("2", "Videos with unknown frames", str(videos_with_unknown)),
        ("3", "Total processed videos", str(videos_total)),
    ]

    summary_counts_rows = [
        ("4", "Total unknown frames", str(total_unknown)),
        ("5", "Total mouth closed frames", str(total_closed)),
        ("6", "Total mouth open frames", str(total_open)),
        ("7", "Total frames", str(total_frames)),
    ]

    all_rows_for_width = summary_primary_rows + summary_counts_rows
    num_width = max(max(len(no) for no, _, _ in all_rows_for_width), len("#"))
    desc_width = max(max(len(desc) for _, desc, _ in all_rows_for_width), len("Description"))
    val_width = max(max(len(val) for _, _, val in all_rows_for_width), len("Value"))
    widths = (num_width, desc_width, val_width)

    def print_table(
        rows: List[Tuple[str, str, str]],
        widths: Tuple[int, int, int],
        separators: Optional[Iterable[int]] = None,
        title: Optional[str] = None,
    ) -> None:
        num_width, desc_width, val_width = widths
        border = f"+-{'-' * num_width}-+-{'-' * desc_width}-+-{'-' * val_width}-+"
        header = f"| {'#':<{num_width}} | {'Description':<{desc_width}} | {'Value':>{val_width}} |"

        if title:
            print(title)
        print(border)
        print(header)
        print(border)
        sep_set = set(separators or [])
        for idx, (no, desc, val) in enumerate(rows, start=1):
            print(f"| {no:<{num_width}} | {desc:<{desc_width}} | {val:>{val_width}} |")
            if idx in sep_set:
                print(border)
        print(border)

    print("\n=== Processing Summary ===")
    print_table(summary_primary_rows, widths, separators=[2])
    print()
    print_table(summary_counts_rows, widths, separators=[3])
    print("8. Histogram (dataset-wide ratios)")

    bar_width = 40
    histogram_labels = [
        ("unknown", "Unknown"),
        ("closed", "Mouth closed"),
        ("open", "Mouth open"),
    ]
    for key, label in histogram_labels:
        ratio = ratios[key]
        count = total_counts[key]
        bar_len = int(round(ratio * bar_width))
        if bar_len == 0 and count > 0:
            bar_len = 1
        bar = "#" * bar_len
        print(f"   - {label:<12} | {bar:<{bar_width}} | {ratio * 100:5.1f}% ({count})")

    hist_output_path = output_root / "overall_histogram.png"
    categories = [label for _, label in histogram_labels]
    ratio_values = [ratios[key] * 100 for key, _ in histogram_labels]
    count_values = [total_counts[key] for key, _ in histogram_labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#a0a0a0", "#3a78d4", "#3cb371"]
    bars = ax.bar(categories, ratio_values, color=colors)

    max_ratio = max(ratio_values) if ratio_values else 0.0
    upper_ylim = max_ratio * 1.2 if max_ratio > 0 else 5.0
    ax.set_ylim(0, upper_ylim)
    ax.set_ylabel("Ratio (%)")
    ax.set_title("Mouth Label Distribution")

    for bar, ratio, count in zip(bars, ratio_values, count_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (max_ratio * 0.05 if max_ratio > 0 else 1.0),
            f"{ratio:.1f}%\n({count})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    fig.savefig(hist_output_path)
    plt.close(fig)
    print(f"   - Histogram image saved to: {hist_output_path}")

if __name__ == "__main__":
    main()
