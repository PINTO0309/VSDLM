#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Iterable, Dict, Any

import matplotlib.pyplot as plt

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# === FAN (Face Alignment Network) ===
from fan import fan_onnx
from fan.fan_onnx import Box

"""
Overview:
- Recursively scan the specified directory for .mp4/.mpg/.mov files
- For each video:
  * Use a FAN (Face Alignment Network) ONNX model to extract 68 Multi-PIE landmarks for every frame
  * Compute MAR (Mouth Aspect Ratio) and assign labels 0/1/2 (unknown/closed/open) using the threshold
    (front/side thresholds chosen automatically by `_front_`/`_side_` in filename)
  * Output <basename>_c_xxxxxx_o_xxxxxx_unk_xxxxxx.csv (frame_index, mouth_label, MAR)
  * Output <basename>_output_c_xxxxxx_o_xxxxxx_unk_xxxxxx.mp4 (renders captions "mouth open"/"mouth closed")

Note: MAR definition based on FAN's 68-point Multi-PIE landmark indices (inner mouth)
    - Horizontal: 60 (left inner lip corner) and 64 (right inner lip corner)
    - Vertical: averaged distances between (61,67), (63,65), (62,66)
    MAR = (||61-67|| + ||63-65|| + ||62-66||) / (2 * ||60-64||)

    The ratio is normalized to remain resolution-independent.
"""

VIDEO_EXTS = {".mp4", ".mpg", ".mov"}
TALKER_COLORS = ["#a0a0a0", "#3a78d4", "#3cb371"]

def iter_videos(src_dir: Path, recursive: bool = True) -> Iterable[Path]:
    if recursive:
        yield from (p for p in src_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    else:
        yield from (p for p in src_dir.glob("*") if p.suffix.lower() in VIDEO_EXTS)

def norm2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def extract_talker_key(base_name: str) -> Optional[str]:
    parts = base_name.split("_")
    if len(parts) < 2:
        return None
    return f"{parts[0]}_{parts[1]}"

def detect_orientation(base_name: str) -> Optional[str]:
    lower = base_name.lower()
    if "_front_" in lower:
        return "front"
    if "_side_" in lower:
        return "side"
    return None

def resolve_providers(execution_provider: str, cache_dir: Optional[Path] = None) -> List[Any]:
    """Map a provider shortcut to ONNXRuntime provider configuration."""
    if execution_provider == "cpu":
        return ["CPUExecutionProvider"]
    if execution_provider == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if execution_provider == "tensorrt":
        cache_root = Path(cache_dir or ".").resolve()
        return [
            (
                "TensorrtExecutionProvider",
                {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(cache_root),
                    "trt_fp16_enable": True,
                },
            ),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    raise ValueError(f"Unsupported execution provider: {execution_provider}")

class FanMouth:
    """Mouth MAR estimation backed by DEIMv2 detector + FAN face alignment."""

    HEAD_CLASS_ID = 7
    FACE_CLASS_ID = 16
    INNER_MOUTH_INDICES = {
        "left": 60,
        "right": 64,
        "top": (61, 62, 63),
        "bottom": (67, 66, 65),
    }

    def __init__(
        self,
        detection_model: Path,
        alignment_model: Path,
        providers: Optional[List[Any]] = None,
        min_score: float = 0.35,
    ) -> None:
        provider_list = providers or ["CPUExecutionProvider"]
        self._detector = fan_onnx.DEIMv2(
            model_path=str(detection_model),
            providers=provider_list,
        )
        self._aligner = fan_onnx.FAN(
            model_path=str(alignment_model),
            providers=provider_list,
        )
        self._min_score = float(min_score)

    @property
    def min_score(self) -> float:
        return self._min_score

    def mar(self, frame_bgr: np.ndarray) -> Tuple[Optional[float], Optional[Box], Optional[np.ndarray]]:
        """
        Compute MAR for the most confident face in the frame.
        Returns a tuple of (MAR, face_box, landmarks); any component may be None if detection fails.
        """
        boxes = self._detector(
            image=frame_bgr,
            disable_generation_identification_mode=True,
            disable_gender_identification_mode=True,
            disable_left_and_right_hand_identification_mode=True,
            disable_headpose_identification_mode=True,
        )
        face_box = self._select_face_box(boxes)
        if face_box is None:
            return None, face_box, None

        landmarks = self._aligner(frame_bgr, [face_box])
        if landmarks.size == 0:
            return None, face_box, None
        mar_value = self._compute_mar(landmarks[0])
        return mar_value, face_box, (landmarks[0] if mar_value is not None else None)

    def _select_face_box(self, boxes: List[Any]) -> Optional[Box]:
        prioritized_classes = [self.HEAD_CLASS_ID, self.FACE_CLASS_ID]
        best_box = None
        for class_id in prioritized_classes:
            for box in boxes:
                if box.classid != class_id:
                    continue
                if best_box is None or box.score > best_box.score:
                    best_box = box
            if best_box is not None:
                break
        return best_box

    def _compute_mar(self, landmarks: np.ndarray) -> Optional[float]:
        if landmarks.shape[0] < 68:
            return None

        scores = landmarks[:, 2] if landmarks.shape[1] >= 3 else None
        required = (
            [self.INNER_MOUTH_INDICES["left"], self.INNER_MOUTH_INDICES["right"]]
            + list(self.INNER_MOUTH_INDICES["top"])
            + list(self.INNER_MOUTH_INDICES["bottom"])
        )
        if scores is not None:
            for idx in required:
                if scores[idx] < self._min_score:
                    return None

        xy = landmarks[:, :2].astype(np.float32)
        left_idx = self.INNER_MOUTH_INDICES["left"]
        right_idx = self.INNER_MOUTH_INDICES["right"]
        top_indices = self.INNER_MOUTH_INDICES["top"]
        bottom_indices = self.INNER_MOUTH_INDICES["bottom"]

        horiz = norm2(xy[left_idx], xy[right_idx])
        if horiz <= 1e-6:
            return None

        verticals = [
            norm2(xy[t], xy[b]) for t, b in zip(top_indices, bottom_indices)
        ]
        mar = float(sum(verticals) / (2.0 * horiz))
        return mar if np.isfinite(mar) else None

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

def draw_landmark(
    frame: np.ndarray,
    landmarks: Optional[np.ndarray],
    *,
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 1,
    thickness: int = 2,
    min_score: Optional[float] = None,
) -> None:
    if landmarks is None:
        return
    if landmarks.ndim != 2 or landmarks.shape[1] < 2:
        return

    coords = landmarks[:, :2]
    scores = landmarks[:, 2] if landmarks.shape[1] >= 3 else None

    for idx, (x, y) in enumerate(coords):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if scores is not None and min_score is not None:
            if idx < len(scores) and scores[idx] < min_score:
                continue
        center = (int(round(float(x))), int(round(float(y))))
        cv2.circle(frame, center, radius, color, thickness, cv2.LINE_AA)

def draw_bbox(frame: np.ndarray, bbox: Optional[Box], color=(255, 255, 255)) -> None:
    if bbox is None:
        return
    cv2.rectangle(frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 1, cv2.LINE_AA)

def render_talker_summary(
    talker_key: str,
    counts: Dict[str, int],
    output_dir: Path,
) -> Path:
    talker_dir = Path(output_dir)
    talker_dir.mkdir(parents=True, exist_ok=True)
    closed_total = counts.get("closed", 0)
    open_total = counts.get("open", 0)
    unknown_total = counts.get("unknown", 0)
    talker_png_path = talker_dir / (
        f"{talker_key}_t_c_{closed_total:06d}_o_{open_total:06d}_unk_{unknown_total:06d}.png"
    )

    fig_talker, ax_talker = plt.subplots(figsize=(5, 3.5))
    categories = ["Unknown", "Mouth closed", "Mouth open"]
    values = [unknown_total, closed_total, open_total]
    bars = ax_talker.bar(categories, values, color=TALKER_COLORS)
    ax_talker.set_ylabel("Frame count")
    ax_talker.set_title(f"Talker {talker_key}")
    max_value = max(values) if values else 0
    ax_talker.set_ylim(0, max_value * 1.15 if max_value > 0 else 1.0)

    for bar, value in zip(bars, values):
        ax_talker.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + (max_value * 0.03 if max_value > 0 else 0.05),
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    fig_talker.savefig(talker_png_path)
    plt.close(fig_talker)
    return talker_png_path

def render_talker_orientation_summary(
    talker_key: str,
    orientation_counts: Dict[str, Dict[str, int]],
    output_dir: Path,
) -> Optional[Path]:
    front_counts = orientation_counts.get("front")
    side_counts = orientation_counts.get("side")
    if not front_counts or not side_counts:
        return None

    front_total = sum(front_counts.values())
    side_total = sum(side_counts.values())
    if front_total <= 0 or side_total <= 0:
        return None

    talker_dir = Path(output_dir)
    talker_dir.mkdir(parents=True, exist_ok=True)
    png_path = talker_dir / f"{talker_key}_t_front_side_summary.png"

    categories = ["Unknown", "Mouth closed", "Mouth open"]
    orientations = [("Front", front_counts), ("Side", side_counts)]
    fig, axes = plt.subplots(1, len(orientations), figsize=(5 * len(orientations), 3.5))
    if len(orientations) == 1:
        axes = [axes]

    for ax, (label, counts) in zip(axes, orientations):
        values = [counts.get("unknown", 0), counts.get("closed", 0), counts.get("open", 0)]
        bars = ax.bar(categories, values, color=TALKER_COLORS)
        ax.set_ylabel("Frame count")
        ax.set_title(f"{label}")
        max_value = max(values) if values else 0
        ax.set_ylim(0, max_value * 1.15 if max_value > 0 else 1.0)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + (max_value * 0.03 if max_value > 0 else 0.05),
                f"{value}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)
    return png_path

def compose_and_save_summary(
    talker_png_path: Path,
    orientation_png_path: Optional[Path] = None,
) -> Path:
    talker_img = cv2.imread(str(talker_png_path), cv2.IMREAD_COLOR)
    if talker_img is None:
        return talker_png_path

    if orientation_png_path and orientation_png_path.exists():
        orientation_img = cv2.imread(str(orientation_png_path), cv2.IMREAD_COLOR)
        if orientation_img is not None:
            top_h, top_w = talker_img.shape[:2]
            bottom_h, bottom_w = orientation_img.shape[:2]
            combined_width = max(top_w, bottom_w)

            def pad_to_width(img: np.ndarray, width: int) -> np.ndarray:
                h, w = img.shape[:2]
                if w == width:
                    return img
                if w > width:
                    scale = width / w
                    resized = cv2.resize(img, (width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
                    return resized
                pad_total = width - w
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                return cv2.copyMakeBorder(
                    img,
                    0,
                    0,
                    pad_left,
                    pad_right,
                    cv2.BORDER_CONSTANT,
                    value=(255, 255, 255),
                )

            top_img = pad_to_width(talker_img, combined_width)
            bottom_img = pad_to_width(orientation_img, combined_width)
            combined_img = np.vstack([top_img, bottom_img])
            cv2.imwrite(str(talker_png_path), combined_img)
            try:
                orientation_png_path.unlink(missing_ok=True)
            except OSError:
                pass
            return talker_png_path

    cv2.imwrite(str(talker_png_path), talker_img)
    return talker_png_path

def process_video(
    video_path: Path,
    mouth_estimator: FanMouth,
    threshold_front: float = 0.35,
    threshold_side: float = 0.35,
    smooth_win: int = 5,
    downscale: Optional[int] = None,
    output_fps: Optional[float] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path, Dict[str, int]]:
    """
    Process a single video and return (csv_path, output_video_path, counts).
    - threshold_front/threshold_side: orientation-specific MAR thresholds for mouth_label
    - smooth_win: moving average window size in frames (1 disables smoothing)
    - downscale: constrain the longer edge to this size (e.g., 960). None keeps the original size.
    - output_fps: output video FPS (None keeps the source FPS)
    """
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
    base_name_lower = base_name.lower()
    # Choose per-orientation threshold based on filename hints
    if "_side_" in base_name_lower:
        selected_threshold = threshold_side
    elif "_front_" in base_name_lower:
        selected_threshold = threshold_front
    else:
        selected_threshold = threshold_front

    frame_mars: List[Optional[float]] = []
    frames_bgr: List[np.ndarray] = []
    frame_face_bboxes: List[Optional[Box]] = []
    frame_landmarks: List[Optional[np.ndarray]] = []

    # First pass: compute MAR and store frames
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if (out_w, out_h) != (orig_w, orig_h):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        mar, face_box, landmarks = mouth_estimator.mar(frame)
        frame_mars.append(mar)
        frames_bgr.append(frame)
        frame_face_bboxes.append(face_box)
        frame_landmarks.append(landmarks)

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
            label = 2 if mar > selected_threshold else 1
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
        box = frame_face_bboxes[i]
        landmarks = frame_landmarks[i] if i < len(frame_landmarks) else None
        if mouth_label == 2:
            text = "mouth open"
            color = (60, 220, 60)
        elif mouth_label == 1:
            text = "mouth closed"
            color = (60, 160, 255)
        else:
            text = "unknown"
            color = (200, 200, 200)
        draw_landmark(frame, landmarks, color=color, min_score=mouth_estimator.min_score)
        draw_caption(frame, text, color=color)
        draw_bbox(frame, box, color=color)
        writer.write(frame)

    writer.release()
    counts = {
        "unknown": unknown_count,
        "closed": closed_count,
        "open": open_count,
        "total_frames": len(labels),
    }
    return csv_path, out_video_path, counts

def main():
    ap = argparse.ArgumentParser(description="Batch mouth open/closed labeling (FAN + MAR)")
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--src_dir", type=str, help="Input directory (search recursively)")
    src_group.add_argument("--src_file", type=str, help="Process single video file")
    ap.add_argument("--no_recursive", action="store_true", help="Disable recursive search")
    ap.add_argument("--threshold_front", type=float, default=0.25, help="MAR threshold for 'open' when filename contains '_front_'")
    ap.add_argument("--threshold_side", type=float, default=0.55, help="MAR threshold for 'open' when filename contains '_side_'")
    ap.add_argument("--smooth_win", type=int, default=1, help="Moving average window on MAR (frames). 1=off")
    ap.add_argument("--downscale_long_edge", type=int, default=None, help="Downscale long edge (e.g., 960). None=original")
    ap.add_argument("--output_fps", type=float, default=None, help="Override output video FPS. None=source FPS")
    ap.add_argument("--output_dir", type=str, default="output", help="Directory to store output CSV and videos")
    ap.add_argument("--detection_model", type=str, default=None, help="Path to DEIMv2 ONNX model. Default: fan/deimv2_dinov3_s_wholebody34_1750query_n_batch_640x640.onnx")
    ap.add_argument("--alignment_model", type=str, default=None, help="Path to FAN ONNX model. Default: fan/2dfan4_1x3x256x256.onnx")
    ap.add_argument("--execution_provider", type=str, choices=["cpu", "cuda", "tensorrt"], default="tensorrt", help="ONNXRuntime execution provider (default: tensorrt)")
    ap.add_argument("--min_kpt_score", type=float, default=0.15, help="Minimum keypoint confidence required to accept MAR per frame.")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    default_detection_model = script_dir.parent / "fan" / "deimv2_dinov3_s_wholebody34_1750query_n_batch_640x640.onnx"
    default_alignment_model = script_dir.parent / "fan" / "2dfan4_1x3x256x256.onnx"

    detection_model_path = (Path(args.detection_model).expanduser() if args.detection_model else default_detection_model).resolve()
    alignment_model_path = (Path(args.alignment_model).expanduser() if args.alignment_model else default_alignment_model).resolve()

    if not detection_model_path.exists():
        raise FileNotFoundError(f"Detection model not found: {detection_model_path}")
    if not alignment_model_path.exists():
        raise FileNotFoundError(f"Alignment model not found: {alignment_model_path}")

    providers = resolve_providers(args.execution_provider, detection_model_path.parent)
    mouth_estimator = FanMouth(
        detection_model=detection_model_path,
        alignment_model=alignment_model_path,
        providers=providers,
        min_score=args.min_kpt_score,
    )

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.src_file:
        video_path = Path(args.src_file).expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Not found: {video_path}")
        if video_path.suffix.lower() not in VIDEO_EXTS:
            raise ValueError(f"Unsupported file type: {video_path.suffix}")
        videos = [(video_path, output_root)]
    else:
        src = Path(args.src_dir).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Not found: {src}")
        collected = sorted(
            iter_videos(src, recursive=(not args.no_recursive)),
            key=lambda p: p.name.lower(),
        )
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

    processed_results: List[Dict[str, Any]] = []
    talker_summaries: Dict[str, Dict[str, Any]] = {}
    progress_bar: Optional[tqdm] = None
    if (not args.src_file) and videos:
        progress_bar = tqdm(total=len(videos), desc="Videos", unit="video", dynamic_ncols=True)

    for idx, (vp, out_dir) in enumerate(videos):
        if progress_bar is not None:
            progress_bar.set_postfix_str(vp.name, refresh=False)
        try:
            _, _, counts = process_video(
                vp,
                mouth_estimator=mouth_estimator,
                threshold_front=args.threshold_front,
                threshold_side=args.threshold_side,
                smooth_win=args.smooth_win,
                downscale=args.downscale_long_edge,
                output_fps=args.output_fps,
                output_dir=out_dir,
            )
            processed_results.append({"video": str(vp), "counts": counts})
            talker_key = extract_talker_key(vp.stem)
            if talker_key:
                summary = talker_summaries.setdefault(
                    talker_key,
                    {
                        "counts": {"unknown": 0, "closed": 0, "open": 0},
                        "output_dir": out_dir,
                        "finalized": False,
                        "orientation_counts": {},
                        "orientation_finalized": False,
                    },
                )
                scounts = summary["counts"]
                scounts["unknown"] += counts.get("unknown", 0)
                scounts["closed"] += counts.get("closed", 0)
                scounts["open"] += counts.get("open", 0)
                summary["output_dir"] = out_dir

                orientation = detect_orientation(vp.stem)
                if orientation:
                    orientation_counts = summary.setdefault("orientation_counts", {})
                    orient_counts = orientation_counts.setdefault(
                        orientation,
                        {"unknown": 0, "closed": 0, "open": 0},
                    )
                    orient_counts["unknown"] += counts.get("unknown", 0)
                    orient_counts["closed"] += counts.get("closed", 0)
                    orient_counts["open"] += counts.get("open", 0)

                next_talker_key = (
                    extract_talker_key(videos[idx + 1][0].stem)
                    if idx + 1 < len(videos)
                    else None
                )
                if talker_key != next_talker_key and not summary.get("finalized", False):
                    talker_png_path = render_talker_summary(
                        talker_key,
                        scounts,
                        summary.get("output_dir", output_root),
                    )
                    orientation_png = render_talker_orientation_summary(
                        talker_key,
                        summary.get("orientation_counts", {}),
                        summary.get("output_dir", output_root),
                    )
                    final_path = compose_and_save_summary(
                        talker_png_path,
                        orientation_png,
                    )
                    summary["finalized"] = True
                    summary["orientation_finalized"] = bool(orientation_png)
                    message = "   - Talker summary saved to: {}".format(final_path)
                    if orientation_png:
                        message += " (front/side combined)"
                    print(message)
        except Exception as e:
            print(f"[ERROR] {vp}: {e}")
        finally:
            if progress_bar is not None:
                progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()

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

    for talker_key, summary in talker_summaries.items():
        talker_png_path = render_talker_summary(
            talker_key,
            summary["counts"],
            summary.get("output_dir", output_root),
        )
        orientation_png = render_talker_orientation_summary(
            talker_key,
            summary.get("orientation_counts", {}),
            summary.get("output_dir", output_root),
        )
        final_path = compose_and_save_summary(
            talker_png_path,
            orientation_png,
        )
        summary["finalized"] = True
        summary["orientation_finalized"] = bool(orientation_png)
        message = f"   - Talker summary saved to: {final_path}"
        if orientation_png:
            message += " (front/side combined)"
        print(message)

if __name__ == "__main__":
    main()
