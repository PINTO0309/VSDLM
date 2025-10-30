import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm


def detect_face_landmarks(image_path: str, model_path: str = "face_landmarker.task"):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")
    h, w = image.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    with FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise RuntimeError("No face landmarks detected")

    face_landmarks = result.face_landmarks[0]
    coords = np.array(
        [[lm.x * w, lm.y * h, lm.z * w] for lm in face_landmarks],
        dtype=np.float32,
    )
    return image, coords


def _expand_range_spec(spec: str) -> List[float]:
    parts = [p.strip() for p in spec.split(":")]
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid angle range specification: {spec}")
    start = float(parts[0])
    end = float(parts[1])
    step = float(parts[2]) if len(parts) == 3 else 1.0
    if step == 0:
        raise ValueError("Angle step cannot be zero.")
    if (end - start) * step < 0:
        raise ValueError(f"Step sign does not reach end value: {spec}")

    values: List[float] = []
    current = start
    epsilon = abs(step) * 1e-6
    if step > 0:
        while current <= end + epsilon:
            values.append(float(round(current, 6)))
            current += step
    else:
        while current >= end - epsilon:
            values.append(float(round(current, 6)))
            current += step
    if not values or abs(values[-1] - end) > epsilon:
        values.append(float(round(end, 6)))
    return values


def _parse_angle_list(value: str) -> List[float]:
    if not value:
        return [0.0]
    raw_tokens = [token.strip() for token in value.split(",") if token.strip()]
    angles: List[float] = []
    for token in raw_tokens:
        if ":" in token:
            angles.extend(_expand_range_spec(token))
        else:
            angles.append(float(token))
    seen = set()
    ordered = []
    for angle in angles:
        if angle not in seen:
            seen.add(angle)
            ordered.append(angle)
    return ordered


def _rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)

    rot_yaw = np.array(
        [[cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]],
        dtype=np.float32,
    )
    rot_pitch = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_p, -sin_p], [0.0, sin_p, cos_p]],
        dtype=np.float32,
    )
    return rot_yaw @ rot_pitch


def _compute_delaunay_triangles(
    points: np.ndarray, image_shape: Tuple[int, int]
) -> List[Tuple[int, int, int]]:
    h, w = image_shape
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    for (x, y) in points:
        if np.isnan(x) or np.isnan(y):
            continue
        if x <= 0 or x >= w - 1 or y <= 0 or y >= h - 1:
            continue
        subdiv.insert((float(x), float(y)))

    triangle_list = subdiv.getTriangleList()
    triangles: List[Tuple[int, int, int]] = []
    seen = set()
    for t in triangle_list:
        pts = np.array([(t[0], t[1]), (t[2], t[3]), (t[4], t[5])], dtype=np.float32)
        if np.any(pts[:, 0] < 0) or np.any(pts[:, 0] >= w) or np.any(pts[:, 1] < 0) or np.any(pts[:, 1] >= h):
            continue
        indices = []
        for px, py in pts:
            distances = np.linalg.norm(points - np.array([px, py]), axis=1)
            idx = int(np.argmin(distances))
            if distances[idx] > 1.0:
                indices = []
                break
            indices.append(idx)
        if len(indices) == 3:
            key = tuple(sorted(indices))
            if key not in seen:
                seen.add(key)
                triangles.append(tuple(indices))
    return triangles


def _warp_triangle(
    src_img: np.ndarray,
    dst_img: np.ndarray,
    triangle_src: np.ndarray,
    triangle_dst: np.ndarray,
    accumulation_mask: np.ndarray,
) -> None:
    triangle_src = triangle_src.astype(np.float32)
    triangle_dst = triangle_dst.astype(np.float32)

    r1 = cv2.boundingRect(triangle_src)
    r2 = cv2.boundingRect(triangle_dst)

    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return

    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2

    h_img, w_img = src_img.shape[:2]
    if (
        x1 < 0
        or y1 < 0
        or x1 + w1 > w_img
        or y1 + h1 > h_img
        or x2 < 0
        or y2 < 0
        or x2 + w2 > w_img
        or y2 + h2 > h_img
    ):
        return

    src_rect = triangle_src - np.array([x1, y1], dtype=np.float32)
    dst_rect = triangle_dst - np.array([x2, y2], dtype=np.float32)

    mask_patch = np.zeros((h2, w2), dtype=np.uint8)
    cv2.fillConvexPoly(mask_patch, np.int32(dst_rect), 255)

    img1_rect = src_img[y1 : y1 + h1, x1 : x1 + w1]
    warp_mat = cv2.getAffineTransform(src_rect, dst_rect)
    img2_rect = cv2.warpAffine(
        img1_rect,
        warp_mat,
        (w2, h2),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    roi = dst_img[y2 : y2 + h2, x2 : x2 + w2]
    mask_inv = cv2.bitwise_not(mask_patch)
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    warp_fg = cv2.bitwise_and(img2_rect, img2_rect, mask=mask_patch)
    dst_img[y2 : y2 + h2, x2 : x2 + w2] = cv2.add(roi_bg, warp_fg)
    accumulation_mask[y2 : y2 + h2, x2 : x2 + w2] = cv2.bitwise_or(
        accumulation_mask[y2 : y2 + h2, x2 : x2 + w2], mask_patch
    )


def rotate_face_image(
    image: np.ndarray,
    coords: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    center = coords.mean(axis=0)

    rotation = _rotation_matrix(yaw_deg, pitch_deg)
    rotated_points = ((coords - center) @ rotation.T) + center

    src_points_2d = coords[:, :2]
    dst_points_2d = rotated_points[:, :2]

    triangles = _compute_delaunay_triangles(src_points_2d, (h, w))
    if not triangles:
        return image.copy(), np.ones((h, w), dtype=np.uint8) * 255

    warped = np.zeros_like(image)
    mask = np.zeros((h, w), dtype=np.uint8)

    for tri in triangles:
        tri_src = src_points_2d[list(tri)]
        tri_dst = dst_points_2d[list(tri)]

        if (
            np.any(tri_dst[:, 0] < 0)
            or np.any(tri_dst[:, 0] >= w)
            or np.any(tri_dst[:, 1] < 0)
            or np.any(tri_dst[:, 1] >= h)
        ):
            continue
        _warp_triangle(image, warped, tri_src, tri_dst, mask)

    face_region = cv2.bitwise_and(warped, warped, mask=mask)
    gray_value = 128
    result = np.full_like(image, gray_value)
    result[mask > 0] = face_region[mask > 0]
    return result, mask


def save_rotated_face_images(
    image_path: str,
    model_path: str,
    output_dir: str,
    yaw_angles: Iterable[float],
    pitch_angles: Iterable[float],
    crop_margin: int = 0,
    output_size: int = 128,
) -> List[Path]:
    image, coords = detect_face_landmarks(image_path, model_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    base_name = Path(image_path).stem
    max_width = 0
    max_height = 0

    pitch_iter = tqdm(list(pitch_angles), desc="pitch", leave=True, dynamic_ncols=True)
    for pitch in pitch_iter:
        yaw_iter = tqdm(list(yaw_angles), desc="  yaw", leave=False, dynamic_ncols=True)
        for yaw in yaw_iter:
            rotated, mask = rotate_face_image(image, coords, yaw, pitch)

            ys, xs = np.where(mask > 0)
            if len(xs) > 0 and len(ys) > 0:
                x_min = max(0, int(xs.min()) - crop_margin)
                x_max = min(rotated.shape[1], int(xs.max()) + crop_margin + 1)
                y_min = max(0, int(ys.min()) - crop_margin)
                y_max = min(rotated.shape[0], int(ys.max()) + crop_margin + 1)
                cropped = rotated[y_min:y_max, x_min:x_max]
                if cropped.size == 0:
                    cropped = rotated
            else:
                cropped = rotated

            ch, cw = cropped.shape[:2]
            if ch > max_height:
                max_height = ch
            if cw > max_width:
                max_width = cw

            scale = output_size / max(ch, cw) if max(ch, cw) > 0 else 1.0
            if scale != 1.0:
                resized = cv2.resize(
                    cropped,
                    (int(round(cw * scale)), int(round(ch * scale))),
                    interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
                )
            else:
                resized = cropped

            rh, rw = resized.shape[:2]
            pad_image = np.full(
                (output_size, output_size, 3), 128, dtype=resized.dtype
            )
            y_start = (output_size - rh) // 2
            x_start = (output_size - rw) // 2
            pad_image[y_start : y_start + rh, x_start : x_start + rw] = resized

            suffix = f"pitch{int(round(pitch)):+03d}_yaw{int(round(yaw)):+03d}.png"
            output_path = output_root / f"{base_name}_{suffix}"
            cv2.imwrite(str(output_path), pad_image)
            saved_paths.append(output_path)

    print(f"max cropped width: {max_width}px")
    print(f"max cropped height: {max_height}px")
    return saved_paths


def _normalize_range_arguments(argv: Sequence[str]) -> List[str]:
    option_keys = {"--yaw", "--pitch"}
    normalized: List[str] = []
    skip = False
    for idx, token in enumerate(argv):
        if skip:
            skip = False
            continue
        if (
            token in option_keys
            and idx + 1 < len(argv)
            and argv[idx + 1].startswith("-")
            and ":" in argv[idx + 1]
        ):
            normalized.append(f"{token}={argv[idx + 1]}")
            skip = True
        else:
            normalized.append(token)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rotate full-face region based on MediaPipe Face Landmarks."
    )
    parser.add_argument("--image", required=True, help="Path to the input image file")
    parser.add_argument(
        "--model",
        default="face_landmarker.task",
        help="Path to the MediaPipe face landmarker task file",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_face_pose",
        help="Output directory",
    )
    parser.add_argument(
        "--yaw",
        default="-25:25:5",
        help="Yaw angles (degrees). Comma separated or start:end[:step] (e.g. -25:25:5)",
    )
    parser.add_argument(
        "--pitch",
        default="-30:30:5",
        help="Pitch angles (degrees). Comma separated or start:end[:step] (e.g. -20:20:5)",
    )
    parser.add_argument(
        "--crop-margin",
        type=int,
        default=0,
        help="Extra padding added to the crop bounds (pixels)",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=128,
        help="Output image side length (pixels)",
    )

    normalized_argv = _normalize_range_arguments(sys.argv[1:])
    args = parser.parse_args(normalized_argv)

    yaw_angles = _parse_angle_list(args.yaw)
    pitch_angles = _parse_angle_list(args.pitch)
    saved_paths = save_rotated_face_images(
        args.image,
        args.model,
        args.output_dir,
        yaw_angles,
        pitch_angles,
        crop_margin=args.crop_margin,
        output_size=args.output_size,
    )


if __name__ == "__main__":
    main()
