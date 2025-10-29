#!/bin/bash
set -euo pipefail

shopt -s nullglob

src_root="."
dst_root="./lombardgrid"
crop_width=640
# Crop the central horizontal region and scale width to 640 to keep a consistent layout.
video_filter="crop=min(iw\\,${crop_width}):ih:(iw-min(iw\\,${crop_width}))/2:0,scale=${crop_width}:-2:flags=lanczos"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: ffmpeg is required but not found in PATH." >&2
  exit 1
fi

mkdir -p "${dst_root}"

dataset_id=1
serial=1

for view in front side; do
  for src_file in "${src_root}/lombardgrid_${view}"/*.mov; do
    filename=$(basename "$src_file")

    talker=$(echo "$filename" | sed -E 's/^s([0-9]+)_.*/\1/')
    talker_num=$(printf "%04d" "${talker}")

    dataset_num=$(printf "%03d" "${dataset_id}")
    serial_num=$(printf "%06d" "${serial}")

    dst_file="${dst_root}/${dataset_num}_${talker_num}_${view}_${serial_num}.mov"

    echo "Crop & copy: $src_file → $dst_file"
    if ! ffmpeg -hide_banner -loglevel error -y -i "$src_file" \
      -vf "${video_filter}" \
      -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
      -c:a copy \
      -movflags +faststart \
      "$dst_file"; then
      echo "Warning: ffmpeg failed for ${src_file}; skipping." >&2
      continue
    fi

    serial=$((serial + 1))
  done
done
