#!/bin/bash
set -e

src_root="./"
dst_root="./lombardgrid"

mkdir -p "${dst_root}"

dataset_id=1
serial=1

for view in front side; do
  for src_file in ${src_root}/lombardgrid_${view}/*.mov; do
    filename=$(basename "$src_file")

    talker=$(echo "$filename" | sed -E 's/^s([0-9]+)_.*/\1/')
    talker_num=$(printf "%04d" "${talker}")

    dataset_num=$(printf "%03d" "${dataset_id}")
    serial_num=$(printf "%06d" "${serial}")

    dst_file="${dst_root}/${dataset_num}_${talker_num}_${view}_${serial_num}.mov"

    echo "Copy: $src_file → $dst_file"
    cp "$src_file" "$dst_file"

    serial=$((serial + 1))
  done
done
