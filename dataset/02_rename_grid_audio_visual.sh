#!/bin/bash
set -e

src_root="./GRID_Audio-Visual_Speech_Corpus"
dst_root="./grid_audio_visual_speech_corpus"

mkdir -p "${dst_root}"

dataset_id=2
serial=1

for talker_dir in "${src_root}"/s*/; do
  talker_name=$(basename "$talker_dir")   # 例: s1
  talker_num=$(echo "$talker_name" | sed -E 's/^s([0-9]+)/\1/')
  talker_padded=$(printf "%04d" "${talker_num}")
  dataset_padded=$(printf "%03d" "${dataset_id}")

  for src_file in "${talker_dir}/${talker_name}"/*.mpg; do
    filename=$(basename "$src_file")
    serial_padded=$(printf "%06d" "${serial}")

    dst_file="${dst_root}/${dataset_padded}_${talker_padded}_front_${serial_padded}.mpg"

    echo "Copy: $src_file → $dst_file"
    cp "$src_file" "$dst_file"

    serial=$((serial + 1))
  done
done
