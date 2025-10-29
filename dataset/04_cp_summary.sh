#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: 04_cp_summary.sh <source_dir> <destination_dir>

Copies every *.png file found under <source_dir> (recursively) into
<destination_dir>, recreating the relative directory structure.
./04_cp_summary.sh output_lombardgrid output_lombardgrid_summary
EOF
}

if [[ $# -ne 2 ]]; then
    usage
    exit 1
fi

SRC_DIR=$1
DEST_DIR=$2

if [[ ! -d "$SRC_DIR" ]]; then
    echo "Error: source directory not found: $SRC_DIR" >&2
    exit 2
fi

mkdir -p "$DEST_DIR"

while IFS= read -r -d '' file; do
    rel_path=${file#"$SRC_DIR"/}
    dest_path="$DEST_DIR/$rel_path"
    mkdir -p "$(dirname "$dest_path")"
    cp -p "$file" "$dest_path"
done < <(find "$SRC_DIR" -type f -name '*.png' -print0)
