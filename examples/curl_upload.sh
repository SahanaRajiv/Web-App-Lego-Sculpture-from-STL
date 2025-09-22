#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 /path/to/model.stl" >&2
  exit 1
fi

STL="$1"
URL="http://127.0.0.1:5000/upload"

curl -X POST "$URL" \
  -F "file=@${STL}" \
  -F "grid_voxel_count=50" \
  -F "grid_direction=z" \
  -F "generate_pdf=" \
  -F "color_by_shape=" \
  -F "remove_hanging_bricks="

echo "Submitted. Open http://127.0.0.1:5000/history to view results."