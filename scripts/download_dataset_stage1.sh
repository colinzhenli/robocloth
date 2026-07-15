#!/bin/bash
# ---------------------------------------------------------------------------
# Download the SPARSE training set for stage-1 decoder training: per-material
# observation tensors + pose metadata (NO dense HDR images; ~800 GB instead of
# ~3.5 TB), plus the dataset-level calibration/split files.
#
# Usage:
#   bash scripts/download_dataset_stage1.sh [DATA_ROOT]           # default ./DATA_ROOT
# ---------------------------------------------------------------------------
set -euo pipefail
DEST=${1:-$PWD/DATA_ROOT}
REPO=koalapenguin/cloth-brdf
# internal: restrict to a few materials (testing); default = all
IDS=${ROBOCLOTH_MATERIALS:-*}

INCLUDES=(--include "globals/*")
for id in $IDS; do
    INCLUDES+=(--include "materials/$id/observations_structured.npz" \
               --include "materials/$id/scan_log.json" \
               --include "materials/$id/rotated_camera.json" \
               --include "materials/$id/point_metadata.json")
done

mkdir -p "$DEST"
hf download $REPO --repo-type dataset "${INCLUDES[@]}" --local-dir "$DEST"
cp -f "$DEST"/globals/* "$DEST"/
for d in "$DEST"/materials/*/; do
    id=$(basename "$d"); rm -rf "$DEST/$id"; mv "$d" "$DEST/$id"
done
rmdir "$DEST/materials" 2>/dev/null || true
echo "[download_dataset_stage1] ready: $DEST ($(ls -d "$DEST"/[0-9]* 2>/dev/null | wc -l) materials)"
