#!/bin/bash
# ---------------------------------------------------------------------------
# Download the FULL RoboCloth dataset (~3.5 TB: all 500 materials including
# dense HDR views) and assemble a training-ready DATA_ROOT. Each hdr.tar is
# extracted and removed after download.
#
# Usage:
#   bash scripts/download_dataset_full.sh [DATA_ROOT]             # default ./DATA_ROOT
# ---------------------------------------------------------------------------
set -euo pipefail
DEST=${1:-$PWD/DATA_ROOT}
REPO=koalapenguin/cloth-brdf
IDS=${ROBOCLOTH_MATERIALS:-*}    # internal: restrict for testing

INCLUDES=(--include "globals/*")
for id in $IDS; do INCLUDES+=(--include "materials/$id/*"); done

echo "[download_dataset_full] full dataset is ~3.5 TB — ensure disk space."
mkdir -p "$DEST"
hf download $REPO --repo-type dataset "${INCLUDES[@]}" --local-dir "$DEST"
cp -f "$DEST"/globals/* "$DEST"/
for d in "$DEST"/materials/*/; do
    id=$(basename "$d"); rm -rf "$DEST/$id"; mv "$d" "$DEST/$id"
    if [ -f "$DEST/$id/hdr.tar" ]; then
        tar -xf "$DEST/$id/hdr.tar" -C "$DEST/$id" && rm "$DEST/$id/hdr.tar"
    fi
done
rmdir "$DEST/materials" 2>/dev/null || true
echo "[download_dataset_full] ready: $DEST ($(ls -d "$DEST"/[0-9]* 2>/dev/null | wc -l) materials)"
