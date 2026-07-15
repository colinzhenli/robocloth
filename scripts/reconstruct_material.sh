#!/bin/bash
# ---------------------------------------------------------------------------
# Reconstruct a single captured material: COLMAP sparse SfM (if not already
# present) followed by robot-frame alignment + observation-tensor building.
#
# Usage:
#   bash scripts/reconstruct_material.sh <material_folder> [gpu_id]
#
# <material_folder> must contain scan_log.json, ldr/ (for COLMAP) and
# hdr_raw/ (16-bit Bayer mosaics). Outputs (hdr/, sparse/, rotated_camera.json,
# bbox.json, point_positions.npz, observations_structured.npz, ...) are written
# in place. For batch processing with optional multi-GPU scheduling see
# reconstruction/scheduler.py.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../reconstruction"

FOLDER=${1:?usage: reconstruct_material.sh <material_folder> [gpu_id]}
GPU=${2:-0}
DATASET_ROOT=$(dirname "$(realpath "$FOLDER")")

if [ ! -f "$FOLDER/sparse/points3D.bin" ] && [ ! -f "$FOLDER/sparse/points3D.txt" ]; then
    echo "[reconstruct] running COLMAP (sequential matcher) on GPU $GPU"
    bash colmap.sh "$FOLDER" "$GPU"
fi

python reconstruct.py \
    shape_matching.folder_path="$FOLDER" \
    shape_matching.num_workers=${NUM_WORKERS:-8} \
    shape_matching.z_outlier_percentile=5.0 \
    dataset_root="$DATASET_ROOT" \
    "${@:3}"
