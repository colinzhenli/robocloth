#!/bin/bash
# Stage 2 — fit one RoboCloth material with a frozen stage-1 decoder.
# All hyperparameters live in configs/experiment/stage2*.yaml.
#
#   DATA_ROOT=... STAGE1_CKPT=/path/to/checkpoints/stage1/Ours.ckpt \
#       bash scripts/train_stage2.sh <material_id>
#
# MODEL=Ours|Bonn|MERL selects the decoder checkpoint semantics (same config);
# MODEL=PBR trains the Disney baseline from scratch. Optional: EXP_NAME.
set -euo pipefail
cd "$(dirname "$0")/../training"

MAT_ID=${1:?usage: train_stage2.sh <material_id>}
shift || true
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the dataset root}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PWD/../outputs}
MODEL=${MODEL:-Ours}
EXP_NAME=${EXP_NAME:-stage2_${MAT_ID}_from_${MODEL}}
export WANDB_MODE=${WANDB_MODE:-offline}

DATASET_FOLDER=$DATA_ROOT/$MAT_ID
EMITTER_CALIB=${EMITTER_CALIB:-$DATASET_FOLDER/emitter_calibration.json}
[ -f "$EMITTER_CALIB" ] || EMITTER_CALIB=$DATA_ROOT/emitter_calibration.json

if [ "$MODEL" = "PBR" ]; then
    EXPERIMENT=stage2_pbr; CKPT_OVERRIDE=()
else
    STAGE1_CKPT=${STAGE1_CKPT:?set STAGE1_CKPT to the stage-1 decoder checkpoint for MODEL=$MODEL}
    EXPERIMENT=stage2; CKPT_OVERRIDE=(model.ckpt_path="$STAGE1_CKPT")
fi

python train.py +experiment=$EXPERIMENT \
    dataset_folder="$DATASET_FOLDER" \
    renderer.emitter.direction_json="$EMITTER_CALIB" \
    output_folder="$OUTPUT_ROOT" \
    exp_output_root_path="$OUTPUT_ROOT/$EXP_NAME" \
    experiment_name="$EXP_NAME" \
    "${CKPT_OVERRIDE[@]}" \
    "$@"
