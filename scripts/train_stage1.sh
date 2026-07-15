#!/bin/bash
# Stage 1 — BRDF-prior training on RoboCloth.
# All hyperparameters live in configs/experiment/stage1.yaml.
#
#   DATA_ROOT=/path/to/DATA_ROOT [OUTPUT_ROOT=...] bash scripts/train_stage1.sh
#
# Optional: TRAINING_LIST (default $DATA_ROOT/training_list_500.txt), EXP_NAME.
set -euo pipefail
cd "$(dirname "$0")/../training"

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the dataset root}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PWD/../outputs}
TRAINING_LIST=${TRAINING_LIST:-$DATA_ROOT/training_list_500.txt}
EXP_NAME=${EXP_NAME:-stage1_prior}
export WANDB_MODE=${WANDB_MODE:-offline}

python train.py +experiment=stage1 \
    dataset_folder="$DATA_ROOT" \
    data.training_list_path="$TRAINING_LIST" \
    renderer.emitter.direction_json="$DATA_ROOT/emitter_calibration.json" \
    output_folder="$OUTPUT_ROOT" \
    exp_output_root_path="$OUTPUT_ROOT/$EXP_NAME" \
    experiment_name="$EXP_NAME" \
    "$@"
