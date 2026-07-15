#!/bin/bash
# Stage 1 on the Bonn (UBOFAB19) corpus. Hyperparameters:
# configs/experiment/stage1_bonn.yaml.
#
#   DATA_ROOT=/path/to/Bonn_train bash train_stage1_bonn.sh
set -euo pipefail
cd "$(dirname "$0")/../../training"
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the Bonn_train folder}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PWD/../outputs}
EXP_NAME=${EXP_NAME:-stage1_bonn}
export WANDB_MODE=${WANDB_MODE:-offline}
python train.py +experiment=stage1_bonn \
    dataset_folder="$DATA_ROOT" output_folder="$OUTPUT_ROOT" \
    exp_output_root_path="$OUTPUT_ROOT/$EXP_NAME" experiment_name="$EXP_NAME" "$@"
