#!/bin/bash
# Stage 2 on a held-out Bonn material. Hyperparameters:
# configs/experiment/stage2_bonn*.yaml.
#
#   DATA_ROOT=/path/to/Bonn_val STAGE1_CKPT=.../stage1/Bonn.ckpt MODEL=Bonn \
#       bash train_stage2_bonn.sh <material_id>          # e.g. 318
set -euo pipefail
cd "$(dirname "$0")/../../training"
MAT_ID=${1:?usage: train_stage2_bonn.sh <material_id>}; shift || true
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the Bonn_val folder}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PWD/../outputs}
MODEL=${MODEL:-Bonn}
EXP_NAME=${EXP_NAME:-stage2_bonn${MAT_ID}_from_${MODEL}}
export WANDB_MODE=${WANDB_MODE:-offline}
if [ "$MODEL" = "PBR" ]; then EXPERIMENT=stage2_bonn_pbr; CKPT_OVERRIDE=()
else
    STAGE1_CKPT=${STAGE1_CKPT:?set STAGE1_CKPT for MODEL=$MODEL}
    EXPERIMENT=stage2_bonn; CKPT_OVERRIDE=(model.ckpt_path="$STAGE1_CKPT")
fi
python train.py +experiment=$EXPERIMENT \
    dataset_folder="$DATA_ROOT" data.overfit_mat_id=$MAT_ID \
    output_folder="$OUTPUT_ROOT" exp_output_root_path="$OUTPUT_ROOT/$EXP_NAME" \
    experiment_name="$EXP_NAME" "${CKPT_OVERRIDE[@]}" "$@"
