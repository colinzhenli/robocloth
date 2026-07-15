#!/bin/bash
# ---------------------------------------------------------------------------
# Stage 1 — BRDF-prior training on the RoboCloth dataset.
#
# Path-configurable version of scripts/jobs/run_stage1_ours_cc.sh (the exact
# script used for the paper runs on Compute Canada). All hyperparameters below
# are identical to the paper configuration; only the paths are parameterized.
#
# Usage:
#   DATA_ROOT=/path/to/capture_data [OUTPUT_ROOT=...] bash scripts/milestone3/train_stage1.sh
#
# Required inputs under DATA_ROOT (see milestone_instruction.md §3):
#   <mat_id>/observations_structured.npz   per material listed in TRAINING_LIST
#   <mat_id>/scan_log.json
#   <mat_id>/rotated_camera.json
#   <mat_id>/point_metadata.json
#   camera_factor.json                      (dataset root)
#   emitter_calibration.json                (dataset root)
#   training_list_500.txt                   (or your own list, one mat id/line)
#
# Outputs:
#   $OUTPUT_ROOT/$EXP_NAME/training/model_0.20_0.20/{epoch=N.ckpt,last.ckpt}
#   The stage-1 checkpoint used for stage 2 is last.ckpt.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../training"

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the dataset root (folder containing <mat_id>/ subfolders)}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PWD/outputs/milestone3}
TRAINING_LIST=${TRAINING_LIST:-$DATA_ROOT/training_list_500.txt}
EXP_NAME=${EXP_NAME:-Stage1_RoboCloth_run_1}
MAX_EPOCHS=${MAX_EPOCHS:-100}          # paper: ~100 epochs x 8000 iters = 4.8e5 steps (ckpt picked at val plateau, epoch 60 used in paper)
RAYS_NUM=${RAYS_NUM:-500000}           # paper: 5e5 rays/batch. Reduce (e.g. 250000) if you hit GPU OOM.
CHECK_VAL_EVERY=${CHECK_VAL_EVERY:-5}
export WANDB_MODE=${WANDB_MODE:-offline}   # set WANDB_MODE=online to log to wandb

python train.py \
    dataset_folder="$DATA_ROOT" \
    data.training_list_path="$TRAINING_LIST" \
    output_folder="$OUTPUT_ROOT" \
    exp_output_root_path="$OUTPUT_ROOT/$EXP_NAME" \
    data=stage1_dense \
    data.legacy_swap_indexing=False \
    data.rays_num=$RAYS_NUM \
    data.point_subsample_ratio=0.1 \
    renderer=robocloth_rig \
    renderer.emitter.direction_json="$DATA_ROOT/emitter_calibration.json" \
    material=stage1_prior \
    experiment_name="$EXP_NAME" \
    model.optimizer.name=Adam8bit \
    model.continue_training=False \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.optimizer.lr=2e-3 \
    model.optimizer.decoder_lr=2e-4 \
    model.loss.recon_loss.name=logrel \
    model.loss.reg_loss.weight=0.0 \
    model.stage=1 \
    model.test=False \
    model.trainer.max_epochs=$MAX_EPOCHS \
    model.trainer.check_val_every_n_epoch=$CHECK_VAL_EVERY \
    model.trainer.limit_train_batches=8000 \
    model.grazing_ratio=0.05 \
    model.grazing_mode=near_zero_brdf \
    material.decoder.use_skip_connection=True \
    material.latent_dim=24 \
    material.decoder.degree=3 \
    material.decoder.smooth_reg=False \
    material.different_decoder=False \
    data.filter_observations=False \
    data.switch_iters=100 \
    data.chunk_size=2 \
    renderer.spp.train=4 \
    "$@"
