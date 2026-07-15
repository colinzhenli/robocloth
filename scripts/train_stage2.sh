#!/bin/bash
# ---------------------------------------------------------------------------
# Stage 2 — per-material BRDF fitting on a RoboCloth material.
#
# Path-configurable version of scripts/jobs/run_stage2_ours_from_Real_cc.sh
# (the exact script used for the paper runs on Compute Canada). All
# hyperparameters are identical to the paper configuration; only the paths are
# parameterized.
#
# The frozen BRDF decoder is warm-started from a stage-1 checkpoint
# (decoder-only load); the dense 2048^2 latent texture, neural geometry and
# per-channel scale (learnable factor) are optimized per material.
#
# Usage:
#   DATA_ROOT=/path/to/capture_data STAGE1_CKPT=/path/to/Ours.ckpt \
#       bash scripts/milestone3/train_stage2.sh <MAT_ID>
#
#   MODEL=PBR bash scripts/milestone3/train_stage2.sh <MAT_ID>
#       trains the Disney-PBR baseline instead (no stage-1 checkpoint needed).
#
# Required inputs (see milestone_instruction.md §3):
#   $DATA_ROOT/<MAT_ID>/hdr/*.png            captured HDR views
#   $DATA_ROOT/<MAT_ID>/scan_log.json
#   $DATA_ROOT/<MAT_ID>/rotated_camera.json
#   $DATA_ROOT/<MAT_ID>/bbox.json
#   $DATA_ROOT/emitter_calibration.json      (dataset root; per-material copy also works)
#
# Outputs:
#   $OUTPUT_ROOT/$EXP_NAME/training/model_0.20_0.20/{epoch=N.ckpt,last.ckpt}
#   $OUTPUT_ROOT/$EXP_NAME/images/{gt,result}_view_*.png   validation renders
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../training"

MAT_ID=${1:?usage: train_stage2.sh <MAT_ID>  (e.g. 145)}
shift || true

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the dataset root (folder containing <mat_id>/ subfolders)}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PWD/outputs/milestone3}
MODEL=${MODEL:-Ours}                    # Ours | Bonn | MERL (stage-1 decoder source) | PBR (Disney baseline)
MAX_EPOCHS=${MAX_EPOCHS:-100}           # paper: 80-120 depending on material (145:120, 226:100, 314/370/452:80)
RAYS_NUM=${RAYS_NUM:-400000}            # paper: 4e5 rays/batch. Reduce if you hit GPU OOM.
EXP_NAME=${EXP_NAME:-Stage2_Ours${MAT_ID}_from_${MODEL}_run_1}
export WANDB_MODE=${WANDB_MODE:-offline}

DATASET_FOLDER=$DATA_ROOT/$MAT_ID
# The emitter's angular-calibration table lives at the dataset root; per-material
# copies are also accepted (the paper's Compute Canada folders had them copied in).
EMITTER_CALIB=${EMITTER_CALIB:-$DATASET_FOLDER/emitter_calibration.json}
[ -f "$EMITTER_CALIB" ] || EMITTER_CALIB=$DATA_ROOT/emitter_calibration.json

if [ "$MODEL" = "PBR" ]; then
    # Disney-PBR baseline: analytic SVBRDF maps, trained from scratch.
    MATERIAL_OVERRIDES=(material=learnable_pbr_texture_model material.disney=True)
    CKPT_OVERRIDE=()
else
    # Neural decoder warm-start (decoder-only load, decoder frozen).
    STAGE1_CKPT=${STAGE1_CKPT:?set STAGE1_CKPT to the stage-1 checkpoint (e.g. Stage-1-Finals/${MODEL}.ckpt)}
    MATERIAL_OVERRIDES=(material=ani_latent_texture_model)
    CKPT_OVERRIDE=(model.ckpt_path="$STAGE1_CKPT")
fi

python train.py \
    output_folder="$OUTPUT_ROOT" \
    exp_output_root_path="$OUTPUT_ROOT/$EXP_NAME" \
    dataset_folder="$DATASET_FOLDER" \
    data=real_dense \
    data.rays_num=$RAYS_NUM \
    data.use_fixed_val=False \
    data.debug=False \
    renderer=multiarea_emitter \
    renderer.spp.train=4 \
    renderer.emitter.direction_json="$EMITTER_CALIB" \
    "${MATERIAL_OVERRIDES[@]}" \
    material.texture_resolution=2048 \
    material.learnable_factor=True \
    material.mono_brdf=False \
    material.different_decoder=False \
    material.use_latent_bank=False \
    material.latent_dim=24 \
    material.neural_geometry.factor=0.08 \
    material.decoder.use_skip_connection=True \
    material.decoder.degree=3 \
    experiment_name="$EXP_NAME" \
    model.stage=2 \
    model.test=False \
    model.continue_training=False \
    model.freeze_decoder=True \
    model.optimizer.name=Adam8bit \
    model.optimizer.lr=0.002 \
    model.loss.recon_loss.name=logrel \
    model.loss.reg_loss.weight=0.0 \
    model.trainer.max_epochs=$MAX_EPOCHS \
    model.trainer.check_val_every_n_epoch=2 \
    model.trainer.limit_train_batches=8000 \
    "${CKPT_OVERRIDE[@]}" \
    "$@"
