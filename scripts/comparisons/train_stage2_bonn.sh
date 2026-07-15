#!/bin/bash
# ---------------------------------------------------------------------------
# Stage 2 — per-material fitting on a held-out Bonn (UBOFAB19) test material
# (bottom block of the per-material PSNR table; test materials 318 377 32 226 37).
#
# Path-configurable version of scripts/jobs/run_stage2_bonn_from_{Bonn,MERL,Real}.sh
# and run_stage2_bonn_PBR.sh. MODEL selects the frozen decoder's stage-1
# training source (or the Disney-PBR baseline):
#   MODEL=Ours  -> STAGE1_CKPT default checkpoints/stage1/Ours.ckpt  (RoboCloth)
#   MODEL=Bonn  -> STAGE1_CKPT default checkpoints/stage1/Bonn.ckpt
#   MODEL=MERL  -> STAGE1_CKPT default checkpoints/stage1/MERL.ckpt
#   MODEL=PBR   -> analytic Disney baseline, trained from scratch
#
# Usage:
#   DATA_ROOT=/path/to/Bonn_val STAGE1_CKPT=/path/to/stage1/<src>.ckpt \
#       MODEL=Bonn bash train_stage2_bonn.sh <MAT_ID>
#
# DATA_ROOT is the Bonn *validation* split folder (same layout as Bonn_train;
# see full_experiments.md §1). Paper: 120 epochs.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../../training"

MAT_ID=${1:?usage: train_stage2_bonn.sh <MAT_ID>  (e.g. 318)}
shift || true

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the Bonn_val folder}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PWD/outputs/full_experiments}
MODEL=${MODEL:-Bonn}
MAX_EPOCHS=${MAX_EPOCHS:-120}
EXP_NAME=${EXP_NAME:-Stage2_Bonn${MAT_ID}_from_${MODEL}_run_1}
export WANDB_MODE=${WANDB_MODE:-offline}

if [ "$MODEL" = "PBR" ]; then
    MATERIAL_OVERRIDES=(material=bonn_pbr_latent material.predict_frame=True
                        material.disney=True material.anisotropic=True
                        material.soft_constraint=True model.freeze_decoder=False)
    CKPT_OVERRIDE=()
else
    STAGE1_CKPT=${STAGE1_CKPT:?set STAGE1_CKPT to the stage-1 checkpoint for MODEL=$MODEL}
    MATERIAL_OVERRIDES=(material=bonn_latent material.different_decoder=False
                        material.latent_dim=24 material.decoder.use_skip_connection=True
                        material.decoder.use_film=False material.decoder.use_color_decomp=False
                        material.decoder.degree=3 material.decoder.smooth_reg=False
                        model.freeze_decoder=True)
    CKPT_OVERRIDE=(model.ckpt_path="$STAGE1_CKPT")
fi

python train.py \
    dataset_folder="$DATA_ROOT" \
    output_folder="$OUTPUT_ROOT" \
    exp_output_root_path="$OUTPUT_ROOT/$EXP_NAME" \
    data=bonn \
    data.overfit_mat_id=$MAT_ID \
    data.rays_num=500000 \
    data.valid_num=20 \
    renderer=multiarea_emitter \
    material.learnable_factor=True \
    "${MATERIAL_OVERRIDES[@]}" \
    experiment_name="$EXP_NAME" \
    model.stage=2 \
    model.test=False \
    model.apply_cosine_weight=True \
    model.continue_training=False \
    model.optimizer.name=Adam8bit \
    model.optimizer.lr=0.002 \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.loss.recon_loss.name=logrel \
    model.loss.recon_loss.log_space.logrel_ref=0.02 \
    model.loss.reg_loss.weight=0.0 \
    model.trainer.max_epochs=$MAX_EPOCHS \
    model.trainer.check_val_every_n_epoch=4 \
    model.trainer.limit_train_batches=743 \
    "${CKPT_OVERRIDE[@]}" \
    "$@"
