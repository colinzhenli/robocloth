#!/bin/bash
# ---------------------------------------------------------------------------
# Stage 2 — per-material fitting on a held-out UBO2014 BTF material
# (cross-dataset transfer table; 12 materials: carpet02/07/09/12,
# fabric02/04/09/11, felt01/03/05/10).
#
# Path-configurable version of scripts/jobs/run_stage2_ubo_from_{Bonn,MERL,Real}.sh
# and run_stage2_ubo_PBR.sh. MODEL selects the frozen decoder's stage-1
# training source (or the Disney-PBR baseline):
#   MODEL=Ours  -> STAGE1_CKPT default checkpoints/stage1/Ours.ckpt (RoboCloth)
#   MODEL=Bonn  -> STAGE1_CKPT default checkpoints/stage1/Bonn.ckpt
#   MODEL=MERL  -> STAGE1_CKPT default checkpoints/stage1/MERL.ckpt
#                  (per-channel scale beta initialized to 0.1, as in the paper)
#   MODEL=PBR   -> analytic Disney baseline, trained from scratch
#
# Usage:
#   DATA_ROOT=/path/to/BTF STAGE1_CKPT=/path/to/stage1/<src>.ckpt \
#       MODEL=Bonn bash train_stage2_ubo.sh <material>       # e.g. felt01
#
# DATA_ROOT contains <material>_W400xH400_L151xV151.btf files (released on
# Hugging Face under datasets/UBO2014/). Paper: 60 epochs.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../../training"

MAT=${1:?usage: train_stage2_ubo.sh <material>  (e.g. felt01)}
shift || true

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the folder with UBO2014 .btf files}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PWD/outputs/full_experiments}
MODEL=${MODEL:-Bonn}
MAX_EPOCHS=${MAX_EPOCHS:-60}
BTF_FILE=${MAT}_W400xH400_L151xV151.btf
EXP_NAME=${EXP_NAME:-Stage2_UBO_${MAT}_from_${MODEL}_run_1}
export WANDB_MODE=${WANDB_MODE:-offline}

EXTRA=()
if [ "$MODEL" = "PBR" ]; then
    MATERIAL_OVERRIDES=(material=ubo_pbr_latent
                        material.disney=True material.anisotropic=True
                        material.soft_constraint=True model.freeze_decoder=False)
    CKPT_OVERRIDE=()
else
    STAGE1_CKPT=${STAGE1_CKPT:?set STAGE1_CKPT to the stage-1 checkpoint for MODEL=$MODEL}
    MATERIAL_OVERRIDES=(material=ubo_latent material.latent_dim=24
                        material.different_decoder=False
                        material.decoder.use_skip_connection=True
                        material.decoder.use_film=False material.decoder.use_color_decomp=False
                        material.decoder.degree=3 material.decoder.smooth_reg=False
                        model.freeze_decoder=True)
    CKPT_OVERRIDE=(model.ckpt_path="$STAGE1_CKPT")
    [ "$MODEL" = "MERL" ] && EXTRA+=(model.factor_init=0.1)
fi

python train.py \
    dataset_folder="$DATA_ROOT" \
    output_folder="$OUTPUT_ROOT" \
    exp_output_root_path="$OUTPUT_ROOT/$EXP_NAME" \
    data=ubo \
    data.btf_filename=$BTF_FILE \
    data.rays_num=500000 \
    data.valid_num=20 \
    renderer=multiarea_emitter \
    material.learnable_factor=True \
    material.predict_frame=True \
    "${MATERIAL_OVERRIDES[@]}" \
    experiment_name="$EXP_NAME" \
    model.optimizer.name=Adam \
    model.optimizer.lr=0.002 \
    model.optimizer.decoder_lr=2e-4 \
    model.loss.recon_loss.name=logrel \
    model.loss.recon_loss.log_space.logrel_ref=0.05 \
    model.loss.reg_loss.weight=0.0 \
    model.stage=2 \
    model.test=False \
    model.continue_training=False \
    model.apply_cosine_weight=True \
    model.trainer.max_epochs=$MAX_EPOCHS \
    model.trainer.check_val_every_n_epoch=2 \
    model.trainer.limit_train_batches=5838 \
    "${EXTRA[@]}" \
    "${CKPT_OVERRIDE[@]}" \
    "$@"
