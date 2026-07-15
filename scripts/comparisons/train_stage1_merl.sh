#!/bin/bash
# ---------------------------------------------------------------------------
# Stage 1 — BRDF-prior training on the MERL measured-BRDF database
# (the "MERL" decoder column of the comparison tables).
#
# Path-configurable version of scripts/jobs/run_stage1_merl.sh (the exact
# paper configuration). The MERL dataset (103 .binary files) is included in
# the released Hugging Face bundle under datasets/MERL/brdfs/.
#
# Usage:
#   DATA_ROOT=/path/to/MERL/brdfs [OUTPUT_ROOT=...] bash train_stage1_merl.sh
#
# Output: $OUTPUT_ROOT/$EXP_NAME/training/model_0.20_0.20/last.ckpt
#         (the released checkpoint of this run is checkpoints/stage1/MERL.ckpt)
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../../training"

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the MERL brdfs folder (*.binary files)}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PWD/outputs/full_experiments}
EXP_NAME=${EXP_NAME:-Stage1_MERL_run_1}
MAX_EPOCHS=${MAX_EPOCHS:-70}
RAYS_NUM=${RAYS_NUM:-500000}
export WANDB_MODE=${WANDB_MODE:-offline}

python train.py \
    dataset_folder="$DATA_ROOT" \
    output_folder="$OUTPUT_ROOT" \
    exp_output_root_path="$OUTPUT_ROOT/$EXP_NAME" \
    data=merl \
    renderer=rig_merl \
    material=merl_neural \
    experiment_name="$EXP_NAME" \
    model.optimizer.name=Adam8bit \
    model.optimizer.reset_latent_momentum_on_chunk_switch=False \
    model.optimizer.lr=2e-3 \
    model.optimizer.decoder_lr=2e-4 \
    model.loss.recon_loss.name=logrel \
    model.loss.recon_loss.log_space.logrel_ref=0.057 \
    model.loss.reg_loss.weight=0.0 \
    model.stage=1 \
    model.test=False \
    model.trainer.max_epochs=$MAX_EPOCHS \
    model.trainer.check_val_every_n_epoch=5 \
    model.trainer.log_every_n_steps=5 \
    model.trainer.limit_train_batches=8000 \
    model.grazing_ratio=0.05 \
    model.grazing_mode=contribution_decay \
    material.decoder.use_skip_connection=True \
    material.decoder.use_film=False \
    material.decoder.use_color_decomp=False \
    material.decoder.smooth_reg=False \
    material.latent_dim=24 \
    material.decoder.degree=3 \
    material.different_decoder=False \
    data.switch_iters=2000 \
    data.chunk_size=20 \
    data.filter_observations=False \
    data.rays_num=$RAYS_NUM \
    "$@"
