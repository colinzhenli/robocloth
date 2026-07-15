#!/bin/bash
# ---------------------------------------------------------------------------
# Stage-2 checkpoint evaluation on a UBO2014 BTF material.
#
# Runs main.py with model.test=True: loads ALL weights from the trained
# stage-2 checkpoint (except emitter buffers) and executes the exact
# training-time validation step over the fixed held-out BTF angle combinations
# (val_view_ratio 0.2, seed 42) — producing the val/psnr metric reported in
# the cross-dataset transfer table of the paper, plus the same GT/prediction
# visualizations the trainer saves during validation.
#
# Usage:
#   DATA_ROOT=/path/to/BTF bash eval_stage2_ubo.sh <material> <CKPT> [TAG]
#   e.g. eval_stage2_ubo.sh felt01 checkpoints/stage2/UBO/felt01/Bonn_epoch60.ckpt
#
# TAG defaults to the checkpoint filename prefix (Ours/Bonn/MERL/PBR);
# TAG=PBR selects the Disney-PBR architecture.
# Results: $OUTPUT_ROOT/eval_results_ubo/<material>_<TAG>.json
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../../training"

MAT=${1:?usage: eval_stage2_ubo.sh <material> <CKPT> [TAG]}
CKPT=${2:?usage: eval_stage2_ubo.sh <material> <CKPT> [TAG]}
TAG=${3:-$(basename "$CKPT" | sed 's/_epoch.*//;s/\.ckpt//')}

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the folder with UBO2014 .btf files}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PWD/outputs/full_experiments}
EXP_NAME=${EXP_NAME:-Eval_UBO_${MAT}_${TAG}}
BTF_FILE=${MAT}_W400xH400_L151xV151.btf

if [ "$TAG" = "PBR" ]; then
    MATERIAL_OVERRIDES=(material=ubo_pbr_latent
                        material.disney=True material.anisotropic=True
                        material.soft_constraint=True)
else
    MATERIAL_OVERRIDES=(material=ubo_latent material.latent_dim=24
                        material.different_decoder=False
                        material.decoder.use_skip_connection=True
                        material.decoder.use_film=False material.decoder.use_color_decomp=False
                        material.decoder.degree=3 material.decoder.smooth_reg=False)
fi

EXP_DIR=$OUTPUT_ROOT/$EXP_NAME
mkdir -p "$EXP_DIR"

python train.py \
    dataset_folder="$DATA_ROOT" \
    output_folder="$OUTPUT_ROOT" \
    exp_output_root_path="$EXP_DIR" \
    data=ubo \
    data.btf_filename=$BTF_FILE \
    data.rays_num=500000 \
    data.valid_num=20 \
    renderer=multiarea_emitter \
    material.learnable_factor=True \
    material.predict_frame=True \
    "${MATERIAL_OVERRIDES[@]}" \
    experiment_name="$EXP_NAME" \
    model.stage=2 \
    model.test=True \
    model.continue_training=False \
    model.apply_cosine_weight=True \
    model.ckpt_path="$CKPT" \
    model.trainer.enable_checkpointing=False \
    'model.logger._target_=pytorch_lightning.loggers.CSVLogger' \
    '~model.logger.project' \
    "${@:4}"

# ---- collect the scalar metrics from the CSV logger --------------------------
RESULTS_DIR=$OUTPUT_ROOT/eval_results_ubo
mkdir -p "$RESULTS_DIR"
METRICS_CSV=$(ls -t "$EXP_DIR/$EXP_NAME"/version_*/metrics.csv 2>/dev/null | head -1)
python - "$METRICS_CSV" "$RESULTS_DIR/${MAT}_${TAG}.json" "$MAT" "$TAG" "$CKPT" <<'EOF'
import csv, json, sys
csv_path, out_path, mat, tag, ckpt = sys.argv[1:6]
vals = {}
with open(csv_path) as f:
    for row in csv.DictReader(f):
        for k, v in row.items():
            if v not in (None, "") and k.startswith("val/"):
                vals[k] = float(v)
res = {"material": mat, "model": tag, "ckpt": ckpt,
       "val_psnr": vals.get("val/psnr"), "val_loss": vals.get("val/loss")}
json.dump(res, open(out_path, "w"), indent=2)
print(f"[eval_stage2_ubo] {mat} / {tag}: val/psnr = {res['val_psnr']:.2f} dB -> {out_path}")
EOF
