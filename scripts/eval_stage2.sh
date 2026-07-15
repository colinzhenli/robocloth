#!/bin/bash
# ---------------------------------------------------------------------------
# Stage-2 checkpoint evaluation on a RoboCloth material.
#
# Runs main.py with model.test=True: this loads ALL weights from the trained
# stage-2 checkpoint (except the emitter buffers) and executes the exact
# Stage2Trainer.validation_step over the full held-out validation split
# (the same fixed 80/20 split, seed 42, used during training):
#   - renders every validation view with the trained model (spp = 16),
#   - saves GT / prediction images exactly like training-time validation
#     (images/gt_view_<i>_0.png and images/result_view_<i>_0_psnr<PSNR>.png),
#   - computes per-view PSNR (peak = per-view GT max over visible pixels) and
#     averages it over the full validation set -> the paper's val/psnr metric.
#
# The scalar result is parsed from the CSV logger output and written to
#   $OUTPUT_ROOT/eval_results/<MAT_ID>_<TAG>.json
#
# Usage:
#   DATA_ROOT=/path/to/capture_data bash scripts/milestone3/eval_stage2.sh <MAT_ID> <CKPT> [TAG]
#
#   <CKPT> is a trained stage-2 checkpoint, e.g.
#     Stage-2-Finals/Ours/145/Ours_epoch112.ckpt        (released checkpoints), or
#     $OUTPUT_ROOT/Stage2_Ours145_from_Ours_run_1/training/model_0.20_0.20/last.ckpt
#   [TAG] labels the result file (default: checkpoint filename prefix before
#     "_epoch", i.e. Ours/Bonn/MERL/PBR for the released checkpoints).
#     TAG=PBR selects the Disney-PBR architecture; anything else the neural one.
#
# Knobs:
#   SAVE_ALL_VIEWS=1   save images for every validation view (default: first 20)
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../training"

MAT_ID=${1:?usage: eval_stage2.sh <MAT_ID> <CKPT> [TAG]}
CKPT=${2:?usage: eval_stage2.sh <MAT_ID> <CKPT> [TAG]}
TAG=${3:-$(basename "$CKPT" | sed 's/_epoch.*//;s/\.ckpt//')}

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the dataset root (folder containing <mat_id>/ subfolders)}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PWD/outputs/milestone3}
EXP_NAME=${EXP_NAME:-Eval_Ours${MAT_ID}_${TAG}}
VALID_NUM=$([ "${SAVE_ALL_VIEWS:-0}" = "1" ] && echo -1 || echo 20)

DATASET_FOLDER=$DATA_ROOT/$MAT_ID
EMITTER_CALIB=${EMITTER_CALIB:-$DATASET_FOLDER/emitter_calibration.json}
[ -f "$EMITTER_CALIB" ] || EMITTER_CALIB=$DATA_ROOT/emitter_calibration.json

if [ "$TAG" = "PBR" ]; then
    MATERIAL_OVERRIDES=(material=learnable_pbr_texture_model material.disney=True)
else
    MATERIAL_OVERRIDES=(material=ani_latent_texture_model)
fi

EXP_DIR=$OUTPUT_ROOT/$EXP_NAME
mkdir -p "$EXP_DIR"

python train.py \
    output_folder="$OUTPUT_ROOT" \
    exp_output_root_path="$EXP_DIR" \
    dataset_folder="$DATASET_FOLDER" \
    data=real_dense \
    data.rays_num=400000 \
    data.use_fixed_val=False \
    data.debug=False \
    data.valid_num=$VALID_NUM \
    renderer=multiarea_emitter \
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
    model.test=True \
    model.continue_training=False \
    model.freeze_decoder=True \
    model.ckpt_path="$CKPT" \
    model.trainer.enable_checkpointing=False \
    'model.logger._target_=pytorch_lightning.loggers.CSVLogger' \
    '~model.logger.project' \
    "${@:4}"

# ---- collect the scalar metrics from the CSV logger --------------------------
RESULTS_DIR=$OUTPUT_ROOT/eval_results
mkdir -p "$RESULTS_DIR"
METRICS_CSV=$(ls -t "$EXP_DIR/$EXP_NAME"/version_*/metrics.csv 2>/dev/null | head -1)
python - "$METRICS_CSV" "$RESULTS_DIR/${MAT_ID}_${TAG}.json" "$MAT_ID" "$TAG" "$CKPT" <<'EOF'
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
print(f"[eval_stage2] material {mat} / {tag}: val/psnr = {res['val_psnr']:.2f} dB "
      f"(val/loss = {res['val_loss']:.4f})  ->  {out_path}")
EOF
