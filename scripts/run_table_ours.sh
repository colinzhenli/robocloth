#!/bin/bash
# ---------------------------------------------------------------------------
# Reproduce Table "Per-material reconstruction PSNR" (top block, our test set)
# from the released stage-2 checkpoints.
#
# Evaluates 5 materials x 4 models (RoboCloth/Bonn/MERL decoder warm-starts +
# Disney-PBR baseline) with eval_stage2.sh and prints the comparison against
# the numbers reported in the paper.
#
# Usage:
#   DATA_ROOT=/path/to/capture_data CKPT_ROOT=/path/to/Stage-2-Finals/Ours \
#       bash scripts/milestone3/run_table_ours.sh
#
#   CKPT_ROOT layout: $CKPT_ROOT/<mat>/{Ours,Bonn,MERL,PBR}_epoch<N>.ckpt
#   MATERIALS / MODELS can be overridden to evaluate a subset, e.g.
#     MATERIALS="145" MODELS="Ours" bash scripts/milestone3/run_table_ours.sh
#
# Each evaluation renders ~118 held-out views at spp 16 (roughly 10-30 min on
# a modern GPU); the full table is 20 evaluations. Results accumulate in
# $OUTPUT_ROOT/eval_results/ and already-evaluated (mat, model) pairs are
# skipped, so the script can be interrupted and re-run.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")"

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the dataset root}
CKPT_ROOT=${CKPT_ROOT:?set CKPT_ROOT to the released stage-2 checkpoints (Stage-2-Finals/Ours)}
OUTPUT_ROOT=${OUTPUT_ROOT:-$(cd ../.. && pwd)/outputs/milestone3}
MATERIALS=${MATERIALS:-"226 314 370 145 452"}
MODELS=${MODELS:-"Ours Bonn MERL PBR"}
export OUTPUT_ROOT DATA_ROOT

for mat in $MATERIALS; do
  for model in $MODELS; do
    result=$OUTPUT_ROOT/eval_results/${mat}_${model}.json
    if [ -f "$result" ]; then
      echo "[run_table] skip material $mat / $model (already have $result)"
      continue
    fi
    ckpt=$(ls "$CKPT_ROOT/$mat/${model}"_epoch*.ckpt 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then
      echo "[run_table] WARNING: no checkpoint for material $mat / $model under $CKPT_ROOT/$mat — skipping"
      continue
    fi
    echo "[run_table] evaluating material $mat / $model ($ckpt)"
    bash eval_stage2.sh "$mat" "$ckpt" "$model" || echo "[run_table] ERROR evaluating $mat/$model (continuing)"
  done
done

python collect_eval_results.py "$OUTPUT_ROOT/eval_results"
