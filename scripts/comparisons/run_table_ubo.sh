#!/bin/bash
# ---------------------------------------------------------------------------
# Evaluate released UBO2014 stage-2 checkpoints and print the reproduced
# "Cross-dataset transfer to UBO2014" table next to the paper values.
#
# Usage:
#   DATA_ROOT=/path/to/BTF CKPT_ROOT=/path/to/checkpoints/stage2/UBO \
#       bash run_table_ubo.sh
#
#   CKPT_ROOT layout: $CKPT_ROOT/<material>/{Ours,Bonn,MERL,PBR}_epoch<N>.ckpt
#   Subsets: MATERIALS="felt01 felt03" MODELS="Bonn" bash run_table_ubo.sh
#
# Results accumulate in $OUTPUT_ROOT/eval_results_ubo/ and existing results
# are skipped, so the script can be interrupted and re-run.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")"

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the folder with UBO2014 .btf files}
CKPT_ROOT=${CKPT_ROOT:?set CKPT_ROOT to the released stage-2 UBO checkpoints}
OUTPUT_ROOT=${OUTPUT_ROOT:-$(cd ../.. && pwd)/outputs/full_experiments}
MATERIALS=${MATERIALS:-"fabric02 fabric04 fabric09 fabric11 felt01 felt03 felt05 felt10 carpet02 carpet07 carpet09 carpet12"}
MODELS=${MODELS:-"Ours Bonn MERL PBR"}
export OUTPUT_ROOT DATA_ROOT

for mat in $MATERIALS; do
  for model in $MODELS; do
    result=$OUTPUT_ROOT/eval_results_ubo/${mat}_${model}.json
    if [ -f "$result" ]; then
      echo "[run_table_ubo] skip $mat / $model (already have $result)"
      continue
    fi
    ckpt=$(ls "$CKPT_ROOT/$mat/${model}"_epoch*.ckpt 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then
      echo "[run_table_ubo] WARNING: no checkpoint for $mat / $model under $CKPT_ROOT/$mat — skipping"
      continue
    fi
    echo "[run_table_ubo] evaluating $mat / $model ($ckpt)"
    bash eval_stage2_ubo.sh "$mat" "$ckpt" "$model" || echo "[run_table_ubo] ERROR on $mat/$model (continuing)"
  done
done

python collect_eval_results_ubo.py "$OUTPUT_ROOT/eval_results_ubo"
