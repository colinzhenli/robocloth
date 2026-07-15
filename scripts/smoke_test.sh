#!/bin/bash
# ---------------------------------------------------------------------------
# End-to-end smoke test (~40 min on one 48 GB GPU).
#
# Requires: DATA_ROOT (with material 145 + globals), STAGE2_CKPT (a released
# stage-2 checkpoint for material 145), and the robocloth environment.
#
#   DATA_ROOT=... STAGE2_CKPT=.../145/Ours_epoch112.ckpt bash scripts/smoke_test.sh
#
# 1. stage-1 training: 2 short epochs on a small material list — loss must drop
# 2. checkpoint evaluation: must print the paper PSNR for the given checkpoint
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT=${DATA_ROOT:?}
STAGE2_CKPT=${STAGE2_CKPT:?}
OUT=${OUTPUT_ROOT:-$PWD/outputs/smoke}
LIST=$OUT/smoke_list.txt
mkdir -p "$OUT"; printf '145\n' > "$LIST"

echo "== [1/2] stage-1 training smoke =="
DATA_ROOT=$DATA_ROOT OUTPUT_ROOT=$OUT TRAINING_LIST=$LIST EXP_NAME=smoke_stage1 \
bash scripts/train_stage1.sh \
    model.trainer.max_epochs=2 model.trainer.check_val_every_n_epoch=1 \
    model.trainer.limit_train_batches=100 \
    'model.logger._target_=pytorch_lightning.loggers.CSVLogger' '~model.logger.project'
python - "$OUT/smoke_stage1/smoke_stage1" <<'PY'
import csv, glob, sys
csvf = sorted(glob.glob(sys.argv[1] + "/version_*/metrics.csv"))[-1]
tl = [float(r["train/total_loss"]) for r in csv.DictReader(open(csvf)) if r.get("train/total_loss")]
assert tl[-1] < tl[0] * 0.7, f"loss did not drop: {tl[0]} -> {tl[-1]}"
print(f"stage-1 OK: loss {tl[0]:.3f} -> {tl[-1]:.3f}")
PY

echo "== [2/2] checkpoint evaluation =="
DATA_ROOT=$DATA_ROOT OUTPUT_ROOT=$OUT bash scripts/eval_stage2.sh 145 "$STAGE2_CKPT" Ours
python - "$OUT/eval_results/145_Ours.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"eval OK: val/psnr = {r['val_psnr']:.2f} dB")
PY
echo "SMOKE TEST PASSED"
