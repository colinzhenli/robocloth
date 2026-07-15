# Two-stage training & evaluation

All commands run from the repository root with the `robocloth` environment
active. Scripts wrap `training/train.py` (Hydra) with the exact paper
configuration; paths come in through environment variables.

## Environment

```bash
conda create -n robocloth python=3.10 -y && conda activate robocloth
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r envs/training.txt
# UBO2014 comparison only:
pip install Cython && pip install btf_extractor==1.7.0 --no-build-isolation
```

pytorch-lightning must stay 1.9.x (the trainers use the 1.x API). Training
logs default to `WANDB_MODE=offline`.

## Data layout (`DATA_ROOT`)

```
DATA_ROOT/
├── training_list_500.txt        # stage-1 material list (one id per line)
├── camera_factor.json           # per-material exposure segments (stage 1)
├── emitter_calibration.json     # LED angular-falloff table
├── 145/                         # one folder per material id
│   ├── observations_structured.npz   # dense observation tensor (stage 1)
│   ├── point_metadata.json
│   ├── scan_log.json
│   ├── rotated_camera.json
│   ├── bbox.json
│   └── hdr/*.png                # linear 16-bit views (stage 2 / evaluation)
└── ...
```

Download: see the README (Hugging Face `koalapenguin/cloth-brdf`;
per-material `hdr.tar` extracts to `hdr/` in place).

## Stage 1 — material prior

```bash
DATA_ROOT=... OUTPUT_ROOT=... bash scripts/train_stage1.sh
```

Customizable variables:

* `TRAINING_LIST` — material list (default `$DATA_ROOT/training_list_500.txt`).
* `MAX_EPOCHS` — default 100; the released checkpoint is epoch 60.
* `RAYS_NUM` — batch size in rays; reduce on GPU OOM.
* `EXP_NAME` — experiment/output folder name.

Output: `$OUTPUT_ROOT/$EXP_NAME/training/model_0.20_0.20/{epoch=N.ckpt,last.ckpt}`.
The released `checkpoints/stage1/Ours.ckpt` is this run's result — you can
skip stage 1 and use it directly. Hardware: one large GPU (paper: 80 GB;
48 GB works) and ~1.3 GB host RAM per material in the training list.

*Smoke test* (~20 min): point `TRAINING_LIST` at ~5 material ids and run
with `MAX_EPOCHS=3 model.trainer.limit_train_batches=300` — the loss drops
steeply within the first epoch.

## Stage 2 — dense per-material fit

```bash
DATA_ROOT=... OUTPUT_ROOT=... STAGE1_CKPT=/path/to/checkpoints/stage1/Ours.ckpt \
    bash scripts/train_stage2.sh 145
```

Decoder-only warm start from `STAGE1_CKPT` (then frozen); optimizes the
2048² latent texture, the parallax query, and the per-channel scale β.
Customizable variables:

* `MODEL=Bonn|MERL` — warm-start from the corresponding stage-1 decoder;
  `MODEL=PBR` — analytic Disney baseline (no checkpoint).
* `MAX_EPOCHS` — paper: 145→120, 226→100, 314/370/452→80 (default 100).

Outputs: checkpoints as above plus validation renders every 2 epochs
(`images/gt_view_<i>_0.png`, `images/result_view_<i>_0_psnr<PSNR>.png`).
The stage-2 dense loader preloads all training views: use a large-memory
node (> 512 GB for a full ~590-view material).

## Evaluation — reproducing the paper table

```bash
# single checkpoint
DATA_ROOT=... OUTPUT_ROOT=... bash scripts/eval_stage2.sh 145 \
    /path/to/checkpoints/stage2/RoboCloth/145/Ours_epoch112.ckpt

# full table: 5 materials x 4 models (~10 min each on a 48 GB GPU; resumable)
DATA_ROOT=... OUTPUT_ROOT=... CKPT_ROOT=/path/to/checkpoints/stage2/RoboCloth \
    bash scripts/run_table_ours.sh
```

Outputs per checkpoint: the validation PSNR (the paper metric), printed and
saved to `$OUTPUT_ROOT/eval_results/<mat>_<model>.json`, plus GT/prediction
renders of the held-out views. `run_table_ours.sh` ends by printing the
reproduced numbers next to the reference values — they match **Table
"Per-material reconstruction PSNR" (our held-out test set)** in the paper.

## Comparison experiments (paper baselines)

Scripts under `scripts/comparisons/` — stage-1 decoders trained on Bonn
(UBOFAB19) and MERL, and stage-2 transfer to the Bonn and UBO2014 test sets:

| Experiment | Script |
|---|---|
| Stage-1 on Bonn / MERL | `train_stage1_bonn.sh`, `train_stage1_merl.sh` |
| Stage-2 on Bonn materials (318 377 32 226 37; 120 ep) | `train_stage2_bonn.sh <mat>` |
| Stage-2 on UBO2014 (12 materials; 60 ep) | `train_stage2_ubo.sh <mat>` |
| UBO2014 table reproduction | `eval_stage2_ubo.sh` / `run_table_ubo.sh` |

`MODEL=Ours|Bonn|MERL|PBR` selects the decoder column (MERL β-init 0.1 as in
the paper). Bonn data download + preprocessing: `scripts/comparisons/bonn_data/`
(`get_UBOFAB19_*.sh`), then run
`python scripts/comparisons/generate_bonn_metadata.py <Bonn_folder>` once per
downloaded folder (writes the `bonn_point_metadata.json` index the loader
needs). MERL and UBO2014 data ship in the Hugging Face bundle
(`datasets/MERL`, `datasets/UBO2014`).

## Troubleshooting

* **GPU OOM** — lower `RAYS_NUM` (or `material.texture_resolution` in stage 2).
* **`emitter_calibration.json` not found** — scripts look in the material
  folder, then the dataset root; set `EMITTER_CALIB=/path/to/file`.
