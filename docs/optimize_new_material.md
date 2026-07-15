# Optimize a material from our dataset (stage 2)

Fit a render-ready neural material — a dense latent grid driven by our
pretrained BRDF decoder — for any of the 500 RoboCloth materials. Result: a
checkpoint you can drop straight into the renderer (see the README).

## 1. Environment

```bash
conda create -n robocloth python=3.10 -y && conda activate robocloth
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r envs/training.txt
```

## 2. Download one material + the pretrained decoder

```bash
# dense capture data for material 145 (~9 GB) -> ./DATA_ROOT
bash scripts/download_material.sh 145 ./DATA_ROOT

# pretrained stage-1 decoder (RoboCloth prior, ~2 GB)
hf download koalapenguin/RoboCloth --repo-type dataset \
    --include "checkpoints/stage1/Ours.ckpt" --local-dir ./ckpts
```

## 3. Run stage 2

```bash
DATA_ROOT=./DATA_ROOT OUTPUT_ROOT=./outputs \
STAGE1_CKPT=./ckpts/checkpoints/stage1/Ours.ckpt \
    bash scripts/train_stage2.sh 145
```

This freezes the decoder and optimizes the 2048² latent texture, the
parallax-aware query, and a per-channel scale for the chosen material. All
hyperparameters live in `configs/experiment/stage2.yaml` (paper defaults;
edit there to change epochs, batch size, texture resolution, ...).

Progress: validation renders of held-out views appear every 2 epochs under
`outputs/stage2_145_from_Ours/images/` (`result_view_*_psnr*.png` next to
their `gt_view_*` references). Checkpoints:
`outputs/stage2_145_from_Ours/training/model_0.20_0.20/last.ckpt`.

Hardware: one large GPU (48 GB works) and a large-memory node — the loader
preloads all ~590 training views (> 512 GB host RAM for a full material).

## 4. Check the result

```bash
DATA_ROOT=./DATA_ROOT OUTPUT_ROOT=./outputs bash scripts/eval_stage2.sh 145 \
    outputs/stage2_145_from_Ours/training/model_0.20_0.20/last.ckpt
```

prints the held-out PSNR and saves GT/prediction pairs. To render the
material on a mesh or in a scene, point the renderer's `materials.json` at
your new checkpoint — see the README.

Paper-trained checkpoints for the 5 test materials are already released
(`checkpoints/stage2/RoboCloth/`) if you want to skip this step.
