# RoboCloth

Code release for the RoboCloth paper: a robotic acquisition pipeline and a
500-material cloth BRDF dataset, with a two-stage neural reconstruction
protocol and Mitsuba 3 rendering of the learned materials.

```
calibration/     offline rig calibration (geometry + radiometric) — code, docs, constants
reconstruction/  capture-time reconstruction: COLMAP + robot alignment -> observation tensors
training/        two-stage neural BRDF training + checkpoint evaluation
rendering/       Mitsuba 3 visualization (relighting + full scenes)
configs/         Hydra configs; configs/renderer/rig_constants.yaml = rig calibration record
scripts/         validated entry-point wrappers (training, evaluation, comparisons)
docs/            per-pipeline documentation
envs/            pinned python environments (training, rendering)
```

## Released assets

| Asset | Where |
|---|---|
| RoboCloth capture data (500 materials) | https://huggingface.co/datasets/koalapenguin/cloth-brdf |
| Pretrained checkpoints (stage 1 + stage 2), comparison datasets (MERL, UBO2014), render meshes | https://huggingface.co/datasets/koalapenguin/RoboCloth |

## Quickstart (~15 min + downloads)

Reproduce one paper number and one render from the released checkpoints:

```bash
# environment (see docs/training.md for details)
conda create -n robocloth python=3.10 -y && conda activate robocloth
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r envs/training.txt

# data: material 145 + calibration globals + the stage-2 checkpoint
huggingface-cli download koalapenguin/cloth-brdf --repo-type dataset \
    --include "globals/*" "materials/145/*" --local-dir cloth_brdf
mkdir -p DATA_ROOT && cp cloth_brdf/globals/* DATA_ROOT/ && mv cloth_brdf/materials/145 DATA_ROOT/145 \
    && tar -xf DATA_ROOT/145/hdr.tar -C DATA_ROOT/145
huggingface-cli download koalapenguin/RoboCloth --repo-type dataset \
    --include "checkpoints/stage2/RoboCloth/145/*" --local-dir ckpts

# evaluate: renders the held-out views, prints the paper's PSNR (28.44 dB)
DATA_ROOT=$PWD/DATA_ROOT OUTPUT_ROOT=$PWD/outputs \
    bash scripts/eval_stage2.sh 145 ckpts/checkpoints/stage2/RoboCloth/145/Ours_epoch112.ckpt
```

## The four pipelines

1. **Calibration** ([docs/calibration.md](calibration/README.md)) — one-time
   rig calibration: ChArUco intrinsics, Tsai hand–eye, turntable axis,
   ColorChecker CCM, and the grey-patch radiometric fit that yields the
   camera scale + LED angular profile. Results live in
   `configs/renderer/rig_constants.yaml` and the dataset `globals/`.
2. **Reconstruction** ([docs/reconstruction.md](docs/reconstruction.md)) —
   per capture session: COLMAP sparse SfM on LDR frames, Umeyama alignment
   to the robot frame, Menon debayering to linear HDR, and reprojection of
   every surface point into every view → the per-material observation
   tensor the training consumes.
3. **Two-stage training** ([docs/training.md](docs/training.md)) — stage 1
   learns a shared BRDF decoder + per-point latents over ~400 materials;
   stage 2 freezes the decoder and fits a dense 2048² latent texture per
   material. Includes evaluation scripts reproducing the paper tables and
   the Bonn/MERL/UBO2014 comparison experiments.
4. **Rendering** ([docs/rendering.md](docs/rendering.md)) — the trained
   BRDF as a Mitsuba 3 BSDF plugin: relight any mesh under an environment
   map (`rendering/relight/`), or render full scenes with per-object neural
   materials like the paper teaser (`rendering/scene/`).

## Reproducing the paper

Every table and qualitative figure maps to a documented command — see
[docs/reproduce_paper.md](docs/reproduce_paper.md). All evaluation paths in
this repository were verified against the original experiment code:
identical PSNR to full float precision on the released checkpoints, and
bit-identical training trajectories on smoke runs.
