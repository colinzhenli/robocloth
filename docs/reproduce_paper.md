# Reproducing the paper

Every table and figure, with the command that reproduces it. Notation:
`DATA_ROOT` = RoboCloth capture data (download scripts in `scripts/`),
`CKPTS` = the [koalapenguin/RoboCloth](https://huggingface.co/datasets/koalapenguin/RoboCloth)
bundle (checkpoints, comparison datasets, render assets). Environments: see
the README (rendering) and [optimize_new_material.md](optimize_new_material.md)
(training); the UBO2014 experiments additionally need
`pip install Cython && pip install btf_extractor==1.7.0 --no-build-isolation`.

## Tables

| Paper table | Command |
|---|---|
| Per-material PSNR, our test set (226, 314, 370, 145, 452) | `DATA_ROOT=... OUTPUT_ROOT=... CKPT_ROOT=$CKPTS/checkpoints/stage2/RoboCloth bash scripts/run_table_ours.sh` |
| Cross-dataset transfer to UBO2014 (12 materials) | `DATA_ROOT=$CKPTS/datasets/UBO2014 OUTPUT_ROOT=... CKPT_ROOT=$CKPTS/checkpoints/stage2/UBO bash scripts/comparisons/run_table_ubo.sh` |
| Per-material PSNR, Bonn block (318, 377, 32, 226, 37) | released checkpoints under `$CKPTS/checkpoints/stage2/Bonn`; runs via `scripts/comparisons/train_stage2_bonn.sh` (metric = per-image poly/gray-weighted PSNR from the validation logs) |

Both table drivers print the reproduced numbers next to the paper values,
save per-checkpoint JSONs to `$OUTPUT_ROOT/eval_results*/`, and write the
same GT/prediction view renders the trainer saves during validation.

## Trainings behind the tables

All hyperparameters live in `configs/experiment/*.yaml`; scripts only take
paths. `MODEL=Ours|Bonn|MERL|PBR` selects the frozen-decoder source or the
Disney baseline (UBO-from-MERL applies the paper's β-init 0.1 automatically).

| Run | Command |
|---|---|
| Stage-1 decoder on RoboCloth / Bonn / MERL | `scripts/train_stage1.sh`, `scripts/comparisons/train_stage1_{bonn,merl}.sh` |
| Stage-2 on our materials | `scripts/train_stage2.sh <mat>` |
| Stage-2 on Bonn / UBO2014 test sets | `scripts/comparisons/train_stage2_{bonn,ubo}.sh <mat>` |
| Dataset-size ablation (100/300) | `TRAINING_LIST=$DATA_ROOT/training_list_{100,300}.txt bash scripts/train_stage1.sh` |
| Grazing-angle ablation | `bash scripts/train_stage1.sh model.grazing_mode={zero_exact,near_zero_brdf,contribution_decay}` |

Bonn data comes from the University of Bonn servers:
`scripts/comparisons/bonn_data/get_UBOFAB19_*.sh`, then
`python scripts/comparisons/generate_bonn_metadata.py <folder>` once per
downloaded folder. MERL and UBO2014 ship in `$CKPTS/datasets/`.

## Qualitative figures (renders of the checkpoints)

All figures use the bundled `rendering/examples/cloth_on_bar` scene — only
`materials.json` changes. The paper grids render at
`render.spp=512 render.width=2048 render.height=2048`.

**Our materials** (RoboCloth columns; supp. figure rows 226/314/370/145/452):

```json
{ "checkpoint_root": "$CKPTS/checkpoints/stage2/RoboCloth",
  "assignments": { "cloth": {"material": "145"} } }
```

**UBO2014 checkpoints** (cross-dataset figure; materials carpet02...felt10) —
select the decoder column via the checkpoint filename and set the UBO wrapper:

```json
{ "checkpoint_root": "$CKPTS/checkpoints/stage2/UBO",
  "assignments": { "cloth": {
      "ckpt": "felt01/Bonn_epoch60.ckpt",
      "material_type": "UBOLatentBRDF",
      "apply_cosine_at_eval": true } } }
```

**Bonn checkpoints**: as above with `"material_type": "BonnLatentBRDF"`,
`"apply_cosine_at_eval": true`, and checkpoints from
`checkpoints/stage2/Bonn/<mat>/`. The Bonn wrapper additionally reads the
per-material latent-grid shape from `bonn_point_metadata.json`
(`rendering/configs/material/BonnLatentBRDF.yaml` points at the Bonn_val
folder — see the Bonn data download above).

**Disney-PBR baselines**: `"material_type": "LearnablePBRTexturedModel"`
for `..._PBR` checkpoints on our materials
(`"UBOPBRLatentBRDF"`/`"BonnPBRLatentBRDF"` + `apply_cosine_at_eval` for the
UBO/Bonn PBR checkpoints).

**Teaser**: `rendering/examples/teaser` as shipped (README, bundled-examples
section).

## Verification status of this repository

Checked against the original experiment code:

* released-checkpoint evaluation: PSNR identical to full float precision
  (material 145/Ours = 28.442256927490234; UBO felt01/Bonn = 33.60);
* training: bit-identical loss trajectories on seeded smoke runs (stage 1),
  clean convergence on stage 2; experiment configs resolve identically to
  the original run commands;
* reconstruction: full re-run of a material byte-matches the released data
  (observation tensor, points, debayered HDR identical; poses within 1e-13 mm);
* rendering: unified renderer matches the original branch renders at the
  Monte-Carlo noise floor on both bundled examples.
