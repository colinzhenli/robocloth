# Paper ↔ code map

Every experiment in the paper, with the command that reproduces it.
`DATA_ROOT` = RoboCloth capture data; `CKPTS` = the Hugging Face checkpoint
bundle (`koalapenguin/RoboCloth`); comparison datasets under `CKPTS/datasets/`.

| Paper item | What it shows | Command |
|---|---|---|
| Table "Per-material reconstruction PSNR", top block (our test set: 226, 314, 370, 145, 452) | in-domain stage-2 vs decoder sources | `DATA_ROOT=... CKPT_ROOT=$CKPTS/checkpoints/stage2/RoboCloth bash scripts/run_table_ours.sh` |
| Table "Cross-dataset transfer to UBO2014" (12 materials) | transfer of the frozen decoders | `DATA_ROOT=$CKPTS/datasets/UBO2014 CKPT_ROOT=$CKPTS/checkpoints/stage2/UBO bash scripts/comparisons/run_table_ubo.sh` |
| Table "Per-material reconstruction PSNR", bottom block (Bonn 318, 377, 32, 226, 37) | cross-dataset on Bonn | stage-2 runs via `scripts/comparisons/train_stage2_bonn.sh`; released checkpoints under `checkpoints/stage2/Bonn` (metric: per-image poly/gray weighted PSNR from the validation logs) |
| Stage-1 prior (decoder columns RoboCloth / Bonn / MERL) | shared-decoder pretraining | `scripts/train_stage1.sh` / `scripts/comparisons/train_stage1_{bonn,merl}.sh`; released as `checkpoints/stage1/{Ours,Bonn,MERL}.ckpt` |
| Dataset-size ablation (100/300/500) | stage-1 corpus size | `scripts/train_stage1.sh` with `TRAINING_LIST=$DATA_ROOT/training_list_{100,300}.txt`, then UBO transfer as above |
| Grazing-angle ablation | decay-loss variants | `scripts/train_stage1.sh model.grazing_mode={zero_exact,near_zero_brdf,contribution_decay} model.grazing_ratio=...` |
| Qualitative figures (cloth on bar, UBO/Ours/Bonn grids) | relit renders of the checkpoints | `rendering/relight/` — see [rendering.md](rendering.md), paper-scene block |
| Teaser | full room scene, per-object neural cloth | `rendering/scene/` — see [rendering.md](rendering.md) |

Verification status of this repository against the original experiment code:

* released-checkpoint evaluation: PSNR identical to full float precision
  (e.g. material 145/Ours = 28.442256927490234; UBO felt01/Bonn = 33.60);
* training: bit-identical loss trajectories on seeded smoke runs (stage 1)
  and matching behavior on stage 2;
* reconstruction: byte-compared outputs on a full material re-run;
* rendering: pixel-level agreement within Monte-Carlo noise.
