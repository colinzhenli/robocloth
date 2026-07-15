# RoboCloth Public Release — Restructuring Plan

Working draft. Goal: a NEW clean codebase (current repos stay untouched) with four
pipelines — Calibration, Reconstruction, Two-stage training, Mitsuba visualization —
that is simple to read, reproduce, and maintain.

## 0. The full story (as-is inventory)

| Pipeline | Where it lives today | State |
|---|---|---|
| **Calibration** (offline, once per rig) | `recon/calibration/` (Tsai hand-eye `camera_calibration.py`, 3 turntable-axis fitters, ChArUco intrinsics in `axis_estimation.py`, CCM `color_matrix.py`) + the grey-patch radiometric flow **buried inside `trainers/stage2_trainer_merl.py`** (produces `linear_factor` + `emitter_calibration.json`) | Works, but scattered; results duplicated as hardcoded constants in ≥5 files |
| **Reconstruction** (per capture) | `recon/scheduler/job_scheduler.py` → `recon/colmap/colmap.sh` → `recon/calibration/shape_matching.py` (debayer → Umeyama robot alignment → 16 mm filter → crop/IQR → reprojection → `observations_structured.npz`) + `recon/preprocess/` | Coherent and already modular; 1300-line monolith + 1800-line scheduler need slimming |
| **Two-stage training** | `main.py`, `trainers/` (7 copy-paste variants), `model/neural_brdf_refactored.py` (5k lines incl. dead classes), `utils/path_tracing.py`, `config/` (large, stale keys) | Validated end-to-end in milestone 3; heavy dead weight around the live path |
| **Rendering** | SGHyperMaterials `gt_render` + `teaser-structured` (clean squashed branches exist) | Validated in milestone 3 |

## 1. Design principles

1. **Move, don't rewrite.** Every number is validated; the milestone eval harness is a
   regression oracle (145/Ours = 28.44 exactly, byte-identical per-view PSNRs;
   UBO felt01/Bonn = 33.60; etc.). Logic is relocated and renamed — not re-implemented.
2. **Repo layout mirrors the paper narrative** (§3 acquisition→calibration→reconstruction,
   §4 two-stage pipeline, §5 experiments/rendering). A reader holding the paper should
   navigate the code by section.
3. **One source of truth for rig calibration.** All calibrated constants (R_c2g, t_c2g,
   intrinsics, turntable center/axis, base2_to_base1, linear factors, LED radius/FWHM)
   live in ONE versioned file (`calibration/rig_constants.yaml`), interpolated into the
   hydra configs — never duplicated in scripts again.
4. **Only the paper path ships.** Everything not exercised by a released command is cut.
5. **Thin, env-var wrappers** for every user-facing action (the milestone-3 script
   pattern, already proven with reviewers in mind).
6. **Data formats are documented contracts** — one `docs/data_formats.md` covering every
   file (scan_log.json, observations_structured.npz, rotated_camera.json, ...), plus a
   small validator script.

## 2. Proposed layout (new repo, working name `robocloth`)

```
robocloth/
├── README.md                     # story, 15-min quickstart, HF links, paper-to-code map
├── docs/
│   ├── calibration.md            # procedure: intrinsics → hand-eye → turntable → radiometric
│   ├── reconstruction.md         # raw capture → material folder (data-flow diagram)
│   ├── training.md               # stage 1 / stage 2 (port of milestone_instruction.md)
│   ├── rendering.md              # cloth-on-bar + teaser
│   ├── data_formats.md           # schema of every produced/consumed file
│   └── reproduce_paper.md        # every table/figure → exact command (port of full_experiments.md)
├── envs/                         # two pinned requirement sets (training, rendering)
├── calibration/
│   ├── rig_constants.yaml        # ← THE calibration record (single source of truth)
│   ├── boards/                   # ChArUco/AprilTag board generators
│   ├── intrinsics_charuco.py     # (axis_estimation.py::calibrate_intrinsics_charuco)
│   ├── hand_eye.py               # (camera_calibration.py: Umeyama metric upgrade + Tsai)
│   ├── turntable_axis.py         # (keep ONE fitter: shape_matching_aixs_fit alternating+LM)
│   ├── color_matrix.py           # CCM from ColorChecker (+ 4000K WB)
│   └── emitter_radiometry/       # grey-patch flow EXTRACTED from stage2_trainer_merl:
│                                 #   fits linear_factor + emitter_calibration.json standalone
├── reconstruction/
│   ├── reconstruct.py            # shape_matching.py, split into stages:
│   │                             #   debayer / align (Umeyama+turntable) / filter /
│   │                             #   crop_bbox / reproject_observations
│   ├── colmap.sh, colmap_exhaustive.sh
│   ├── preprocess/               # debayer tool, background_mask, purple_filter, process_json
│   ├── scheduler.py              # job_scheduler slimmed (keep state machine + 0.90 gate)
│   └── checks.py                 # registration_check + quality_check merged
├── training/
│   ├── train.py                  # main.py with the dataset if/elif → a small registry
│   ├── evaluate.py               # promoted from "model.test=True" + milestone eval wrapper
│   ├── trainers/                 # base.py (shared: losses, PSNR, val-image saving, logging)
│   │                             # + stage1.py / stage2.py + thin bonn/merl/ubo adapters
│   ├── models/                   # decoder.py, latent (bank/texture), neural_geometry (Q),
│   │                             # emitters.py, disney_pbr.py  (only live classes)
│   ├── renderer/                 # forward renderer + path_tracing (single-bounce)
│   ├── datasets/                 # robocloth_{stage1,stage2}.py, bonn.py, merl.py, ubo.py
│   └── configs/                  # pruned hydra tree; rig constants referenced, not copied
├── rendering/                    # DECISION: merged clean copy of SGHyperMaterials
│   ├── render.py                 # gt_render entry (single object, env light)
│   ├── render_scene.py           # teaser entry (XML scene + per-object ckpts)
│   ├── brdf_plugin/              # custom_bsdf (mlpbrdf + material wrappers)
│   ├── configs/ + assets/        # cloth.obj, envmap.exr
└── scripts/                      # validated wrappers (train_stage1/2, eval, tables,
                                  # comparisons, smoke_test.sh) — direct port of
                                  # scripts/milestone3 + scripts/full_experiments
```

## 3. What gets cut (largest wins first)

- **Model zoo**: `model/neural_brdf.py` (old), `mipmap_brdf.py`, `basis_brdf.py`, MoE,
  compressed/isotropic variants — keep only decoder + MultiMaterialLatentBRDF +
  AnisotropicLatentTexturedModel + PBR baselines + emitters + GreyPatch (moves to calibration).
- **Trainer copy-paste**: 7 variants → base class with shared loss/PSNR/val-saving/logging;
  per-dataset subclasses become ~100-line files. (Behavior-preserving consolidation only;
  regression suite decides.)
- **Repo-root debris**: `test.py`, `pointlight_test.py`, `align_bit.py`, `svbrdf_js.js`,
  notebooks, `nvdiffrast/`, `btflib/`, `Bonn_visualizer/`, `mesh_test/`, wandb/log dirs,
  `.deb` file, stale READMEs (fold anything still true into docs/).
- **scripts/**: keep the validated wrappers + dataset_submission tooling; drop
  `scripts/jobs/stacked/` (~60 files), `scripts/debugging/`, one-off analysis scripts.
- **Config tree**: remove dead keys, commented-out graveyards (multiarea_emitter.yaml is
  50 % commented history), unused data/material/renderer groups.
- **Deps**: drop `viztracer`, `torchviz`, `open3d` (training side), `bpy`, `nerfstudio`,
  `tinycudann`, `pnoise`→lazy; **vendor `LinearWarmupCosineAnnealingLR`** (~40 lines) to
  drop `lightning-bolts` entirely. Keep torch 2.5.1 + PL 1.9.5 pins for v1.0 parity
  (PL-2.x migration is post-release work, not now).
- **Reconstruction**: drop the legacy chunk path (`save_points_pixel_data`,
  `scripts/reformat_data/convert_*`), alternative colmap flows (`colmap_known_pose.sh`
  etc.), 2 of the 3 turntable-axis fitters (archived, referenced from docs).

## 4. Specific cleanups worth doing during the move

1. **Extract the radiometric calibration** out of `stage2_trainer_merl.py` into
   `calibration/emitter_radiometry/` (grey-patch fit → `linear_factor` +
   `emitter_calibration.json`). This is the single most confusing placement in the
   current code.
2. **`rig_constants.yaml`**: kill the hardcoded R_c2g/t_c2g/axis copies in
   `axis_estimation.py`, `solve_rot_axis_colmap_space.py`, `build_camera.py`,
   `shape_matching_aixs_fit.py`, and the `/media/raid/...camera_factor.json` path inside
   `Stage2Trainer.__init__` (resolve from dataset root instead, same fallback semantics).
3. **Renames** (old → new, table kept in docs): `points_dense`→`stage1_dense`,
   `real_dense`→`stage2_dense`, `multiarea_emitter`→`robocloth_rig`,
   `shape_matching.py`→`reconstruct.py`, `from_Real`→`from_robocloth`, output subdir
   `Bonn-Theia2/`→ gone, `Ours.ckpt`→`robocloth_decoder.ckpt` (HF gets both names).
4. **`evaluate.py` as a first-class entry** instead of `model.test=True` folklore.
5. Split `reconstruct.py` into stage functions with a `--stage` flag so each step is
   independently runnable/debuggable (debayer / align / observations).

## 5. Verification strategy (the refactor's safety net)

Old repo = read-only oracle. After every port phase, run the regression suite:

| Check | Oracle | Pass criterion |
|---|---|---|
| Stage-2 eval, 145/Ours (+ felt01/Bonn UBO) | milestone results | val/psnr 28.44 / 33.60; per-view PSNR filenames byte-identical |
| Full tab:ours (5×4) | milestone run | ≤ ±0.01 dB per cell |
| Stage-1 + stage-2 smoke trains | milestone metrics.csv | loss curves match within noise |
| Reconstruction on 1–2 materials | existing material folders | `observations_structured.npz` bit-identical (agent-verified: current code already mirrors legacy path bit-for-bit) |
| Render, fixed seed/spp | milestone debug PNGs | visually identical / low pixel diff |
| Fresh-env install on clean machine | envs/*.txt | smoke_test.sh green |

## 6. Documentation plan

- **README.md**: 3-paragraph story (rig → dataset → two-stage → rendering), one
  quickstart that gives a result in ~15 min (download 1 material + 1 ckpt → evaluate →
  view renders), links (HF data/ckpts, paper, project page), paper-section → code map.
- **docs/** one file per pipeline, each with: purpose, input/output diagram, exact
  commands, knobs table, expected runtime/hardware. Calibration doc is procedure-ordered
  (boards → intrinsics → hand-eye → turntable → CCM → grey-patch radiometry) and states
  clearly which artifacts a dataset user can just take from the release vs. re-derive.
- **docs/reproduce_paper.md**: every table/figure ↔ command (direct port of the validated
  milestone_instruction.md + full_experiments.md).
- **docs/data_formats.md**: schema for every file in the material folder + globals +
  checkpoints (merge of paper supp tab:schema + this session's pipeline maps).

## 7. Phases

- **P0 — Decisions + skeleton** (this doc; user sign-off on §8).
- **P1 — Training core port** + regression green (biggest win, fully validated oracle).
- **P2 — Reconstruction port** (reconstruct.py split, scheduler slim) + bit-parity check.
- **P3 — Calibration port** (standalone scripts, rig_constants.yaml, extracted grey-patch
  flow) + docs; runnable demo depends on §8.5.
- **P4 — Rendering merge** (clean gt_render/teaser into `rendering/`).
- **P5 — Docs, smoke CI, fresh-machine dry run, tag v1.0.**

Each phase = separate PR-sized chunk against the new repo, with the regression suite run
at the end. Suggest P1 → P2 → P4 → P3 order if calibration decisions lag.

## 8. Open decisions (need user input)

1. **Repo name / org**: `robocloth`? Under which account (double-blind timing)?
2. **Rendering**: merge cleaned SGHyperMaterials into `rendering/` (recommended — one
   repo, no submodule friction, the clean branches are already small) vs keep submodule.
3. **Comparison experiments** (Bonn/MERL/UBO trainers+datasets): ship in main repo
   (recommended, they're small once consolidated) vs separate "experiments" extra.
4. **Scheduler**: ship the full 4-GPU job scheduler, or a simplified
   `reconstruct.py --material <folder>` single-material entry + scheduler as optional?
5. **Calibration runnability**: docs-only (scripts + procedure, constants provided) vs
   also releasing sample calibration captures (ChArUco scans, ColorChecker, grey-patch
   session) so users can re-run each solver end-to-end. Affects P3 scope.
6. **License** (dataset is CC-BY-NC-4.0; code MIT/Apache-2?).
7. **Naming sign-off** for §4.3 renames.
