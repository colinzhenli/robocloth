# Data formats

Schemas of every file produced/consumed by the pipelines. All images are
linear (no gamma); "mm" fields are millimetres in the robot base frame.

## Per-material capture folder (`DATA_ROOT/<mat_id>/`)

| File | Producer | Contents |
|---|---|---|
| `scan_log.json` | capture rig | per-frame records: `filename`, `scan_id`, `overall_id`, `camera_id`, gripper `position` (mm) + `rotation_matrix` (gripper→base), `position_light` (mm), `turn_angle` (deg), `emitter_id` |
| `ldr/*.png` | capture rig / preprocess | 8-bit demosaiced frames (COLMAP input only) |
| `hdr_raw/*.png` | capture rig | 16-bit Bayer mosaics (RGGB) |
| `hdr/*.png` | reconstruction (debayer) | 16-bit linear RGB, Menon-2007 demosaiced, WB gains applied; released as `hdr.tar` |
| `sparse/` | COLMAP | `cameras/images/points3D` (.bin + .txt) + `points3D.ply`; `points3D_transformed_filtered.ply` = robot-frame cropped cloud |
| `rotated_camera.json` | reconstruction | per-frame `camera_id`, `position` (mm), `rotation_matrix` (camera→base, includes the Umeyama scale — renormalize columns for a pure rotation) in the 0°-turntable world |
| `bbox.json` | reconstruction | `bbox_min/max/center/size` of the cropped sample cloud + `num_points` |
| `point_positions.npz` | reconstruction | `point_ids (V,) int`, `positions (V,3) float32` |
| `point_metadata.json` | reconstruction | `num_points`, `num_observations` (sizes the stage-1 latent bank) |
| `observations_structured.npz` | reconstruction | the training tensor: `rgbs (K,V,3) uint16` (0 = unobserved), `xyz (V,3) float32`, `point_ids (V,) int32`, `cam_pos (K,3)` mm, `light_pos (K,3)` mm — K frames × V surface points |
| `unmatched_scan_ids.json` | reconstruction | scan ids never registered by COLMAP |
| `excluded_high_error_scan_ids.json` | reconstruction | frames dropped by the 16 mm alignment gate |
| `hdr_crop_bboxes.json` | release packaging | per-view crop polygons applied to the released `hdr/` |

## Dataset root (`DATA_ROOT/`, released as `globals/`)

| File | Contents |
|---|---|
| `training_list_{100,300,442,500}.txt`, `test_list_*.txt` | material-id splits (one id per line) |
| `camera_factor.json` | `camera_factor_segments: [{id_start,id_end,factor}]` — maps id ranges to `linear_factor{1,2,3}` (exposure regimes; factor3 = factor2·8000/20000) |
| `emitter_calibration.json` | LED angular falloff: `{resolution_degrees, angle_range, max_cam_rad_ratio, data:{angle: ratio}}` (grey-patch calibration; `max_cam_rad_ratio` is the camera radiance scale) |
| `sample_size.json` | `sample_sizes: [{id_start,id_end,width,length}]` — physical footprint used to crop the point cloud |

## Checkpoints

* Stage 1 (`checkpoints/stage1/*.ckpt`): Lightning checkpoint;
  `state_dict['material.decoder.*']` = the shared decoder (all stage 2 loads),
  `material.point_latent_bank.weight` = per-point latent/normal/tangent bank.
  `hyper_parameters` stores the full Hydra config of the run.
* Stage 2 (`checkpoints/stage2/<set>/<mat>/<Model>_epoch<N>.ckpt`):
  `material.latent_texture.params (1,C,R,R)` dense texture (C = latent 24 +
  geometry 16 + frame 6), `material.decoder.*` (frozen copy),
  `material.factor (3,)` = per-channel scale β, `material.neural_geometry.*`
  = parallax query Q. Filename encodes the decoder source (Ours/Bonn/MERL)
  or PBR baseline, and the training epoch.

## Rig calibration record

`configs/renderer/rig_constants.yaml` — intrinsics, hand–eye `R_c2g/t_c2g`,
turntable center/axis, LED mount `R_l2g/t_l2g`, dual-base `base2_to_base1`,
radiometric `linear_factor*`, LED radius/FWHM. See
[calibration/README.md](../calibration/README.md) for provenance.
