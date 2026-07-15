# Calibration pipeline (offline, once per rig)

Everything the capture rig needs to know about itself is calibrated **once,
before capturing**, and recorded in
[`configs/renderer/rig_constants.yaml`](../configs/renderer/rig_constants.yaml)
— the single source of truth consumed by both the reconstruction pipeline and
the differentiable renderer. The released dataset already ships with all
calibration results applied; the scripts here document how each constant was
obtained and let you recalibrate for a new rig.

Rig recap: two UFACTORY X-ARM 6 robots (base1 carries the camera, base2 the
LED), a turntable, a ChArUco fiducial next to the sample.

## Procedure (in order)

| Step | What is calibrated | Script | Input captures | Output (→ rig_constants.yaml) |
|---|---|---|---|---|
| 1 | Calibration boards | `boards/generate_AprilTag_board*.py` | — | printed ChArUco/AprilTag boards |
| 2 | Camera intrinsics + distortion | `charuco_calibration.py` (Zhang via ChArUco) — cross-checked against COLMAP self-calibration | board scans | `camera.intrinsics.*` |
| 3 | Hand–eye: camera → gripper | `hand_eye.py` (Umeyama metric upgrade + Tsai `cv2.calibrateHandEye`) | robot `scan_log.json` + COLMAP poses of the same session | `camera.R_c2g`, `camera.t_c2g` |
| 4 | Turntable axis + center | `turntable_axis.py` (alternating Umeyama ↔ axis refinement, Huber-robust LM) | fixed-camera scans across turntable angles | `emitter.turntable.center/axis` |
| 5 | Color-correction matrix + white balance | `color_matrix.py` (ColorChecker reference in `data/`, Bradford adaptation to the 4000 K LED) | one ColorChecker capture | 3×3 CCM (shipped with the dataset; images are released *without* it applied) |
| 6 | Radiometric scale + LED angular profile | grey-patch optimization — see below | one grey-patch capture session | `camera.linear_factor*` + `emitter_calibration.json` |

Fixed rig constants measured externally (not solved here): `emitter.R_l2g/t_l2g`
(LED mount), `emitter.base2_to_base1` (dual-robot base registration),
`emitter.radius` / `fwhm_deg` (LED spec sheet).

## Step 6: grey-patch radiometric calibration

The camera-counts-per-radiance scale (`linear_factor`) and the LED's angular
falloff table Λ(θ) (`emitter_calibration.json`, consumed by the renderer's
emitter) are fitted **by optimization through the differentiable renderer**: a
diffuse grey patch with known reflectance (0.9/π) is captured like a normal
material, then rendered with a constant emitter; the per-angle ratio between
captured and rendered radiance, binned at 1°, *is* the falloff table, and its
maximum is the camera scale.

Run it with the training stack (see `training/`):

```bash
python train.py <usual stage-2 overrides for the grey-patch capture folder> \
    material=graypatch model.emitter_calibration=True
```

Each validation epoch writes `emitter_calibration_table_epoch_<N>.json`
(same schema as the released `emitter_calibration.json`) plus a diagnostic
scatter plot. The fitting code lives in
`training/trainers/stage2_trainer_merl.py` (`_radiometric_calibration_step` /
`on_validation_epoch_end`).

## Note on per-material exposure (`camera_factor.json`)

Materials were captured under a small number of exposure regimes; the dataset's
`globals/camera_factor.json` maps material-id ranges to the matching
`linear_factor` variant (factor3 = factor2 · 8000/20000 for the short-exposure
captures). Stage-1 training consumes it automatically; stage-2/evaluation uses
the single global factor unless `data.camera_factor_json` is set (the paper
configuration keeps the global factor).
