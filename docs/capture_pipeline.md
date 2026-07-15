# Capture pipeline: calibration & reconstruction

How the RoboCloth dataset was produced. This document explains the rig
calibration (done once, offline) and the per-session reconstruction that
turns raw robot captures into the released material folders. It is written
for understanding — the released data already contains every output
described here, and reproducing it requires our physical rig. The code lives
in `calibration/` and `reconstruction/`; all resulting constants are recorded
in [`configs/renderer/rig_constants.yaml`](../configs/renderer/rig_constants.yaml).

## The rig

Two UFACTORY X-ARM 6 robot arms inside a matte-black enclosure: one carries
the camera, one carries an LED. A ferromagnetic turntable holds the cloth
sample (fixed flat by corner magnets) and adds a third degree of freedom —
rotating the sample reuses each arm pose at several sample-relative azimuths.
A ChArUco fiducial next to the sample supports calibration and drift checks.
Each material is captured in one automated ~20-minute session: ~580 steps,
each recording a raw Bayer HDR frame plus the robot-logged camera pose,
light pose, and turntable angle (`scan_log.json`).

## Offline calibration (once per rig)

Performed in this order; each step's solver is in `calibration/`:

1. **Calibration boards** — `boards/generate_AprilTag_board*.py` produces the
   printed ChArUco/AprilTag targets.
2. **Camera intrinsics** — `charuco_calibration.py` (Zhang's method on
   ChArUco captures), cross-checked against COLMAP self-calibration; the
   shipped intrinsics use a SIMPLE_RADIAL model (focal, principal point, one
   radial coefficient).
3. **Hand–eye (camera → gripper)** — `hand_eye.py`: COLMAP camera centres of
   a calibration session are metric-scaled to the robot frame with a closed-
   form Umeyama Sim(3) fit, then Tsai's method (`cv2.calibrateHandEye`)
   recovers the fixed camera-to-flange transform `R_c2g, t_c2g`.
4. **Turntable axis** — `turntable_axis.py`: alternating refinement (Umeyama
   alignment ↔ least-squares center ↔ Levenberg–Marquardt axis step, Huber
   robust) over fixed-camera scans at many table angles, yielding the
   rotation axis and center used to undo the turntable rotation per frame.
5. **Color** — `color_matrix.py`: a 3×3 color-correction matrix fitted on a
   ColorChecker capture (Lab-D50 reference → Bradford-adapted to the 4000 K
   LED). The released images are white-balanced sensor RGB *without* the CCM
   applied, so users keep colorimetric control; the matrix ships with the
   dataset.
6. **Radiometry (camera scale + LED profile)** — a grey patch of known
   reflectance (0.9/π) is captured like a material and *rendered* with the
   differentiable renderer under a constant emitter; the captured/rendered
   ratio per emitter angle, binned at 1°, gives the LED angular falloff
   Λ(θ) (`emitter_calibration.json`) and its maximum gives the camera
   counts-per-radiance scale (`linear_factor`). Materials were captured
   under a few exposure regimes; `globals/camera_factor.json` maps material
   ids to the matching factor.

Fixed constants measured externally: the LED mount transform (`R_l2g/t_l2g`),
the dual-robot base registration (`base2_to_base1`), and the LED's physical
radius and beam FWHM (spec sheet).

## Per-session reconstruction (during/after capture)

`reconstruction/` — orchestrated per material by
`scripts/reconstruct_material.sh` (or the multi-GPU `scheduler.py` for whole
capture batches):

1. **Sparse SfM** — COLMAP (feature extraction → sequential matching →
   mapping) on the 8-bit LDR copies. A registration gate requires ≥ 90 % of
   frames to register; failures retry once with exhaustive matching.
2. **Debayering** — the 16-bit Bayer mosaics (`hdr_raw/`) are demosaiced with
   Menon 2007 and white-balance gains into the linear 16-bit `hdr/` views.
   No gamma or tone mapping is ever applied.
3. **Robot-frame alignment** — each frame's robot-logged pose is lifted to a
   common 0°-turntable world (undoing the table angle about the calibrated
   axis); one Umeyama Sim(3) then aligns the COLMAP reconstruction to this
   metric robot frame. Frames with > 16 mm residual are discarded
   (`excluded_high_error_scan_ids.json`); the refined per-frame cameras are
   written to `rotated_camera.json`.
4. **Sample cropping** — the aligned sparse cloud is cropped to the sample's
   recorded physical footprint (`globals/sample_size.json`) and depth-
   filtered by the IQR rule, producing `bbox.json` and the filtered cloud.
5. **Observation tensor** — every kept 3D point is reprojected into every
   HDR view (SIMPLE_RADIAL projection) and its RGB bilinearly sampled,
   assembling `observations_structured.npz`: `rgbs (K frames × V points × 3)`
   uint16 with zeros marking missing observations, plus point positions and
   per-frame camera/light positions. This single ~1.6 GB tensor is what
   stage-1 training consumes — no per-frame image decoding at train time.

Quality checks (`reconstruction/checks`): the COLMAP registration gate, the
alignment-residual statistics, and post-run warning flags (excess excluded
frames, catastrophic translation error). File-level schemas for everything
above: [data_formats.md](data_formats.md).
