# Calibration code

Offline rig-calibration solvers. The full explanation of the procedure — what
each script calibrates, from which captures, and where the results live — is
in [docs/capture_pipeline.md](../docs/capture_pipeline.md). All resulting
constants are recorded in
[configs/renderer/rig_constants.yaml](../configs/renderer/rig_constants.yaml).

| Script | Solves |
|---|---|
| `boards/generate_AprilTag_board*.py` | printable ChArUco/AprilTag targets |
| `charuco_calibration.py` | camera intrinsics (Zhang) + board-based axis estimate |
| `hand_eye.py` | camera→gripper transform (Umeyama metric upgrade + Tsai) |
| `turntable_axis.py` | turntable rotation axis + center (robust alternating fit) |
| `color_matrix.py` | ColorChecker CCM + 4000 K white balance (`data/` holds the reference chart) |

The radiometric grey-patch flow (camera scale + LED angular profile) runs
through the training stack — see docs/capture_pipeline.md §"Radiometry".
