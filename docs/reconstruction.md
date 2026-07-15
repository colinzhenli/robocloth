# Reconstruction pipeline (capture → training-ready material folder)

Converts one raw capture session into the per-material dataset folder the
training consumes. Runs in the `robocloth` environment; COLMAP must be on
`PATH` for the SfM step.

## Data flow

```
capture rig writes:    <material>/scan_log.json     robot poses + light poses + turntable angle per frame
                       <material>/ldr/*.png         8-bit frames (COLMAP input)
                       <material>/hdr_raw/*.png     16-bit Bayer mosaics

colmap.sh              feature_extractor -> sequential_matcher -> mapper
                       => sparse/ (cameras/images/points3D)      [retried with
                       exhaustive matching if <90% of frames register]

reconstruct.py         1. debayer hdr_raw/ -> hdr/ (Menon 2007, WB gains)
                       2. undo turntable rotation per frame; fit one Umeyama
                          Sim(3) COLMAP-world -> robot-base; drop frames with
                          >16 mm residual
                       3. transform + crop point cloud to the sample footprint
                          (sample_size.json), IQR depth filter  => bbox.json
                       4. reproject every kept 3D point into every hdr/ view
                          => point_positions.npz, point_metadata.json,
                             observations_structured.npz  (rgbs (K,V,3) uint16,
                             xyz, point_ids, cam_pos, light_pos)
                       + rotated_camera.json, unmatched_scan_ids.json
```

## Single material

```bash
bash scripts/reconstruct_material.sh /path/to/DATA_ROOT/145 [gpu_id]
```

Runs COLMAP only if `sparse/` is missing, then the alignment + observation
build. Knobs: `NUM_WORKERS` (debayer/reprojection workers, default 8),
`COLMAP_TMP` (COLMAP scratch dir, default `/tmp/robocloth_colmap`),
`RECON_TMP` (debayer staging dir).

## Batch processing (optional multi-GPU scheduler)

```bash
python reconstruction/scheduler.py --dataset /path/to/DATA_ROOT --mode streaming
python reconstruction/scheduler.py --dataset /path/to/DATA_ROOT --status
```

Resource-aware state machine over all numeric material folders: per-GPU
COLMAP slots, CPU/disk gates, the 90 % registration-rate gate with automatic
sequential→exhaustive retry, and restart-safe per-material state
(`scheduler_state.json`). Quality flags after completion come from
`reconstruction/quality_check.py`.

## Preprocessing tools (`reconstruction/preprocess/`)

Used between capture and COLMAP when starting from raw camera output:
`debayering_multi_thread.py` (mosaics → hdr/ + ldr/),
`background_mask_multi_thread.py` (turntable-ellipse masking),
`purple_filter.py` (drop corrupted frames).

The calibration constants consumed here (hand–eye `R_c2g/t_c2g`, turntable
axis, intrinsics, sample rectangle) come from
`configs/renderer/rig_constants.yaml` — see [calibration/README.md](../calibration/README.md).
