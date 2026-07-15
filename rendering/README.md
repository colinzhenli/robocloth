# Neural-BRDF rendering

Path-trace any Mitsuba 3 scene with our Stage-2 neural BRDF checkpoints
attached to selected shapes.

## Scene folder

A renderable scene is a folder containing:

```
my_scene/
├── scene.xml        # Mitsuba 3 scene; shapes carry ids
├── materials.json   # shape id -> checkpoint assignment
└── meshes/ ...      # anything scene.xml references relatively (optional)
```

`materials.json`:

```json
{
  "checkpoint_root": "/path/to/checkpoints/stage2/RoboCloth",
  "two_sided": false,
  "assignments": {
    "shape_id_in_xml": {"material": "145"},
    "another_shape":   {"material": "370", "uv_tiling": 12.0},
    "custom":          {"ckpt": "/absolute/path/to/some.ckpt"}
  }
}
```

- `"material": "<id>"` resolves to `<checkpoint_root>/<id>/Ours_epoch*.ckpt`
  (the glob must match exactly one file).  `"ckpt"` takes a path instead —
  absolute, or relative to `checkpoint_root`.
- Shapes not listed keep the BSDF from `scene.xml`.
- Optional per-assignment keys: `material_type` (defaults to
  `AnisotropicLatentTexturedModel`, the wrapper for our released Stage-2
  checkpoints; UBO/Bonn/PBR wrapper classes are also available), `uv_tiling`
  (texture repeats across the mesh UV, default 5.0), `two_sided`, `uv_inset`,
  `grazing_mask_deg`.
- Optional top-level keys: `two_sided` (default from `configs/render.yaml`:
  true — back-face hits render fabric instead of black), `uv_inset`,
  `grazing_mask_deg`, and `radiance` (`{shape_id: intensity}` area-emitter
  override).

Environment variables: `${VAR:-default}` and `$VAR` are expanded inside
`checkpoint_root` / `ckpt` values and inside `<default value="...">` elements
of `scene.xml` (the bundled examples use `BRDF_CKPT_ROOT`,
`TEASER_SCENE_ROOT` and `MESH_DIR` to relocate the checkpoint / asset
downloads).

## Rendering

```bash
conda activate fipt-mitsuba
python render.py scene=examples/cloth_on_bar render.spp=64 \
    output_base=/tmp/renders output_name=cloth_on_bar
```

Quality/output knobs live in `configs/render.yaml` (`render.spp`,
`render.batch_spp`, `render.width/height/fov/max_depth` — null keeps the
scene XML's own values — `tonemap`, `variant`, `output_base`,
`output_name`).  Output is `<output_base>/<output_name>.png` plus a linear
`.exr`.

## Bundled examples

- `examples/cloth_on_bar` — cloth draped over a metal bar, HDRI lighting
  (meshes via `$MESH_DIR`).
- `examples/teaser` — the paper-teaser room; 17 shapes mapped to 13 Stage-2
  checkpoints (room assets via `$TEASER_SCENE_ROOT`, checkpoints via
  `$BRDF_CKPT_ROOT`).
