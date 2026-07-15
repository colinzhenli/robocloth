# SGHyperMaterials — UBO2014 BTF Renderer (BTF / Neural / PBR Disney / Bonn-pretrain)

A Mitsuba 3 renderer that compares ground-truth UBO2014 BTF measurements against three neural decoders (UBOLatentBRDF from-Real, UBOPBRLatentBRDF Disney, UBOLatentBRDF from-Bonn-pretrain) on the same scene under selectable lighting.

---

## Part 1 — Changes & Conventions

### Changes
- **Removed empirical BRDF scale factors** (e.g. the `camera_factor` multiplier used in earlier sessions). The decoder's raw output is fed straight into Mitsuba and a single global Reinhard tone-map (`x / (1 + x)`) is applied to the linear HDR film output before saving the PNG.
- **Separated lighting modes** — `lighting=point` (point light only), `lighting=constant` (constant envmap only), `lighting=all` (both, the original setup). Toggled per-render via the same code path, no separate scene files.
- **Fixed the UV-flip inconsistency.** Previously the neural / PBR materials had `uv_flip_v: true` while `BTF.py` had no flip → the same mesh UV looked up *different* texels in the GT vs. neural paths. Now all paths use `uv_flip_v: false` end-to-end, matching the elerac reference and the training dataloader's row-major pixel layout.

### Conventions confirmed
- **The BTF data has the cosine term baked in** (`BTF ≈ BRDF × cos`). Empirical sweep over `theta_l` shows `BTF / cos` is roughly constant ≈ 0.17 for felt09, consistent with a near-Lambertian material plus implicit cos foreshortening. Same convention as `elerac/btf-rendering` and matches Mitsuba's expectation that `BSDF::eval` returns `BSDF × cos`.
- **Neural / PBR decoder output × cosine ≈ BTF** (when training used `apply_cosine_weight=True`). At inference the renderer multiplies the decoder output by `NoL` to produce `BSDF × cos` — same form as the GT BTF. Bonn-pretrain checkpoints used `apply_cosine_weight=False`, so the network output already includes cos and the renderer skips the multiplication for those (controlled by `apply_cosine_at_eval` in the renderer yaml).
- **Channel order** verified end-to-end as RGB (training reads BGR→RGB once at load; the renderer's BTF.py does an equivalent net-one-flip). felt01 renders blue, matching the official UBO2014 page.

---

## Part 2 — How to use

### Run a single render

```bash
conda activate fipt-mitsuba
python rendering.py renderer=neural lighting=all material_id=felt09
```

`renderer` ∈ `{gt, neural, pbr, bonn}`, `lighting` ∈ `{point, constant, all}`, `material_id` ∈ `{felt09, felt01}` (any string for which the ckpt / BTF paths exist).

### Change config

Hydra dot-syntax overrides any field at the command line:

```bash
python rendering.py renderer=pbr material_id=felt01 \
    render.spp=64 render.batch_spp=4 render.max_depth=4 \
    render.width=512 render.height=512
```

Or edit the yaml directly:

| File | What lives there |
|---|---|
| [`config/config.yaml`](config/config.yaml) | shared params: render (`spp`, `width/height`, `max_depth`, `batch_spp`), `lighting`, `camera`, `material_id`, `output_base` |
| [`config/renderer/{gt,neural,pbr,bonn}.yaml`](config/renderer/) | per-renderer: `material_type`, `ckpt_path` (or `ckpt_paths` dict), `use_btf`, `btf_path`, `apply_cosine_at_eval` |
| [`config/material/{UBOLatentBRDF,UBOPBRLatentBRDF}.yaml`](config/material/) | model-architecture hyperparams (latent dim, predict_frame, etc.); auto-loaded based on `material_type` |

### Customize a checkpoint or BTF path

Every decoder renderer yaml has the same 7 fields:

```yaml
# config/renderer/<name>.yaml
# @package _global_
renderer_name: <name>
material_type: UBOLatentBRDF | UBOPBRLatentBRDF
use_btf: false                        # true only for the gt path
btf_path: /media/raid/cloth/BTF/${material_id}_W400xH400_L151xV151.btf
ckpt_path: <path with ${material_id}> # "" for gt
apply_cosine_at_eval: true            # false only for trainers with apply_cosine_weight=False
```

`${material_id}` is interpolated at runtime, so adding a new material just means dropping the matching `.btf` and `.ckpt` files in the standard locations.

**Adding a new decoder (e.g. MERL pretrain):**
1. Create `config/renderer/merl.yaml` by copy-pasting `neural.yaml` and changing `renderer_name: merl` plus the `ckpt_path` template.
2. Append `"merl"` to the `ROWS` list at the top of [`diagnose_grid_3x3.py`](diagnose_grid_3x3.py).
3. Run `python rendering.py renderer=merl ...` and rebuild the grid.

If a decoder ever needs *non*-templated paths (filenames differ between materials), set `ckpt_path: ""` and add a `ckpt_paths: {felt09: ..., felt01: ...}` dict; `rendering.py:resolve_per_material_ckpt()` will pick the right entry.

### Output structure

Renders and Hydra job logs live **outside the repo** so git doesn't track or upload them:

```
/media/raid/cloth/output/Misuba_rendering/
├── output/render/<material_id>/<renderer_name>/<lighting>.png            # tone-mapped (Reinhard)
├── output/render/<material_id>/<renderer_name>/<lighting>_pre_tonemap.png # raw linear HDR
├── output/render/<material_id>/_compare/grid_3x3.png                      # 4×3 comparison grid
├── output/render/<material_id>/_compare/headline.png                      # 4-tile horizontal strip (lighting=all)
├── output/render/<material_id>/_compare/diff_*.png                        # cross-renderer R↔B diff maps
└── outputs/<date>/<time>/.hydra/                                          # Hydra job snapshots
```

Change the location by editing `output_base` in [`config/config.yaml`](config/config.yaml) (and `hydra.run.dir` for the Hydra logs).

### How to read the figures

Inside each `output/render/<material_id>/`:

- **`<renderer>/<lighting>.png`** — the actual render after Reinhard tone-map (`x/(1+x)`). This is what you'd put in a paper figure.
- **`<renderer>/<lighting>_pre_tonemap.png`** — same render *before* tone-map, i.e. the linear HDR film output stored as PNG. Use to inspect raw radiance / clipping.
- **`_compare/grid_3x3.png`** — 4×3 grid: rows are the four renderers (gt → neural → pbr → bonn), columns are lighting (all → point → constant). Read across a row to see how one decoder behaves under different lighting; read down a column to compare decoders for the same lighting condition. The `gt` row is the reference; the others should match it visually.
- **`_compare/headline.png`** — a 1×4 horizontal strip of the four `lighting=all` renders. Quick at-a-glance "do all four agree?" view.
- **`_compare/diff_<a>_vs_<b>.png`** — per-pixel signed difference between two `lighting=all` renders. **Red = `<a>` is brighter than `<b>`**, **blue = `<b>` is brighter than `<a>`**, black = agree. Magnitude is amplified ×4 for visibility, so any visible color is small in absolute terms. Use to localize *where* on the sphere two renderers disagree (e.g. terminator, weave anisotropy direction).

### Render the full comparison matrix and rebuild grids in one shot

Hydra `--multirun` (alias `-m`) sweeps every combination in a single command:

```bash
conda activate fipt-mitsuba && \
python rendering.py -m \
    renderer=gt,neural,pbr,bonn \
    lighting=all,point,constant \
    material_id=felt09,felt01 \
&& python diagnose_grid_3x3.py felt09 \
&& python diagnose_grid_3x3.py felt01
```

That's 24 renders (2 materials × 4 renderers × 3 lighting modes) + 2 grids. Add `merl` to the `renderer=` list once it exists. Use `python diagnose_grid_3x3.py <material_id>` on its own to rebuild a grid after re-rendering only some cells.
