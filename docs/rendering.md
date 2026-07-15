# Rendering (Mitsuba 3)

The trained BRDF is wrapped as a custom Mitsuba 3 BSDF plugin (`mlpbrdf`).
Two self-contained entry points:

* `rendering/relight/` — one object (sphere, the bundled cloth, or any mesh)
  under a point light / constant light / environment map. Used for the
  paper's qualitative comparison figures.
* `rendering/scene/` — a pre-authored Mitsuba XML scene where selected shapes
  get their BSDF replaced by neural checkpoints (the paper teaser).

## Environment

```bash
conda create -n robocloth-render -c conda-forge python=3.11 -y && conda activate robocloth-render
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r envs/rendering.txt
```

## Relighting a trained material

Run from `rendering/relight/` (configs resolve relative to it):

```bash
cd rendering/relight
python rendering.py \
    renderer=ours_anisotropic material_id="'145'" \
    ckpt_path=/path/to/checkpoints/stage2/RoboCloth/145/Ours_epoch112.ckpt \
    lighting=envmap +scene.mesh=cloth +scene.mesh_scale=0.2 \
    camera.origin='[0,0,-1.5]' camera.target='[0,0,0]' \
    render.width=512 render.height=512 render.spp=16 render.batch_spp=4 \
    output_base=/path/to/render_out +output_suffix=_debug
```

Output: `<output_base>/145/ours_anisotropic/envmap_debug.png` (raw linear
sRGB, no tone mapping). Notes:

* `renderer=ours_anisotropic` for RoboCloth neural checkpoints,
  `renderer=ours_pbr` for the Disney-PBR baselines; UBO2014/Bonn checkpoints
  use `renderer=ours|bonn_neural|...` (see `config/renderer/`).
* `data/cloth.obj` + `data/envmap.exr` ship in the folder;
  `+scene.mesh=/abs/path.obj` renders any mesh (auto-centered/scaled),
  `+envmap_path=... +envmap_scale=N` swaps the environment map.
* **Paper figure scene** (draped cloth on a chrome bar; meshes from the
  Hugging Face `render_assets/`): add
  `+scene.mesh=.../onBars01_st_hp.ply +scene.mesh_scale=0.3
  +bar.enabled=true +bar.mesh=.../pole_spheres_v4.obj +bar.material=Cr
  +bar.alpha=0.1` and render at `2048², spp 512` for paper quality.
* Quality: debug = 512²/spp 16; paper = 2048²/spp 512. `batch_spp ≤ 4` is
  the GPU-memory ceiling — total spp accumulates in chunks. `max_depth=4`.

## Full scenes (teaser)

Run from `rendering/scene/`:

```bash
cd rendering/scene
export BRDF_CKPT_ROOT=/path/to/checkpoints/stage2      # per-object checkpoints
export TEASER_SCENE_ROOT=/path/to/teaser_assets        # contains room_2/room/room.xml

python rendering.py render.spp=16 render.batch_spp=4 resx=960 resy=540 \
    output_base=/path/to/render_out output_name=teaser_debug     # debug
python rendering.py output_base=... output_name=teaser           # paper: 4K, spp 1024
```

The `objects:` map in `config/config.yaml` assigns a renderer config
(= checkpoint) to each shape id; swap one with `objects.elm__2=ours_370`.
For your own scene, point `scene.xml_path` at any Mitsuba XML and list your
shape ids under `objects:` — unlisted shapes keep their XML BSDF.

## Why two subtrees

The two entry points evolved on separate branches with diverging BSDF
wrappers (the relight plugin evaluates two-sided with cosine handling for
GT-BTF comparison; the scene plugin adds XML BSDF injection and per-shape UV
tiling). They are kept verbatim to preserve validated behavior; unifying the
plugin is tracked as post-release cleanup.
