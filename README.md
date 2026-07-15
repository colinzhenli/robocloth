# RoboCloth

500 real cloth materials, captured robotically and reconstructed as neural
BRDFs you can drop into any Mitsuba 3 scene. This README shows the main
application — **rendering our materials on your own scene**. The other entry
points each have their own guide:

| I want to... | Guide |
|---|---|
| Optimize a new material from our dataset (with the pretrained decoder) | [docs/optimize_new_material.md](docs/optimize_new_material.md) |
| Train the shared BRDF decoder from scratch | [docs/train_decoder.md](docs/train_decoder.md) |
| Reproduce the paper's numbers and figures | [docs/reproduce_paper.md](docs/reproduce_paper.md) |
| Understand how the dataset was captured (calibration + reconstruction) | [docs/capture_pipeline.md](docs/capture_pipeline.md) |

Released data: capture dataset at
[koalapenguin/cloth-brdf](https://huggingface.co/datasets/koalapenguin/cloth-brdf),
checkpoints + render assets at
[koalapenguin/RoboCloth](https://huggingface.co/datasets/koalapenguin/RoboCloth).

## Render our materials on your scene

### 1. Environment

```bash
conda create -n robocloth-render -c conda-forge python=3.11 -y && conda activate robocloth-render
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r envs/rendering.txt
```

### 2. Get checkpoints

```bash
hf download koalapenguin/RoboCloth --repo-type dataset \
    --include "checkpoints/stage2/RoboCloth/*" --local-dir ./ckpts
```

Any of the released stage-2 checkpoints is a complete material (a latent
texture + decoder); checkpoints you optimize yourself
([guide](docs/optimize_new_material.md)) work identically.

### 3. Describe your scene

A renderable scene is a folder:

```
my_scene/
├── scene.xml        # ordinary Mitsuba 3 scene; shapes carry ids
├── meshes/…         # UV-mapped geometry referenced by the XML
└── materials.json   # which shapes get which material
```

```json
{
  "checkpoint_root": "./ckpts/checkpoints/stage2/RoboCloth",
  "assignments": {
    "sofa":    {"material": "370"},
    "curtain": {"material": "145", "uv_tiling": 12.0}
  }
}
```

Shapes not listed keep their XML BSDF. Meshes must have UVs (the material is
a latent *texture*) — if yours don't, generate them with
`python rendering/tools/generate_uv.py your_mesh.obj` (headless-Blender
Smart-UV; needs `pip install bpy==3.6.0` under Python 3.10), and use
`uv_tiling` to set the weave repeat density. Full `materials.json` reference:
[rendering/README.md](rendering/README.md).

### 4. Render

```bash
cd rendering
python render.py scene=/path/to/my_scene output_base=./out output_name=my_render
```

Quality and output settings (spp, resolution, tone mapping, path depth) live
in [rendering/configs/render.yaml](rendering/configs/render.yaml) — edit
there or override on the command line (`render.spp=512`).

### Bundled examples

Cloth draped over a bar under an environment map:

```bash
hf download koalapenguin/RoboCloth --repo-type dataset \
    --include "render_assets/*" --local-dir ./assets
cd rendering
MESH_DIR=../assets/render_assets BRDF_CKPT_ROOT=../ckpts/checkpoints/stage2 \
    python render.py scene=examples/cloth_on_bar output_base=./out output_name=cloth_on_bar
```

The paper teaser (a room where sofas, pillows, carpet and curtains are all
our reconstructed cloths):

```bash
hf download koalapenguin/RoboCloth --repo-type dataset \
    --include "render_assets/teaser_room/*" --local-dir ./assets
cd rendering
TEASER_SCENE_ROOT=../assets/render_assets/teaser_room BRDF_CKPT_ROOT=../ckpts/checkpoints/stage2 \
    python render.py scene=examples/teaser output_base=./out output_name=teaser
```

Draft quality for both: append `render.spp=16 render.width=960 render.height=540`.

## Repository map

```
rendering/        Mitsuba 3 integration (render.py, BRDF plugin, examples, UV tools)
training/         two-stage neural BRDF training + evaluation
scripts/          download / train / evaluate entry points
configs/          Hydra configs; experiment/*.yaml hold all hyperparameters,
                  renderer/rig_constants.yaml is the rig-calibration record
calibration/      offline rig-calibration solvers (see docs/capture_pipeline.md)
reconstruction/   capture-time reconstruction (COLMAP + robot alignment + tensors)
docs/             the guides linked above + data_formats.md
envs/             pinned environments (training, rendering)
```

Everything in this repository is regression-verified against the original
experiment code — see the verification notes in
[docs/reproduce_paper.md](docs/reproduce_paper.md).
