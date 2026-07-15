"""Render the felt09 sphere scene with a selectable BRDF backend and lighting.

Configuration is split into reusable groups under `config/`:

    config/config.yaml          shared params (resolution, spp, lighting, camera, output base)
    config/renderer/gt.yaml     UBO BTF ground-truth interpolator
    config/renderer/neural.yaml UBOLatentBRDF (neural decoder)
    config/renderer/pbr.yaml    UBOPBRLatentBRDF (Disney PBR)

Usage:

    python rendering.py renderer=neural lighting=all
    python rendering.py renderer=gt    lighting=point
    python rendering.py renderer=pbr   lighting=constant

The output goes to ``<output_base>/<renderer_name>/<lighting>.png`` (and
``<lighting>_pre_tonemap.png``).  No CLI arguments — everything is in the
config files; override at the command line via Hydra dot-syntax
(``renderer=...``, ``lighting=...``, ``render.spp=64``, etc.).
"""
from __future__ import annotations

import gc
import os
import random
import time
from types import SimpleNamespace

import hydra
from omegaconf import DictConfig, OmegaConf

import drjit as dr

# Disable autodiff bookkeeping we don't need for forward rendering.
dr.set_flag(dr.JitFlag.LoopRecord, False)
dr.set_flag(dr.JitFlag.VCallRecord, False)

import numpy as np
import torch

import mitsuba as mi
# Variant is set inside main() from cfg.variant. Custom BSDF imports and
# mi.register_bsdf() must run AFTER set_variant because BSDF subclasses bind to
# the active variant at class-construction time.

from custom_bsdf.utils.cuda_manage import print_cuda_memory_info
from custom_bsdf.utils.scene_loader import load_uv_obj_to_mitsuba_scene


def _build_floor_dict(cfg: DictConfig):
    """Build the floor rectangle dict, or None to omit the floor entirely.

    `cfg.scene.floor_mode` selects the floor type:
        "none"    : no floor (default) — clean envmap-only background, matches
                    the convention in most modern BRDF/material paper figures
        "checker" : the legacy black/white checkerboard

    Other knobs (only consulted when floor_mode != "none"):
        `cfg.scene.floor_scale`     : half-width in world units. Mitsuba's
            rectangle is 2x2 in local space, so total world width = 2 × scale.
            Default 50 → 100×100 plane, effectively infinite at our camera
            distances.
        `cfg.scene.checker_uv_scale`: tile density when mode=checker. Tile size
            in world units = 2 × floor_scale / checker_uv_scale. Default 125
            keeps the same 0.8 world-unit tile size as the original 4×4 floor.
    """
    scene_cfg = getattr(cfg, "scene", None)
    floor_mode = str(getattr(scene_cfg, "floor_mode", "none") if scene_cfg else "none").lower()
    if floor_mode == "none":
        return None
    floor_scale = float(getattr(scene_cfg, "floor_scale", 50.0)) if scene_cfg else 50.0
    # Floor sits at y = -sphere_scale so the sphere's bottom (which now stays
    # centered at the world origin regardless of scale) rests on it.
    sphere_scale = float(getattr(scene_cfg, "sphere_scale", 1.0)) if scene_cfg else 1.0
    base = {
        "type": "rectangle",
        "to_world": (
            mi.ScalarTransform4f.translate([0, -sphere_scale, 0])
            @ mi.ScalarTransform4f.rotate([1, 0, 0], -90)
            @ mi.ScalarTransform4f.scale(floor_scale)
        ),
    }
    if floor_mode == "checker":
        checker_uv_scale = (
            float(getattr(scene_cfg, "checker_uv_scale", 125.0)) if scene_cfg else 125.0
        )
        base["bsdf"] = {
            "type": "diffuse",
            "reflectance": {
                "type": "checkerboard",
                "to_uv": mi.ScalarTransform4f.scale(checker_uv_scale),
            },
        }
        return base
    raise ValueError(
        f"cfg.scene.floor_mode must be one of 'none' | 'checker', got {floor_mode!r}"
    )


def build_scene_dict(cfg: DictConfig, args: SimpleNamespace) -> dict:
    """Build the Mitsuba scene dict, including only the requested emitters."""
    scene = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": int(cfg.render.max_depth)},
        "sensor": {
            "type": "perspective",
            # Horizontal FOV in degrees. Mitsuba's default is 45° (narrow,
            # makes the sphere fill the frame and the envmap looks "zoomed in").
            # Widen (e.g. 70-80°) to show more of the env background.
            "fov": float(getattr(cfg.camera, "fov", 45.0)),
            "to_world": mi.ScalarTransform4f.look_at(
                origin=list(cfg.camera.origin),
                target=list(cfg.camera.target),
                up=[0, 1, 0],
            ),
            "film": {
                "type": "hdrfilm",
                "width": int(cfg.render.width),
                "height": int(cfg.render.height),
            },
        },
        "shape": load_uv_obj_to_mitsuba_scene(args, cfg),
    }
    floor_dict = _build_floor_dict(cfg)
    if floor_dict is not None:
        scene["floor"] = floor_dict
    # Optional metal support bar: a thin `cylinder` with a `roughconductor`
    # (metal PBR) BSDF, meant to sit inside the cloth's top fold so a draped
    # mesh looks supported instead of floating. Coordinates are in the SAME
    # post-auto-fit space as the cloth (centered at origin, longest axis = 0.4).
    # Enable + position via Hydra, e.g.:
    #   +bar.enabled=true +bar.p0=[-0.25,0.12,0] +bar.p1=[0.25,0.12,0] +bar.radius=0.012
    bar_cfg = getattr(cfg, "bar", None)
    if bar_cfg is not None and bool(getattr(bar_cfg, "enabled", False)):
        bar_bsdf = {
            "type": "roughconductor",
            "material": str(getattr(bar_cfg, "material", "Al")),
            "alpha": float(getattr(bar_cfg, "alpha", 0.15)),
        }
        # `+bar.mesh=<path>` renders a real bar mesh (assumed pre-positioned in
        # the cloth's post-auto-fit space). Otherwise fall back to a procedural
        # cylinder from p0/p1/radius.
        bar_mesh = getattr(bar_cfg, "mesh", None)
        if bar_mesh:
            mp = str(bar_mesh)
            scene["bar"] = {
                "type": "obj" if mp.lower().endswith(".obj") else "ply",
                "filename": mp,
                "bsdf": bar_bsdf,
            }
        else:
            scene["bar"] = {
                "type": "cylinder",
                "p0": [float(v) for v in bar_cfg.p0],
                "p1": [float(v) for v in bar_cfg.p1],
                "radius": float(getattr(bar_cfg, "radius", 0.012)),
                "bsdf": bar_bsdf,
            }
    if cfg.lighting in ("all", "constant"):
        scene["constant_emitter"] = {
            "type": "constant",
            "radiance": {"type": "rgb", "value": 0.5},
        }
    if cfg.lighting in ("all", "point"):
        point_intensity = float(getattr(cfg, "point_intensity", 10.0))
        scene["point_emitter"] = {
            "type": "point",
            "position": [-2, 2, 0],
            "intensity": {"type": "rgb", "value": [point_intensity, point_intensity, point_intensity]},
        }
        # Optional second point light at (2, 2, 2): position vector is
        # perpendicular to the first ((-2)*2 + 2*2 + 0*2 = 0) and lies on
        # the camera-facing hemisphere, so the new highlight appears 90°
        # around the visible part of the sphere from the existing one.
        # Enabled only when point2_intensity > 0.
        point2_intensity = float(getattr(cfg, "point2_intensity", 0.0))
        if point2_intensity > 0:
            scene["point_emitter2"] = {
                "type": "point",
                "position": [2, 2, 2],
                "intensity": {"type": "rgb", "value": [point2_intensity, point2_intensity, point2_intensity]},
            }
    if cfg.lighting == "envmap":
        # HDRI environment map as the sole emitter. Use one with a sharp
        # directional lobe (e.g. visible sun) to get crisp highlights similar
        # to a directional light + soft fill.
        envmap_path = str(getattr(cfg, "envmap_path", "data/envmap.exr"))
        if not os.path.isabs(envmap_path):
            envmap_path = os.path.join(
                hydra.utils.get_original_cwd(), envmap_path
            )
        envmap_scale = float(getattr(cfg, "envmap_scale", 1.0))
        scene["env_emitter"] = {
            "type": "envmap",
            "filename": envmap_path,
            "scale": envmap_scale,
        }
    if cfg.lighting not in ("all", "point", "constant", "envmap"):
        raise ValueError(
            f"cfg.lighting must be one of 'point' | 'constant' | 'all' | 'envmap', "
            f"got {cfg.lighting!r}"
        )
    return scene


def render_with_batches(scene, total_spp: int, batch_spp: int):
    """Render `total_spp` in chunks, clearing CUDA / DrJit caches between batches."""
    accumulated = None
    n_batches = (total_spp + batch_spp - 1) // batch_spp
    t0 = time.time()
    for b in range(n_batches):
        bs = min(batch_spp, total_spp - b * batch_spp)
        seed = random.randint(0, 2**31 - 1)
        print_cuda_memory_info(f"before batch {b + 1}/{n_batches}")
        img = mi.render(scene, spp=bs, seed=seed)
        accumulated = img * bs if accumulated is None else accumulated + img * bs
        torch.cuda.empty_cache()
        dr.flush_malloc_cache()
        gc.collect()
        print_cuda_memory_info(f"after batch  {b + 1}/{n_batches}")
    print(f"render time: {time.time() - t0:.1f}s")
    return accumulated / total_spp


def attach_material_config(cfg: DictConfig) -> None:
    """Load `config/material/<material_type>.yaml` and attach it under `cfg.material`.

    `scene_loader` and `MLPBRDF` read `cfg.material.type` to pick the decoder
    class.  We resolve the right material yaml at runtime based on the active
    renderer config so each renderer can specify its own material module.
    """
    project_root = hydra.utils.get_original_cwd()
    yaml_path = os.path.join(project_root, "config", "material", f"{cfg.material_type}.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Material yaml not found: {yaml_path}")
    material_cfg = OmegaConf.load(yaml_path)
    OmegaConf.set_struct(cfg, False)
    cfg.material = material_cfg
    OmegaConf.set_struct(cfg, True)


def resolve_per_material_ckpt(cfg: DictConfig) -> None:
    """If the active renderer provides `ckpt_paths.<material_id>`, copy that into
    `cfg.ckpt_path`. Lets bonn-style renderers ship a per-material dict instead
    of relying on `${material_id}` interpolation, when the path templates differ
    too much between materials to share one string."""
    if "ckpt_paths" in cfg and cfg.material_id in cfg.ckpt_paths:
        OmegaConf.set_struct(cfg, False)
        cfg.ckpt_path = cfg.ckpt_paths[cfg.material_id]
        OmegaConf.set_struct(cfg, True)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    attach_material_config(cfg)
    resolve_per_material_ckpt(cfg)

    # Set the Mitsuba variant from cfg.variant (default cuda_ad_rgb) BEFORE
    # importing the custom BSDF plugins — they bind to the active variant on
    # import. cuda_ad_rgb keeps drjit arrays on GPU end-to-end; llvm_ad_rgb is
    # the legacy CPU path (still used historically for the BTF interpolator
    # before it was ported to torch+CUDA).
    variant = str(getattr(cfg, "variant", "cuda_ad_rgb"))
    print(f"Variant:   {variant}")
    mi.set_variant(variant)

    from custom_bsdf.cook_torrancebrdf import CookTorranceBRDF
    from custom_bsdf.mlp import MLPBRDF
    mi.register_bsdf("cooktorrancebrdf", lambda props: CookTorranceBRDF(props))
    mi.register_bsdf("mlpbrdf", lambda props: MLPBRDF(props))

    project_root = hydra.utils.get_original_cwd()
    out_dir = os.path.join(project_root, cfg.output_base, str(cfg.material_id), cfg.renderer_name)
    os.makedirs(out_dir, exist_ok=True)
    # Optional suffix appended to the lighting-mode basename. Useful for
    # comparison sweeps (e.g. `output_suffix=_pi20` writes all_pi20.png).
    suffix = str(getattr(cfg, "output_suffix", "") or "")
    out_path = os.path.join(out_dir, f"{cfg.lighting}{suffix}.png")

    print("=" * 64)
    print(f"Material:  {cfg.material_id}")
    print(f"Renderer:  {cfg.renderer_name}  (material: {cfg.material_type})")
    print(f"  ckpt:    {cfg.ckpt_path or '<n/a>'}")
    print(f"  use_btf: {cfg.use_btf}    btf_path: {cfg.btf_path}")
    print(f"Lighting:  {cfg.lighting}")
    print(f"Render:    {cfg.render.width}x{cfg.render.height}  "
          f"spp={cfg.render.spp}  batch_spp={cfg.render.batch_spp}  "
          f"max_depth={cfg.render.max_depth}")
    print(f"Output:    {out_path}")
    print("=" * 64)

    # scene_loader expects an args namespace with .model_path
    args = SimpleNamespace(model_path=str(cfg.ckpt_path))

    scene = mi.load_dict(build_scene_dict(cfg, args))

    print_cuda_memory_info("before render")
    image = render_with_batches(scene, int(cfg.render.spp), int(cfg.render.batch_spp))
    print_cuda_memory_info("after render")

    arr = np.array(image)
    # Tone mapping is OFF by default: write a SINGLE raw (pre-tonemap) image.
    # Enable the legacy global Reinhard map with `+tonemap=true`. (Previously we
    # always wrote both the tonemapped and the raw file and discarded one —
    # wasteful.) write_bitmap still applies sRGB display gamma either way.
    out_arr = arr / (1.0 + arr) if bool(getattr(cfg, "tonemap", False)) else arr
    # Guard against non-finite pixels (e.g. a diverged checkpoint whose extreme
    # latents make the decoder emit NaN/Inf). Identity for well-behaved renders.
    out_arr = np.nan_to_num(out_arr, nan=0.0, posinf=0.0, neginf=0.0)
    mi.util.write_bitmap(out_path, mi.Bitmap(out_arr))

    print(f"\nSaved {out_path}")
    print(f"  raw  min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")


if __name__ == "__main__":
    main()
