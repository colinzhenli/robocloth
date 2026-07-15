"""Render the teaser room scene with per-object neural BRDFs from checkpoints.

Configuration is split into reusable groups under ``config/``:

    config/config.yaml          shared params (scene, render, output, per-object map)
    config/renderer/ours.yaml   AnisotropicLatentTexturedModel + Stage2_Ours_Material-226 ckpt
    config/renderer/legacy.yaml AnisotropicLatentTexturedModel + original Haoran ckpt
    config/material/*.yaml      per-material decoder configs (unchanged)

Usage:

    # Render with defaults
    python rendering.py

    # Send one object to the legacy checkpoint
    python rendering.py objects.elm__2=legacy

    # Override render quality
    python rendering.py render.spp=64 render.batch_spp=4

Output goes to ``<output_base>/<output_name>.png`` (and ``_pre_tonemap.png``).
"""
from __future__ import annotations

import gc
import os
import random
import time

import hydra
from omegaconf import DictConfig, OmegaConf

import drjit as dr

dr.set_flag(dr.JitFlag.LoopRecord, False)
dr.set_flag(dr.JitFlag.VCallRecord, False)

import numpy as np
import torch

import mitsuba as mi
# Variant is set from cfg.variant at runtime in main().

# Imports of custom BSDFs deferred until after mi.set_variant() in main().


def load_renderer_config(project_root: str, renderer_key: str) -> DictConfig:
    """Load ``config/renderer/<renderer_key>.yaml``."""
    yaml_path = os.path.join(project_root, "config", "renderer", f"{renderer_key}.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Renderer yaml not found: {yaml_path}")
    return OmegaConf.load(yaml_path)


def build_overrides(cfg: DictConfig, project_root: str) -> dict:
    """Resolve ``cfg.objects`` (shape_id -> renderer key) into a
    ``{shape_id: bsdf_dict}`` map suitable for ``apply_bsdf_overrides``."""
    overrides: dict = {}
    if "objects" not in cfg or cfg.objects is None:
        return overrides
    grazing_deg = float(getattr(cfg, "grazing_mask_deg", 0) or 0)
    # Per-shape UV tiling override map (shape_id -> uv multiplier, default 3.0).
    # Takes precedence over a renderer yaml's `tiling`. Set via Hydra, e.g.
    #   +tiling_map.elm__54=12
    tiling_map = cfg.get("tiling_map", {}) or {}
    for shape_id, renderer_key in cfg.objects.items():
        rcfg = load_renderer_config(project_root, str(renderer_key))
        bsdf = {
            "type": "mlpbrdf",
            "model_path": str(rcfg.ckpt_path),
            "material_type": str(rcfg.material_type),
        }
        if grazing_deg > 0:
            bsdf["grazing_mask_deg"] = grazing_deg
        # Per-shape UV tiling: config tiling_map overrides renderer-yaml tiling.
        tval = None
        if shape_id in tiling_map and tiling_map[shape_id] is not None:
            tval = float(tiling_map[shape_id])
        elif "tiling" in rcfg and rcfg.tiling is not None:
            tval = float(rcfg.tiling)
        if tval is not None:
            bsdf["tiling"] = tval
            print(f"[tiling] {shape_id} -> uv*{tval}")
        overrides[shape_id] = bsdf
    return overrides


def render_with_batches(scene, total_spp: int, batch_spp: int):
    """Render ``total_spp`` in chunks, clearing CUDA/DrJit caches between batches."""
    from custom_bsdf.utils.cuda_manage import print_cuda_memory_info

    accumulated = None
    n_batches = (total_spp + batch_spp - 1) // batch_spp
    t0 = time.time()
    for b in range(n_batches):
        bs = min(batch_spp, total_spp - b * batch_spp)
        seed = random.randint(0, 2**31 - 1)
        print_cuda_memory_info(f"before batch {b + 1}/{n_batches}")
        # Forward render only: suspend AD so mi.render doesn't build/retain an
        # autodiff graph attached to scene params (second unbounded-memory source).
        with dr.suspend_grad():
            img = mi.render(scene, spp=bs, seed=seed)
            accumulated = img * bs if accumulated is None else accumulated + img * bs
        # Force evaluation so the lazy DrJit graph collapses to a concrete buffer
        # each batch. Without this the accumulator chains 1 add-node per batch and
        # GPU memory grows unboundedly over many batches (OOM at high spp).
        dr.eval(accumulated)
        dr.sync_thread()
        del img
        torch.cuda.empty_cache()
        dr.flush_malloc_cache()
        gc.collect()
        print_cuda_memory_info(f"after batch  {b + 1}/{n_batches}")
    print(f"render time: {time.time() - t0:.1f}s")
    return accumulated / total_spp


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    mi.set_variant(str(cfg.variant))

    # Late imports: BSDF plugins must be registered after the variant is set.
    from custom_bsdf.cook_torrancebrdf import CookTorranceBRDF
    from custom_bsdf.mlp import MLPBRDF
    from custom_bsdf.utils.cuda_manage import print_cuda_memory_info
    from custom_bsdf.utils.scene_loader import load_scene_with_bsdf_overrides

    mi.register_bsdf("cooktorrancebrdf", lambda props: CookTorranceBRDF(props))
    mi.register_bsdf("mlpbrdf", lambda props: MLPBRDF(props))

    project_root = hydra.utils.get_original_cwd()

    # ------------------------------------------------------------------ Scene
    scene_xml = cfg.scene.xml_path
    if not os.path.isabs(scene_xml):
        scene_xml = os.path.join(project_root, scene_xml)
    if not os.path.isfile(scene_xml):
        raise FileNotFoundError(f"Scene xml not found: {scene_xml}")

    print("=" * 64)
    print(f"Scene:     {scene_xml}")
    print(f"Variant:   {cfg.variant}")
    print(f"Render:    spp={cfg.render.spp}  batch_spp={cfg.render.batch_spp}")
    print(f"Objects -> renderer:")
    for sid, rkey in (cfg.objects or {}).items():
        rcfg = load_renderer_config(project_root, str(rkey))
        print(f"  {sid:>10s}  {rkey:>10s}  ({rcfg.material_type}, {rcfg.ckpt_path})")
    print("=" * 64)

    # Optional fixed seed so unchanged regions are bit-identical across runs
    # (lets us diff per-object pixels to detect cross-object contamination).
    if getattr(cfg, "seed", None) is not None:
        random.seed(int(cfg.seed))

    fov_override = getattr(cfg, "fov", None)
    resx_override = getattr(cfg, "resx", None)
    resy_override = getattr(cfg, "resy", None)
    # Optional per-shape translations from a JSON file ({shape_id: [dx,dy,dz]}).
    translations = None
    tfile = getattr(cfg, "translations_file", None)
    if tfile:
        import json
        with open(tfile) as f:
            translations = json.load(f)
        print(f"loaded {len(translations)} translations from {tfile}")
    # Optional emitter radiance overrides ({shape_id: intensity}); also reads an
    # optional "_radiance" block from the translations JSON for convenience.
    radiance = dict(getattr(cfg, "radiance", {}) or {})
    if translations and "_radiance" in translations:
        radiance.update(translations.pop("_radiance"))
    radiance = {k: float(v) for k, v in radiance.items()} or None
    overrides = build_overrides(cfg, project_root)
    if overrides:
        # Bake per-shape BSDFs into the XML before Mitsuba loads it, so the
        # chosen mlpbrdf is instantiated exactly once with the right ckpt.
        scene = load_scene_with_bsdf_overrides(
            scene_xml, overrides, fov=fov_override, resx=resx_override, resy=resy_override,
            translations=translations, radiance=radiance)
    else:
        scene = mi.load_file(scene_xml)
    print("loaded scene")

    # ----------------------------------------------------------------- Render
    print_cuda_memory_info("before render")
    image = render_with_batches(scene, int(cfg.render.spp), int(cfg.render.batch_spp))
    print_cuda_memory_info("after render")

    # ----------------------------------------------------------------- Output
    out_dir = cfg.output_base
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cfg.output_name}.png")
    out_path_hdr = os.path.join(out_dir, f"{cfg.output_name}_pre_tonemap.png")

    arr = np.array(image)
    # NO tonemapping by default: write linear radiance (sRGB + clamp to [0,1]).
    # Over-bright pixels clip to white, which honestly shows brightness bugs.
    mi.util.write_bitmap(out_path, mi.Bitmap(arr))
    # Always dump EXR for quantitative (pre-clip) analysis.
    mi.util.write_bitmap(os.path.join(out_dir, f"{cfg.output_name}.exr"), mi.Bitmap(arr))
    # Tonemap only if explicitly requested (+tonemap=true).
    if bool(getattr(cfg, "tonemap", False)):
        arr_tm = arr / (1.0 + arr)
        mi.util.write_bitmap(os.path.join(out_dir, f"{cfg.output_name}_reinhard.png"), mi.Bitmap(arr_tm))

    print(f"\nSaved {out_path}")
    print(f"  raw  min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")


if __name__ == "__main__":
    main()
