"""
Batch-render every ``*.ckpt`` in a folder with the same scene setup as ``rendering.py``.

Usage (from ``SGHyperMaterials_axf`` root, after Hydra args):

    python batch_rendering.py --ckpt_dir /path/to/ckpts --output_dir output/mybatch

Rules:
    - One PNG per checkpoint; basename matches the ckpt stem (e.g. ``felt05_PBR.ckpt`` → ``felt05_PBR.png``).
    - If the ckpt **filename** contains the substring ``PBR`` (case-insensitive), use ``UBOPBRLatentBRDF``;
      otherwise ``UBOLatentBRDF``.

Other Hydra/material options follow ``config/config.yaml`` defaults; only ``cfg.material.type`` is
switched per file so ``create_anisotropic_model`` loads the matching YAML under ``config/material/``.
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import random
import sys
import time

# Strip our args before Hydra parses the remainder
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--ckpt_dir",
    type=str,
    required=True,
    help="Directory containing .ckpt checkpoints",
)
_parser.add_argument(
    "--output_dir",
    type=str,
    default="output/batch_ckpt_render",
    help="Directory to write PNGs (relative paths are resolved from project root)",
)
_parser.add_argument("--batch_spp", type=int, default=2)
_parser.add_argument("--render_spp", type=int, default=32)
_parser.add_argument("--mode", type=str, default="cuda_ad_rgb")
_pre_args, _hydra_argv = _parser.parse_known_args()
sys.argv = [sys.argv[0]] + _hydra_argv

import drjit as dr
import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import open_dict

dr.set_flag(dr.JitFlag.LoopRecord, False)
dr.set_flag(dr.JitFlag.VCallRecord, False)

import mitsuba as mi

mi.set_variant(_pre_args.mode)

from custom_bsdf.cook_torrancebrdf import CookTorranceBRDF
from custom_bsdf.mlp import MLPBRDF
from custom_bsdf.mybsdf import MyBSDF
from custom_bsdf.utils.cuda_manage import print_cuda_memory_info
from custom_bsdf.utils.scene_loader import load_uv_obj_to_mitsuba_scene


class RenderArgs:
    """Minimal namespace compatible with ``load_uv_obj_to_mitsuba_scene`` / ``rendering.py``."""

    pass


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    dr.set_flag(dr.JitFlag.LoopRecord, False)
    dr.set_flag(dr.JitFlag.VCallRecord, False)

    orig = get_original_cwd()
    ckpt_dir = (
        _pre_args.ckpt_dir
        if os.path.isabs(_pre_args.ckpt_dir)
        else os.path.join(orig, _pre_args.ckpt_dir)
    )
    out_dir = (
        _pre_args.output_dir
        if os.path.isabs(_pre_args.output_dir)
        else os.path.join(orig, _pre_args.output_dir)
    )

    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"ckpt_dir is not a directory: {ckpt_dir}")

    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found under: {ckpt_dir}")

    os.makedirs(out_dir, exist_ok=True)

    rargs = RenderArgs()
    rargs.forward_only = False
    rargs.batch_spp = _pre_args.batch_spp
    render_spp = _pre_args.render_spp

    mi.register_bsdf("cooktorrancebrdf", lambda props: CookTorranceBRDF(props))
    mi.register_bsdf("mlpbrdf", lambda props: MLPBRDF(props))
    mi.register_bsdf("mybsdf", lambda props: MyBSDF(props))

    def render_with_cache_clearing(scene, total_spp: int, batch_spp: int):
        accumulated_image = None
        total_batches = (total_spp + batch_spp - 1) // batch_spp
        t0 = time.time()
        for batch_idx in range(total_batches):
            current_batch_spp = min(batch_spp, total_spp - batch_idx * batch_spp)
            current_seed = random.randint(0, 2**31 - 1)
            batch_image = mi.render(scene, spp=current_batch_spp, seed=current_seed)
            if accumulated_image is None:
                accumulated_image = batch_image * current_batch_spp
            else:
                accumulated_image += batch_image * current_batch_spp
            torch.cuda.empty_cache()
            dr.flush_malloc_cache()
            gc.collect()
        print(f"Rendered {total_spp} spp in {time.time() - t0:.2f}s")
        return accumulated_image / total_spp

    batch_size = min(rargs.batch_spp, render_spp)

    for ckpt_path in ckpts:
        base = os.path.basename(ckpt_path)
        stem, _ext = os.path.splitext(base)
        material_type = "UBOPBRLatentBRDF" if "PBR" in base.upper() else "UBOLatentBRDF"

        print("=" * 60)
        print(f"Checkpoint: {ckpt_path}")
        print(f"Material type: {material_type}")
        print(f"Output: {os.path.join(out_dir, stem + '.png')}")

        rargs.model_path = ckpt_path
        with open_dict(cfg):
            cfg.material.type = material_type

        '''
        {
            'type': 'scene',
            'integrator': {'type': 'path', 'max_depth': 4},
            'sensor': {
                'type': 'perspective',
                'near_clip': 0.1,
                'far_clip': 10.0,
                'to_world': mi.ScalarTransform4f().look_at(origin=[0, 1, -2], target=[0, -1, 1], up=[0, 1, 0]),
                'fov': 30.0,
                'sampler': {
                    'type': 'independent',
                    'sample_count': 4
                },
                'film': {
                    'type': 'hdrfilm',
                    'width': 1600,
                    'height': 1600,
                    'rfilter': {'type': 'gaussian'},
                    'pixel_format': 'rgb',
                    'banner': False
                }
            },
            'emitter': {
                'type': 'envmap',
                'filename': '/home/haoran/SGHyperMaterials_axf/data/envmap.exr'
            },
            #'emitter1': {
            #    'type': 'point',
            #    'position': [-1.0, -1.0, -2.0],
            #    'intensity': {'type': 'spectrum', 'value': 25.0}  # Changed from 2.50
            #},
            'shape': load_uv_obj_to_mitsuba_scene(rargs,cfg)
        }
        '''

        opt_scene = mi.load_dict(
            
            {
                "type": "scene",
                "integrator": {"type": "path"},
                "sensor": {
                    "type": "perspective",
                    "to_world": mi.ScalarTransform4f.look_at(
                        origin=[-3.5, 2, 3.5],
                        target=[0, 0, 0],
                        up=[0, 1, 0],
                    ),
                    "film": {
                        "type": "hdrfilm",
                        "width": 1024,
                        "height": 1024,
                    },
                },
                "constant_emitter": {
                    "type": "constant",
                    "radiance": {"type": "rgb", "value": 0.5},
                },
                "point_emitter": {
                    "type": "point",
                    "position": [-2, 2, 0],
                    "intensity": {"type": "rgb", "value": [10, 10, 10]},
                },
                'shape': load_uv_obj_to_mitsuba_scene(rargs,cfg),
                "floor": {
                    "type": "rectangle",
                    "to_world": (mi.ScalarTransform4f.translate([0, -1, 0]) @ mi.ScalarTransform4f.rotate([1, 0, 0], -90) @ mi.ScalarTransform4f.scale(2)),
                    "bsdf": {
                        "type": "diffuse",
                        "reflectance": {
                            "type": "checkerboard",
                            "to_uv": mi.ScalarTransform4f.scale(5),
                        },
                    },
                },
            }
        )

        print_cuda_memory_info("Before rendering")
        with torch.no_grad():
            initial_image = render_with_cache_clearing(opt_scene, render_spp, batch_size)
        bmp = mi.Bitmap(initial_image)
        out_path = os.path.join(out_dir, stem + ".png")
        mi.util.write_bitmap(out_path, bmp)
        print_cuda_memory_info("After rendering")
        print(f"Saved: {out_path}\n")

        del opt_scene, initial_image, bmp
        # Let the previous scene / PyTorch model drop before the next mi.load_dict (improvement 2)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        dr.flush_malloc_cache()
        gc.collect()


if __name__ == "__main__":
    main()
