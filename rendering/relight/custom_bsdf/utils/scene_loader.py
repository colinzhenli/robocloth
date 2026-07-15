import mitsuba as mi
import os
import numpy as np

mi.set_variant("llvm_ad_rgb")  # or "scalar_rgb" / "llvm_ad_rgb"
from omegaconf import OmegaConf


def _mitsuba_mesh_type_from_path(mesh_path: str) -> str:
    """Return Mitsuba mesh plugin name: ``obj`` or ``ply``."""
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext == ".obj":
        return "obj"
    if ext == ".ply":
        return "ply"
    raise ValueError(
        f"Unsupported mesh extension {ext!r}: use .obj or .ply "
        "(or convert with Blender / meshio / Mitsuba importers)."
    )


def load_uv_obj_to_mitsuba_scene(args,cfg,obj_path="data/cloth.obj"):
    """
    Read a mesh file and build a Mitsuba shape dict with BSDF attached.

    Parameters:
        obj_path (str): Path to `.obj` (with UV) or `.ply` mesh.
                        For UV-dependent rendering, OBJ usually carries ``vt`` rows;
                        PLY requires texture coordinates embedded in the PLY attributes
                        (Mitsuba exposes them only if present in the file).
        texture_path (str or None): Optional, texture image path (like .jpg/.png).
        scale (float): Scale ratio for geometry.
        rotate_x90 (bool): Whether to rotate geometry -90° around X-axis, converting from Blender export coordinates to Mitsuba's default orientation.

    Returns:
        scene (mi.Scene): Mitsuba scene object.
    """
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"Mesh file does not exist: {obj_path}")

    mesh_type = _mitsuba_mesh_type_from_path(obj_path)

    swap_yz = mi.ScalarTransform4f.rotate([0, 1, 0], -90)

    # Then scale x and y axes to 0.2 (note scaling order and application order)
    scale_xy = mi.ScalarTransform4f.scale([0.03, 0.03, -0.03])

    # First rotate then scale (matrix multiplication executes right multiplication first)
    transform = scale_xy @ swap_yz
    # transform = scale_xy
    
    # Build shape loading dictionary
    '''
    shape_dict = {
        "type": mesh_type,
        "filename": obj_path,
        "to_world": transform
    }
    '''
    scene_cfg = getattr(cfg, "scene", None)
    # Keep the original-case string for use as a filesystem path; only the
    # lowercased copy is used for the "sphere"/"cloth" keyword comparison.
    # (Lowercasing the path itself corrupts absolute paths that contain
    # uppercase dirs, e.g. ".../Misuba_rendering/...".)
    mesh_raw = str(getattr(scene_cfg, "mesh", "sphere")) if scene_cfg else "sphere"
    mesh_kind = mesh_raw.lower()

    if mesh_kind == "sphere":
        # Mitsuba primitive sphere is a unit sphere at origin; we keep its
        # CENTER at the world origin so it stays centered in frame when the
        # camera looks at (0, 0, 0), regardless of scale.
        sphere_scale = float(getattr(scene_cfg, "sphere_scale", 1.0)) if scene_cfg else 1.0
        sphere_to_world = (
            mi.ScalarTransform4f.scale(sphere_scale)
            @ mi.ScalarTransform4f.rotate([1, 0, 0], 90)
        )
        shape_dict = {
            "type": "sphere",
            "to_world": sphere_to_world,
        }
    else:
        # `mesh_kind` is either "cloth" (the bundled data/cloth.obj draped
        # cloth from cloth_simulation.blend) or an absolute path to any
        # OBJ/PLY. Auto-fit: center the bbox at origin and uniformly scale
        # so the longest axis spans `mesh_scale * 2` world units (default
        # 0.4 → max half-extent 0.2, matches the sphere_scale=0.2 default
        # we settled on for the envmap framing).
        if mesh_kind == "cloth":
            mesh_path = "data/cloth.obj"
        else:
            mesh_path = mesh_raw
        if not os.path.isabs(mesh_path):
            from hydra.utils import get_original_cwd
            mesh_path = os.path.join(get_original_cwd(), mesh_path)
        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(f"mesh file not found: {mesh_path}")

        ext = os.path.splitext(mesh_path)[1].lower()
        mi_mesh_type = "obj" if ext == ".obj" else "ply"

        # Probe the bbox so we can normalize the placement before attaching
        # the (expensive) neural BSDF. Loading without a BSDF is cheap.
        probe = mi.load_dict({"type": "scene", "shape": {"type": mi_mesh_type, "filename": mesh_path}})
        bbox = probe.bbox()
        center = (bbox.min + bbox.max) * 0.5
        extents = bbox.extents()
        # numpy-style max over a Vector3f via Python max
        longest = max(float(extents.x), float(extents.y), float(extents.z))
        target_half = float(getattr(scene_cfg, "mesh_scale", 0.2)) if scene_cfg else 0.2
        s = (target_half * 2.0) / longest if longest > 0 else 1.0

        cloth_to_world = (
            mi.ScalarTransform4f.scale(s)
            @ mi.ScalarTransform4f.translate([-float(center.x), -float(center.y), -float(center.z)])
        )
        shape_dict = {
            "type": mi_mesh_type,
            "filename": mesh_path,
            "to_world": cloth_to_world,
        }
    
    shape_dict["bsdf"] = {
        'type': 'mlpbrdf',
        'model_path': args.model_path,  # Pass model path
        'material_type': cfg.material.type,
        'use_btf': bool(cfg.use_btf),
        'btf_path': str(cfg.btf_path),
        # When false, the model's eval skips the final brdf*NoL multiplication.
        # Used for checkpoints whose training already baked cos into the network
        # output (e.g. apply_cosine_weight=False trainers).  Default true.
        'apply_cosine_at_eval': bool(getattr(cfg, 'apply_cosine_at_eval', True)),
        # UV tiling factor: texture repeats this many times across the mesh UV
        # (eval does uv = (uv * uv_tiling) % 1). Default 5.0; override per-render
        # via +uv_tiling=N (e.g. 25 = 5x finer tiling than the 5.0 baseline).
        'uv_tiling': float(getattr(cfg, 'uv_tiling', 5.0)),
        #'eta': 1.33,
        #'emitter1': {
        #    'type': 'point',
        #    'position': [1.0, 2.0, 3.0],
        #    'intensity': {'type': 'spectrum', 'value': 20.0}
        #},
        #'emitter2': {
        #    'type': 'point',
        #    'position': [-2.0, 1.0, 0.0],
        #    'intensity': {'type': 'spectrum', 'value': 15.0}
        #}
        
    }
    
    '''
    shape_dict["bsdf"] = {
        'type': 'roughconductor'
    }
    '''
    #bsdf=mi.load_dict(shape_dict["bsdf"])

    # NOTE: the mlpbrdf is two-sided internally (see MLPBRDF.eval), so an open
    # drape's interior renders fabric instead of black WITHOUT Mitsuba's stock
    # `twosided` wrapper — that wrapper silently dims this custom BSDF's front.
    return shape_dict

    # Construct scene
    scene_dict = {
        "type": "scene",
        "shape": shape_dict,
        "flip_normals": True
    }

    scene = mi.load_dict(scene_dict)
    
    return scene
