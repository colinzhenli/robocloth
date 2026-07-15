import mitsuba as mi
import os
import xml.etree.ElementTree as ET
import numpy as np

mi.set_variant("cuda_ad_rgb")  # or "scalar_rgb" / "llvm_ad_rgb"
from omegaconf import OmegaConf


def load_scene_with_bsdf_overrides(xml_path: str, overrides: dict, fov=None, resx=None, resy=None, translations=None, radiance=None, cameras=None):
    """Load a Mitsuba XML scene, substituting per-shape mlpbrdf BSDFs first.

    Editing the XML before load avoids two problems:
      1. Double-instantiating BSDFs (once for the XML, once for the rebuilt
         scene dict).
      2. Shape-id lookups after load — in some Mitsuba 3 builds, shapes that
         share the exact same BSDF can be merged/renamed, so their original
         ids are no longer reachable via ``scene.shapes()``.

    Args:
        xml_path: Path to the scene XML.
        overrides: ``{shape_id: bsdf_dict}``. ``bsdf_dict`` must have at least
            ``{"type": "mlpbrdf", "model_path": ..., "material_type": ...}``.
            Shape ids must match the XML's ``<shape id="...">``.

    Returns:
        ``mi.Scene`` loaded with the overrides applied.
    """
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"Scene xml not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    declared_ids = {s.get("id") for s in root.findall(".//shape") if s.get("id")}
    missing = sorted(set(overrides.keys()) - declared_ids)
    if missing:
        raise KeyError(
            f"Shape id(s) not declared in {xml_path}: {missing}\n"
            f"Available shape ids: {sorted(declared_ids)}"
        )

    for shape_id, bsdf in overrides.items():
        shape_el = root.find(f".//shape[@id='{shape_id}']")
        # Drop any existing inline BSDF or BSDF ref so we can replace it.
        for child in list(shape_el):
            if child.tag in ("bsdf", "ref") and child.get("name", "bsdf") == "bsdf":
                shape_el.remove(child)
        bsdf_el = ET.SubElement(shape_el, "bsdf", {"type": bsdf.get("type", "mlpbrdf")})
        for k, v in bsdf.items():
            if k == "type":
                continue
            ET.SubElement(bsdf_el, "string", {"name": k, "value": str(v)})

    # Optional per-shape layout edit. Value is either:
    #   [dx,dy,dz]                       -> pure translation, or
    #   {"d":[dx,dy,dz],"ry":deg,"c":[cx,cy,cz]} -> rotate ry deg about the
    #   vertical (y) axis through center c, then translate by d (rigid in-place
    #   rotation + shift). Mesh is baked in world space at center c.
    for shape_id, val in (translations or {}).items():
        shape_el = root.find(f".//shape[@id='{shape_id}']")
        if shape_el is None:
            print(f"[scene_loader] WARNING: layout target '{shape_id}' not found")
            continue
        # Read + remove any existing to_world so we can COMPOSE with it (e.g. the
        # area light elm__4 has its own matrix; we must not clobber it).
        M_old = np.eye(4)
        for ch in list(shape_el):
            if ch.tag == "transform" and ch.get("name") == "to_world":
                m = ch.find("matrix")
                if m is not None:
                    M_old = np.array([float(x) for x in m.get("value").split()],
                                     dtype=float).reshape(4, 4)
                shape_el.remove(ch)
        # Build my world-space transform M_mine.
        if isinstance(val, dict):
            d = np.array([float(x) for x in val.get("d", [0, 0, 0])])
            c = np.array([float(x) for x in val.get("c", [0, 0, 0])])
            # Full Euler rotation about center c. rx/rz give *tilt* (lean); ry is
            # yaw (backward-compatible: omitted angles default to 0). Order Rz@Ry@Rx.
            rx = np.radians(float(val.get("rx", 0.0)))
            ry = np.radians(float(val.get("ry", 0.0)))
            rz = np.radians(float(val.get("rz", 0.0)))
            cx, sx = np.cos(rx), np.sin(rx)
            cy, sy = np.cos(ry), np.sin(ry)
            cz, sz = np.cos(rz), np.sin(rz)
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx
            M_mine = np.eye(4); M_mine[:3, :3] = R; M_mine[:3, 3] = (np.eye(3) - R) @ c + d
            print(f"[scene_loader] rotate {shape_id} rx={np.degrees(rx):.1f} ry={np.degrees(ry):.1f} rz={np.degrees(rz):.1f} + translate {d.tolist()}")
        else:
            M_mine = np.eye(4); M_mine[:3, 3] = [float(val[0]), float(val[1]), float(val[2])]
            print(f"[scene_loader] translate {shape_id} by ({val[0]}, {val[1]}, {val[2]})")
        M = M_mine @ M_old   # apply my world transform on top of the existing one
        tf = ET.SubElement(shape_el, "transform", {"name": "to_world"})
        ET.SubElement(tf, "matrix", {"value": " ".join(f"{v:.6f}" for v in M.flatten())})

    # Optional emitter radiance override (shape_id -> scalar intensity, r=g=b).
    for shape_id, r in (radiance or {}).items():
        shape_el = root.find(f".//shape[@id='{shape_id}']")
        if shape_el is None:
            print(f"[scene_loader] WARNING: radiance target '{shape_id}' not found")
            continue
        rgb = shape_el.find(".//emitter/rgb[@name='radiance']")
        if rgb is not None:
            rgb.set("value", f"{float(r)} {float(r)} {float(r)}")
            print(f"[scene_loader] set {shape_id} radiance -> {float(r)}")
        else:
            print(f"[scene_loader] WARNING: {shape_id} has no area-emitter radiance")

    # Optional multi-camera override: replace the scene's sensor(s) with N
    # perspective look-at sensors (for close-up multi-view inspection). Each
    # spec: {"origin":[x,y,z], "target":[x,y,z], "up":[x,y,z]=+Y, "fov":deg}.
    # Render each by index: mi.render(scene, sensor=i).
    if cameras:
        for s in root.findall("sensor"):
            root.remove(s)
        for i, cam in enumerate(cameras):
            o = cam["origin"]; t = cam["target"]; up = cam.get("up", [0, 1, 0])
            sen = ET.SubElement(root, "sensor", {"type": "perspective", "id": f"cam_{i}"})
            ET.SubElement(sen, "string", {"name": "fov_axis", "value": "x"})
            ET.SubElement(sen, "float", {"name": "fov", "value": str(float(cam.get("fov", 45.0)))})
            ET.SubElement(sen, "float", {"name": "near_clip", "value": "0.01"})
            ET.SubElement(sen, "float", {"name": "far_clip", "value": "1000.0"})
            tf = ET.SubElement(sen, "transform", {"name": "to_world"})
            ET.SubElement(tf, "lookat", {
                "origin": ",".join(f"{v:.5f}" for v in o),
                "target": ",".join(f"{v:.5f}" for v in t),
                "up": ",".join(f"{v:.5f}" for v in up)})
            sm = ET.SubElement(sen, "sampler", {"type": "independent", "name": "sampler"})
            ET.SubElement(sm, "integer", {"name": "sample_count", "value": "$spp"})
            fl = ET.SubElement(sen, "film", {"type": "hdrfilm", "name": "film"})
            ET.SubElement(fl, "integer", {"name": "width", "value": "$resx"})
            ET.SubElement(fl, "integer", {"name": "height", "value": "$resy"})
        print(f"[scene_loader] replaced sensors with {len(cameras)} close-up cameras")
    # Optional camera FOV override (degrees).
    elif fov is not None:
        fov_el = root.find(".//sensor//float[@name='fov']")
        if fov_el is not None:
            fov_el.set("value", str(float(fov)))
            print(f"[scene_loader] FOV overridden to {float(fov)}")
        else:
            print("[scene_loader] WARNING: no <float name='fov'> found to override")

    # Write to a sibling file so relative paths in the XML still resolve.
    out_path = os.path.join(os.path.dirname(xml_path), ".rendering.generated.xml")
    tree.write(out_path)
    # Resolution overrides the XML's $resx/$resy <default> params.
    load_params = {}
    if resx is not None:
        load_params["resx"] = str(int(resx))
    if resy is not None:
        load_params["resy"] = str(int(resy))
    if load_params:
        print(f"[scene_loader] resolution override {load_params}")
    try:
        return mi.load_file(out_path, **load_params)
    finally:
        # Don't leave stray artifacts next to the canonical scene XML.
        try:
            os.remove(out_path)
        except OSError:
            pass


def scene_to_reload_dict(scene: mi.Scene) -> dict:
    """
    Flatten a Mitsuba Scene into the dict accepted by ``mi.load_dict``.

    Keeps meshes that carry embedded emitters intact (only enumerate ``shapes()``,
    sensors, integrator — no separate emitter list needed for typical meshes).
    """
    if not isinstance(scene, mi.Scene):
        raise TypeError(f"Expected mi.Scene, got {type(scene)}")

    unknown_counter = 0

    def take_id(child) -> str:
        nonlocal unknown_counter
        oid = child.id()
        if oid == "":
            oid = f"_unnamed_emit_{unknown_counter}"
            unknown_counter += 1
        return oid

    children = [*scene.shapes(), *scene.sensors(), scene.integrator()]
    return {"type": "scene", **{take_id(child): child for child in children}}


def merge_shape_into_scene_dict(
    scene: mi.Scene, shape_dict, shape_db_key="custom_nn_shape"
) -> dict:
    """
    Merge XML (or dict) loaded scene with one or more mesh descriptions from
    ``load_uv_obj_to_mitsuba_scene``. Pass result to ``mi.load_dict``.

    ``shape_dict`` may be a single shape dict or a list/tuple of shape dicts.
    ``shape_db_key``: if ``shape_dict`` is a list, either a string prefix (ids become
    ``prefix``, ``prefix_1``, …) or a list of ids with the same length as ``shape_dict``.
    """
    d = scene_to_reload_dict(scene)

    def add_shape(key: str, sd: dict):
        d[key] = {**sd, "id": key}

    if isinstance(shape_dict, dict):
        if isinstance(shape_db_key, (list, tuple)):
            if len(shape_db_key) != 1:
                raise ValueError(
                    "For a single shape dict, shape_db_key must be a str or a one-element sequence"
                )
            key = shape_db_key[0]
        else:
            key = shape_db_key
        add_shape(key, shape_dict)
        return d

    shapes = list(shape_dict)
    if isinstance(shape_db_key, str):
        keys = [shape_db_key] + [f"{shape_db_key}_{i}" for i in range(1, len(shapes))]
    else:
        keys = list(shape_db_key)
        if len(keys) != len(shapes):
            raise ValueError(
                f"shape_db_key sequence length ({len(keys)}) must match "
                f"number of shapes ({len(shapes)})"
            )
    for key, sd in zip(keys, shapes):
        add_shape(key, sd)
    return d

def apply_bsdf_overrides(scene, overrides: dict) -> dict:
    """Rebuild scene dict, replacing the BSDF of each shape in ``overrides``.

    Args:
        scene: Loaded ``mi.Scene``.
        overrides: ``{shape_id: bsdf_dict}`` — one entry per shape to override.
            ``bsdf_dict`` is a full Mitsuba bsdf spec, e.g.
            ``{"type": "mlpbrdf", "model_path": "...", "material_type": "..."}``.

    Returns:
        Scene dict suitable for ``mi.load_dict``. Shapes not in ``overrides``
        keep their existing BSDF.
    """
    if not isinstance(scene, mi.Scene):
        raise TypeError(f"Expected mi.Scene, got {type(scene)}")
    if not overrides:
        return scene_to_reload_dict(scene)

    d = scene_to_reload_dict(scene)
    params = mi.traverse(scene)
    shape_by_id = {s.id(): s for s in scene.shapes()}

    missing = sorted(set(overrides.keys()) - set(shape_by_id.keys()))
    if missing:
        raise KeyError(f"Shape id(s) not found in scene: {missing}")

    for sid, bsdf_dict in overrides.items():
        shape = shape_by_id[sid]
        filename_key = f"{sid}.filename"
        to_world_key = f"{sid}.to_world"

        filename = str(params[filename_key]) if filename_key in params else None
        to_world = params[to_world_key] if to_world_key in params else None

        if filename is None:
            shape_params = mi.traverse(shape)
            if "filename" in shape_params:
                filename = str(shape_params["filename"])
            if to_world is None and "to_world" in shape_params:
                to_world = shape_params["to_world"]

        if filename is not None:
            shape_dict = {
                "type": _mitsuba_mesh_type_from_path(filename),
                "filename": filename,
                "id": sid,
                "bsdf": bsdf_dict,
            }
        else:
            faces_key = f"{sid}.faces"
            vpos_key = f"{sid}.vertex_positions"
            if faces_key not in params or vpos_key not in params:
                raise NotImplementedError(
                    f"Shape '{sid}' does not expose filename or mesh buffers "
                    f"('{faces_key}', '{vpos_key}'). Cannot replace BSDF safely."
                )
            shape_dict = {
                "type": "mesh",
                "id": sid,
                "faces": params[faces_key],
                "vertex_positions": params[vpos_key],
                "bsdf": bsdf_dict,
            }
            vnorm_key = f"{sid}.vertex_normals"
            vtex_key = f"{sid}.vertex_texcoords"
            if vnorm_key in params:
                shape_dict["vertex_normals"] = params[vnorm_key]
            if vtex_key in params:
                shape_dict["vertex_texcoords"] = params[vtex_key]

        if to_world is not None:
            shape_dict["to_world"] = to_world

        d[sid] = shape_dict

    return d


def change_shape_bsdf(scene, shape_ids, model_path="", material_type=""):
    """
    Replace BSDF of selected shapes in an existing scene with neural BSDF.

    Args:
        scene: Loaded ``mi.Scene``.
        shape_ids: One shape id (str) or multiple ids (list/tuple/set).
        model_path: Checkpoint path passed to ``mlpbrdf``.
        material_type: Material type passed to ``mlpbrdf``.

    Returns:
        dict: Scene dict that can be reloaded by ``mi.load_dict``.

    Notes:
        - Only selected file-mesh shapes (``.obj``/``.ply``) are reconstructed and
          BSDF-replaced; all other scene children are kept untouched.
        - This function raises if a requested id does not exist in scene shapes.
    """
    if not isinstance(scene, mi.Scene):
        raise TypeError(f"Expected mi.Scene, got {type(scene)}")

    if isinstance(shape_ids, str):
        target_ids = {shape_ids}
    else:
        target_ids = set(shape_ids)
    if not target_ids:
        raise ValueError("shape_ids is empty")

    d = scene_to_reload_dict(scene)
    params = mi.traverse(scene)
    shape_by_id = {s.id(): s for s in scene.shapes()}

    missing = sorted(target_ids - set(shape_by_id.keys()))
    if missing:
        raise KeyError(f"Shape id(s) not found in scene: {missing}")

    neural_bsdf = {
        "type": "mlpbrdf",
        "model_path": model_path,
        "material_type": material_type,
    }

    for sid in target_ids:
        shape = shape_by_id[sid]
        filename_key = f"{sid}.filename"
        to_world_key = f"{sid}.to_world"

        filename = None
        to_world = None

        # Prefer scene-level traverse keys (fast path)
        if filename_key in params:
            filename = str(params[filename_key])
        if to_world_key in params:
            to_world = params[to_world_key]

        # Fallback to shape-local traverse for scenes where global keys are not exposed
        if filename is None:
            shape_params = mi.traverse(shape)
            if "filename" in shape_params:
                filename = str(shape_params["filename"])
            if to_world is None and "to_world" in shape_params:
                to_world = shape_params["to_world"]

        if filename is not None:
            mesh_type = _mitsuba_mesh_type_from_path(filename)
            shape_dict = {
                "type": mesh_type,
                "filename": filename,
                "id": sid,
                "bsdf": neural_bsdf,
            }
        else:
            # Fallback for scenes where file path is not exposed by traverse:
            # rebuild shape from in-memory mesh buffers.
            faces_key = f"{sid}.faces"
            vpos_key = f"{sid}.vertex_positions"
            if faces_key not in params or vpos_key not in params:
                raise NotImplementedError(
                    f"Shape '{sid}' does not expose filename or mesh buffers "
                    f"('{faces_key}', '{vpos_key}'). Cannot replace BSDF safely."
                )

            shape_dict = {
                "type": "mesh",
                "id": sid,
                "faces": params[faces_key],
                "vertex_positions": params[vpos_key],
                "bsdf": neural_bsdf,
            }
            vnorm_key = f"{sid}.vertex_normals"
            vtex_key = f"{sid}.vertex_texcoords"
            if vnorm_key in params:
                shape_dict["vertex_normals"] = params[vnorm_key]
            if vtex_key in params:
                shape_dict["vertex_texcoords"] = params[vtex_key]

        if to_world is not None:
            shape_dict["to_world"] = to_world

        d[sid] = shape_dict

    return d


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


def load_uv_obj_to_mitsuba_scene(
    args,
    cfg,
    obj_path="/home/haoran/SGHyperMaterials/data/material_out/mesh_00621.ply",
    world_translation=(0.0,0.0,0.0),  # e.g. ``(2.5, 0.48, -1.0)`` in ``room``
    model_path="",
    material_type=""
):
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
        world_translation (tuple[float, float, float]): World-space translation applied after rotate and scale ([x,y,z] in scene units).

    Returns:
        dict: Mitsuba shape specification for ``mi.load_dict`` (not ``mi.Scene``).
    """
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"Mesh file does not exist: {obj_path}")

    mesh_type = _mitsuba_mesh_type_from_path(obj_path)

    swap_yz = mi.ScalarTransform4f.rotate([0, 1, 0], -90)

    # Then scale x and y axes to 0.2 (note scaling order and application order)
    scale_xy = mi.ScalarTransform4f.scale([0.03, 0.03, -0.03])

    # Right operand applied first (local): rotate → scale → world translate
    #world_translation=(0.623886,0.0,0.104845)
    tx, ty, tz = world_translation
    translate_world = mi.ScalarTransform4f.translate([tx, ty, tz])
    transform = translate_world @ scale_xy @ swap_yz
    # transform = scale_xy
    
    # Build shape loading dictionary
    
    shape_dict = {
        "type": mesh_type,
        "filename": obj_path,
        #"to_world": transform
    }
    '''
    shape_dict = {
        "type": "rectangle",
        # Default Mitsuba rectangle lies on the XZ plane at y=0.
        # Rotate it +90° about the X-axis to make it lie on the XY plane (z=0).
        "to_world": mi.ScalarTransform4f()
            .rotate(axis=[1, 0, 0], angle=180)
            .scale(0.3),  # adjust size if needed
    }
    '''
    shape_dict["bsdf"] = {
        'type': 'mlpbrdf',
        'model_path': model_path,  # Pass model path
        'material_type': material_type,

        
    }
    
    '''
    shape_dict["bsdf"] = {
        'type': 'roughconductor'
    }
    '''
    return shape_dict

    # Construct scene
    scene_dict = {
        "type": "scene",
        "shape": shape_dict,
        "flip_normals": True
    }

    scene = mi.load_dict(scene_dict)
    
    return scene
