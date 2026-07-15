"""Load a Mitsuba XML scene, substituting per-shape neural BSDFs before load.

Editing the XML before ``mi.load_file`` avoids two problems:
  1. Double-instantiating BSDFs (once for the XML, once for a rebuilt scene
     dict).
  2. Shape-id lookups after load — in some Mitsuba 3 builds, shapes that share
     the exact same BSDF can be merged/renamed, so their original ids are no
     longer reachable via ``scene.shapes()``.
"""
import os
import re
import xml.etree.ElementTree as ET

import mitsuba as mi

_ENV_DEFAULT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")


def expand_env(value: str) -> str:
    """Expand environment variables in a string.

    Supports ``${VAR:-default}`` (default used when VAR is unset) plus the
    plain ``$VAR`` / ``${VAR}`` forms handled by ``os.path.expandvars``.
    """
    value = _ENV_DEFAULT_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2)), value)
    return os.path.expandvars(value)


def _set_bsdf_param(bsdf_el, key, value):
    """Append a typed Mitsuba XML parameter element for one BSDF property."""
    if isinstance(value, bool):
        ET.SubElement(bsdf_el, "boolean", {"name": key, "value": "true" if value else "false"})
    elif isinstance(value, (int, float)):
        ET.SubElement(bsdf_el, "float", {"name": key, "value": str(float(value))})
    else:
        ET.SubElement(bsdf_el, "string", {"name": key, "value": str(value)})


def load_scene_with_bsdf_overrides(
    xml_path: str,
    overrides: dict,
    fov=None,
    width=None,
    height=None,
    max_depth=None,
    radiance=None,
):
    """Load a Mitsuba XML scene with per-shape ``mlpbrdf`` BSDFs injected.

    Args:
        xml_path: Path to the scene XML. ``${VAR:-default}`` / ``$VAR``
            patterns inside ``<default value="...">`` elements are expanded
            from the environment before Mitsuba parses the file, so scenes can
            relocate their assets (mesh dirs, scene roots) via env vars.
        overrides: ``{shape_id: bsdf_dict}``. ``bsdf_dict`` must contain at
            least ``{"type": "mlpbrdf", "model_path": ..., "material_type":
            ...}``; remaining keys become typed BSDF properties (bool ->
            <boolean>, number -> <float>, str -> <string>). Shape ids must
            match the XML's ``<shape id="...">``. Shapes not listed keep
            their XML BSDF.
        fov: Optional camera FOV override (degrees).
        width/height: Optional film resolution override (pixels).
        max_depth: Optional path-tracer depth override.
        radiance: Optional ``{shape_id: intensity}`` area-emitter override
            (scalar, applied to r=g=b).

    Returns:
        ``mi.Scene`` loaded with the overrides applied.
    """
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"Scene xml not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Env-var expansion in <default> values ($MESH_DIR-style asset relocation).
    for d in root.findall("default"):
        v = d.get("value", "")
        expanded = expand_env(v)
        if expanded != v:
            d.set("value", expanded)
            print(f"[scene_loader] default {d.get('name')} = {expanded}")

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
            _set_bsdf_param(bsdf_el, k, v)

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

    # Optional camera FOV override (degrees).
    if fov is not None:
        fov_el = root.find(".//sensor//float[@name='fov']")
        if fov_el is not None:
            fov_el.set("value", str(float(fov)))
            print(f"[scene_loader] FOV overridden to {float(fov)}")
        else:
            print("[scene_loader] WARNING: no <float name='fov'> found to override")

    # Optional film resolution override (works whether the XML hardcodes the
    # size or routes it through $resx/$resy <default> parameters).
    for name, value in (("width", width), ("height", height)):
        if value is None:
            continue
        el = root.find(f".//sensor//film/integer[@name='{name}']")
        if el is None:
            film = root.find(".//sensor//film")
            if film is None:
                raise ValueError(f"Cannot override film {name}: no <film> in {xml_path}")
            el = ET.SubElement(film, "integer", {"name": name})
        el.set("value", str(int(value)))
        print(f"[scene_loader] film {name} overridden to {int(value)}")

    # Optional integrator depth override.
    if max_depth is not None:
        el = root.find(".//integrator/integer[@name='max_depth']")
        if el is None:
            integrator = root.find(".//integrator")
            if integrator is None:
                raise ValueError(f"Cannot override max_depth: no <integrator> in {xml_path}")
            el = ET.SubElement(integrator, "integer", {"name": "max_depth"})
        el.set("value", str(int(max_depth)))
        print(f"[scene_loader] max_depth overridden to {int(max_depth)}")

    # Write to a sibling file so relative paths in the XML still resolve.
    out_path = os.path.join(os.path.dirname(xml_path), ".render.generated.xml")
    tree.write(out_path)
    try:
        return mi.load_file(out_path)
    finally:
        # Don't leave stray artifacts next to the canonical scene XML.
        try:
            os.remove(out_path)
        except OSError:
            pass
