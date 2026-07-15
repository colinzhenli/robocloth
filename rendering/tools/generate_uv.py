"""UV toolkit: unwrap a mesh that has no UV coordinates.

The neural material is a latent *texture*, so every shape it is applied to
needs a UV parametrization. This tool Smart-UV-unwraps an OBJ/PLY/GLB in
headless Blender and exports a UV-mapped PLY ready for render.py; combine
with the per-object "uv_tiling" knob in materials.json to control the
texture repeat density.

Requires the bpy wheel (Python 3.10): pip install bpy==3.6.0

Usage:
    python rendering/tools/generate_uv.py input_mesh.obj  [--angle-limit 89] [--island-margin 0.02]
"""
import bpy
import bmesh
import numpy as np

def uv_unwrap_and_compute_TBN(obj_name, angle_limit=89.0, island_margin=0.02):
    # Get the object
    obj = bpy.data.objects.get(obj_name)
    if not obj or obj.type != 'MESH':
        print(f"❌ Mesh object named {obj_name} not found")
        return None, None

    # Ensure we're in object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Deselect all, select target object and make it active
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # Enter edit mode and perform UV unwrapping
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    # Check if UV map already exists
    if obj.data.uv_layers.active is None:
        print("No UV map found, creating one...")
        bpy.ops.uv.smart_project(angle_limit=angle_limit, island_margin=island_margin)
    else:
        print(f"UV map '{obj.data.uv_layers.active.name}' already exists, skipping unwrapping")
    
    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Calculate tangents (must have UV and normals first)
    uvmap_name = obj.data.uv_layers.active.name
    #for uv in obj.data.uv_layers:
    #    print(uv.name)
    obj.data.calc_tangents(uvmap=uvmap_name)

    # Get UV, Tangent, Bitangent, Normal (per loop)
    uv_map = []
    tbn_list = []



    uv_layer = obj.data.uv_layers.active.data
    loops = obj.data.loops
    tangents = obj.data.loops

    for loop in obj.data.loops:
        loop_index = loop.index
        vertex_index = loop.vertex_index
        uv = uv_layer[loop_index].uv
        tangent = loop.tangent
        normal = loop.normal
        bitangent = loop.bitangent_sign * normal.cross(tangent)
        #print(tangent,normal,bitangent)

        uv_map.append((uv.x, uv.y))
        tbn_list.append((
            (tangent.x, tangent.y, tangent.z),
            (bitangent.x, bitangent.y, bitangent.z),
            (normal.x, normal.y, normal.z),
        ))

    print(f"✅ UV unwrapping and TBN calculation completed, total {len(uv_map)} loops")
    return uv_map, tbn_list

def export_mesh_with_uv_ply(input_path, angle_limit=89.0, island_margin=0.02, apply_modifiers=False, triangulate=False):
    # Fetch object
    output_path = input_path.replace(".ply", "_uv.ply")
    # Import the PLY file first
    bpy.ops.import_mesh.ply(filepath=input_path)
    
    # Get the imported object (it will be the active object after import)
    obj = bpy.context.active_object
    if not obj or obj.type != 'MESH':
        raise ValueError(f"❌ Mesh object named {input_path} not found")

    # Make active/selected
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Ensure there is a UV map (unwrap if missing)
    if obj.data.uv_layers.active is None:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=angle_limit, island_margin=island_margin)
        bpy.ops.object.mode_set(mode='OBJECT')

    # Optionally apply modifiers so the exported mesh matches the viewport
    if apply_modifiers:
        for m in list(obj.modifiers):
            bpy.ops.object.modifier_apply(modifier=m.name)

    # Optionally triangulate (some pipelines prefer triangulated faces)
    if triangulate:
        tri = obj.modifiers.new(name="__triangulate__", type='TRIANGULATE')
        bpy.ops.object.modifier_apply(modifier=tri.name)

    # Export as PLY with UVs
    print(obj.scale)
    bpy.ops.export_mesh.ply(
        filepath=output_path,
        use_selection=True,
        use_normals=True,
        use_uv_coords=True
    )
    print(f"📦 Exported '{input_path}' with UVs → {output_path}")

# Example
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process mesh with UV unwrapping")
    parser.add_argument("input_path", help="Path to the input mesh file")
    args = parser.parse_args()
    
    export_mesh_with_uv_ply(input_path=args.input_path)

if __name__ == "__main__":
    main()

# # Example usage
# uvs, tbn = uv_unwrap_and_compute_TBN("Cut_mesh")