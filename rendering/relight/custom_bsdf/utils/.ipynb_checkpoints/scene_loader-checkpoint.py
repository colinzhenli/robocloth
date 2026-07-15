import mitsuba as mi
import os
import numpy as np

mi.set_variant("cuda_ad_rgb")  # or "scalar_rgb" / "llvm_ad_rgb"
from omegaconf import OmegaConf

def load_uv_obj_to_mitsuba_scene(args,cfg,obj_path="/home/featurize/data/cloth.obj"):
    """
    Read OBJ file with UV and construct a Mitsuba Scene.

    Parameters:
        obj_path (str): .obj file path, needs to include UV information.
        texture_path (str or None): Optional, texture image path (like .jpg/.png).
        scale (float): Scale ratio for geometry.
        rotate_x90 (bool): Whether to rotate geometry -90° around X-axis, converting from Blender export coordinates to Mitsuba's default orientation.

    Returns:
        scene (mi.Scene): Mitsuba scene object.
    """
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"OBJ file does not exist: {obj_path}")

    swap_yz = mi.ScalarTransform4f.rotate([0, 1, 0], -90)

    # Then scale x and y axes to 0.2 (note scaling order and application order)
    scale_xy = mi.ScalarTransform4f.scale([0.3, 0.3, -0.3])

    # First rotate then scale (matrix multiplication executes right multiplication first)
    transform = scale_xy @ swap_yz
    # transform = scale_xy
    
    # Build shape loading dictionary
    
    shape_dict = {
        "type": "obj",
        "filename": obj_path,
        "to_world": transform
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
        'model_path': args.model_path,  # Pass model path
        'material_type': cfg.material.type,
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
    bsdf=mi.load_dict(shape_dict["bsdf"])
    
    return shape_dict

    # Construct scene
    scene_dict = {
        "type": "scene",
        "shape": shape_dict,
        "flip_normals": True
    }

    scene = mi.load_dict(scene_dict)
    
    return scene
