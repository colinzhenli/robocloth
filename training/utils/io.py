import json
import os
import glob
import torch
from utils.transform import build_rot_about_point, build_cw_rotz_from_deg, rodrigues_axis_angle, build_4x4

def load_camera_turntable_light_metadata(json_path):
    """
    Load camera and light relation from JSON file and create metadata for images.
    
    Args:
        json_path (str): Path to the JSON file containing camera-light relations
    
    Returns:
        dict: Metadata containing:
            - metadata: list of dicts with keys ['overall_id', 'camera_id', 'emitter_id', 'filename', 'turn_angle']
            - camera_metadata: dict mapping overall_id to camera info
            - emitter_metadata: dict mapping light_id to light info
    """
    # Load JSON file
    with open(json_path, 'r') as f:
        camera_light_data = json.load(f)
    
    # Create camera and emitter metadata dictionaries
    camera_metadata = {}
    emitter_metadata = {}
    
    # Create metadata list
    metadata = []
    
    for item in camera_light_data:
        overall_id = item['scan_id']
        camera_id = item['camera_id']
        light_id = item['light_id']
        if 'turn_angle' in item:
            turn_angle = item['turn_angle']
        else:
            turn_angle = None
        filename = item['filename']
        
        # Store camera metadata using overall_id
        camera_metadata[str(camera_id)] = {
            'position': [pos / 1000.0 for pos in item['position']],
            'rotation_matrix': item['rotation_matrix'],
        }
        
        # # Store emitter metadata (light info)
        # if str(light_id) not in emitter_metadata:
        #     emitter_metadata[str(light_id)] = {
        #         'position': [pos / 1000.0 for pos in item['position_light']],
        #         'rotation_matrix': item['rotation_matrix_light']
        #     }
        
        # Create metadata entry
        metadata.append({
            'overall_id': overall_id,
            'camera_id': camera_id,
            'emitter_id': light_id,  # Using emitter_id to match SphereImageDataset
            'filename': filename,
            'turn_angle': turn_angle
        })
    
    return metadata, camera_metadata, None

def load_camera_metadata_from_robotic_log(json_path, turntable_center, turntable_axis, R_c2g, t_c2g):
    """
    Camera is in openGL convention.
    Load camera metadata from gripper JSON file.
    """
    import json
    
    # Load JSON file
    with open(json_path, 'r') as f:
        camera_light_data = json.load(f)
    
    # Create camera metadata dictionary
    camera_metadata = {}
    
    for item in camera_light_data:
        camera_id = item['camera_id']
        # Store camera metadata only for non-appeared camera id
        if str(camera_id) not in camera_metadata:
            # Get turn angle if available
            turn_angle = item['turn_angle']
            
            # Build camera-to-world transformation
            # First get gripper-to-world, then transform from camera-to-gripper
            g2w = build_4x4(item['rotation_matrix'], [pos / 1000.0 for pos in item['position']])
            c2g = build_4x4(R_c2g, t_c2g)
            c2w = g2w @ c2g
            
            # Transform camera back to 0-angle world
            Tw2w0 = build_rot_about_point(rodrigues_axis_angle(turntable_axis, -turn_angle), turntable_center)
            c2w0 = Tw2w0 @ c2w
            
            # Extract position and rotation matrix from c2w0
            camera_metadata[str(camera_id)] = {
                'position': c2w0[:3, 3].tolist(),
                'rotation_matrix': c2w0[:3, :3].tolist(),
            }
    
    return camera_metadata

def load_camera_metadata(json_path):
    """
    Load only camera metadata from JSON file. Camera is in openGL convention.
    
    Args:
        json_path (str): Path to the JSON file containing camera and light data
        
    Returns:
        dict: Dictionary mapping camera_id (str) to camera metadata
    """
    import json
    
    # Load JSON file
    with open(json_path, 'r') as f:
        camera_light_data = json.load(f)
    
    # Create camera metadata dictionary
    camera_metadata = {}
    
    for item in camera_light_data:
        camera_id = item['camera_id']
        # Store camera metadata only for non-appeared camera id
        if str(camera_id) not in camera_metadata:
            camera_metadata[str(camera_id)] = {
                'position': [pos / 1000.0 for pos in item['position']],
                'rotation_matrix': item['rotation_matrix'],
            }
    
    return camera_metadata

def rotation_position_to_light2world(rotation_matrix, position):
    """
    Convert rotation matrix and position to light-to-world transformation matrix.
    
    Args:
        rotation_matrix (list or tensor): 3x3 rotation matrix
        position (list or tensor): 3D position vector
        
    Returns:
        torch.Tensor: 4x4 light-to-world transformation matrix
    """

    
    # Convert to tensors if needed
    if not isinstance(rotation_matrix, torch.Tensor):
        rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32)
    if not isinstance(position, torch.Tensor):
        position = torch.tensor(position, dtype=torch.float32)
    
    # Ensure proper shapes
    rotation_matrix = rotation_matrix.view(3, 3)
    position = position.view(3)
    
    # Create 4x4 transformation matrix
    light2world = torch.eye(4, dtype=torch.float32)
    light2world[:3, :3] = rotation_matrix
    light2world[:3, 3] = position / 1000.0 # convert to meter
    
    return light2world

def read_light_transforms(json_path, turntable_center, turntable_axis, base2_to_base1):
    """
    Read light data from JSON file and return light-to-world transformation matrices.
    
    Args:
        json_path (str): Path to the JSON file containing light data
        
    Returns:
        torch.Tensor: (N, 4, 4) tensor where N is number of lights,
                     each 4x4 matrix is a light-to-world transformation
    """
    import json
    import torch
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    light_transforms = []
    
    # Read the first position for each unique light ID
    seen_light_ids = set()
    
    for item in data:
        light_id = item.get('light_id', 0)
        turn_angle = item.get('turn_angle', 0.0)
        
        # Only process if we haven't seen this light ID before
        if light_id not in seen_light_ids:
            seen_light_ids.add(light_id)
            
            rotation_matrix = item['rotation_matrix_light']
            position = item['position_light'] # convert to meter

            # Convert to light-to-world transformation matrix
            light2world = rotation_position_to_light2world(rotation_matrix, position)
            light2world = base2_to_base1 @ light2world.cuda()
            Tw2w0 = build_rot_about_point(rodrigues_axis_angle(turntable_axis, -turn_angle), turntable_center) # tranform world back to 0-angle world
            Tw2w0 = torch.from_numpy(Tw2w0).float().cuda()
            light_transforms.append(Tw2w0 @ light2world)
    
    return torch.stack(light_transforms).cuda()  # (N, 4, 4)

def load_colmap_sparse_pointcloud(sparse_path):
    """
    Load COLMAP sparse reconstruction and extract 3D points with their 2D pixel observations.
    
    Args:
        sparse_path (str): Path to COLMAP sparse reconstruction folder (containing cameras.bin/txt, images.bin/txt, points3D.bin/txt)
    
    Returns:
        tuple: (points3d_tensor, observations_tensor)
            - points3d_tensor: torch.Tensor of shape (N, 3) containing 3D point coordinates
            - observations_tensor: torch.Tensor of shape (N, 3) where:
                - [:, 0] is image_id
                - [:, 1] is pixel x coordinate
                - [:, 2] is pixel y coordinate
    
    Note: Each 3D point may be observed by multiple cameras. This function returns one observation
          per 3D point (the first observation in the track). If you need all observations, 
          the output will have more than N rows.
    """
    import sys
    import os
    
    # Add the calibration directory to path to import read_write_model
    calib_dir = os.path.join(os.path.dirname(__file__), '..', 'recon', 'calibration')
    if calib_dir not in sys.path:
        sys.path.insert(0, calib_dir)
    
    from read_write_model import read_points3D_binary, read_points3D_text, read_images_binary, read_images_text, detect_model_format
    
    # Detect format (.bin or .txt)
    if detect_model_format(sparse_path, ".bin"):
        ext = ".bin"
        points3D = read_points3D_binary(os.path.join(sparse_path, "points3D" + ext))
        images = read_images_binary(os.path.join(sparse_path, "images" + ext))
    elif detect_model_format(sparse_path, ".txt"):
        ext = ".txt"
        points3D = read_points3D_text(os.path.join(sparse_path, "points3D" + ext))
        images = read_images_text(os.path.join(sparse_path, "images" + ext))
    else:
        raise ValueError(f"Could not detect COLMAP model format in {sparse_path}")
    
    # Extract data for all observations
    points_list = []
    observations_list = []
    
    for point3D_id, point3D in points3D.items():
        xyz = point3D.xyz  # 3D coordinates
        image_ids = point3D.image_ids  # Which images see this point
        point2D_idxs = point3D.point2D_idxs  # Index into the image's xys array
        
        # For each observation of this 3D point
        for img_id, point2D_idx in zip(image_ids, point2D_idxs):
            # Get the 2D pixel coordinates from the image
            if img_id in images:
                image = images[img_id]
                xy = image.xys[point2D_idx]  # 2D pixel coordinates
                
                # Store the 3D point and its observation
                points_list.append(xyz)
                observations_list.append([img_id, xy[0], xy[1]])
    
    # Convert to torch tensors
    points3d_tensor = torch.tensor(points_list, dtype=torch.float32)  # Shape: (M, 3)
    observations_tensor = torch.tensor(observations_list, dtype=torch.float32)  # Shape: (M, 3)
    
    return points3d_tensor, observations_tensor