import torch
import torch.nn.functional as NF
from torch.utils.data import Dataset
import json
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import math
from pathlib import Path
from torch.utils.data import IterableDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.io import load_camera_turntable_light_metadata, load_camera_metadata, load_camera_metadata_from_robotic_log
import threading, queue, time
from dataclasses import dataclass

def build_4x4(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T

def _cv_to_gl(cv):
    # convert to GL convention used in iNGP
    gl = cv * torch.tensor([1, -1, -1, 1])
    return gl

def get_ray_directions(H, W, focal, cx, cy, distortion):
    """ get camera ray direction with radial distortion correction, using opengl convention
    Args:
        H,W: height and width
        focal: focal length
        cx, cy: principal point coordinates
        distortion: radial distortion coefficient k1
    """
    x_coords = torch.linspace(0.5, W - 0.5, W)
    y_coords = torch.linspace(0.5, H - 0.5, H)
    j, i = torch.meshgrid([y_coords, x_coords])
    
    # Convert to normalized coordinates relative to principal point
    x_norm = (i - cx) / focal
    y_norm = (j - cy) / focal
    
    # Apply radial distortion correction
    r_squared = x_norm**2 + y_norm**2
    distortion_factor = 1 + distortion * r_squared
    
    x_corrected = x_norm * distortion_factor
    y_corrected = y_norm * distortion_factor
    
    directions = torch.stack([x_corrected, -y_corrected, -torch.ones_like(i)], -1)

    return directions

def get_rays(directions, c2w, focal=None):
    """ world space camera ray
    Args:
        directions: camera ray direction (local)
        c2w: 3x4 camera to world matrix
        focal: if not None, return ray differentials as well
    """
    R = c2w[:,:3]
    rays_d = directions @ R.T
    
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    if focal is not None:
        dxdu = torch.tensor([1.0/focal,0,0])[None,None].expand_as(directions)@R.T
        dydv = torch.tensor([0,1.0/focal,0])[None,None].expand_as(directions)@R.T
        dxdu = dxdu.view(-1,3)
        dydv = dydv.view(-1,3)
        return rays_o, rays_d, dxdu, dydv
    else:
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        return rays_o, rays_d

def read_image(path, img_hw):
    img = plt.imread(path)[...,:3]
    assert img.shape[0] == img_hw[0]
    assert img.shape[1] == img_hw[1]
    return torch.from_numpy(img.astype(np.float32))

def open_exr(file,img_hw):
    """ open image exr file """
    img = cv2.imread(str(file),cv2.IMREAD_UNCHANGED)
    assert img.shape[0] == img_hw[0]
    assert img.shape[1] == img_hw[1]
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = img[...,[2,1,0]]
    img = torch.from_numpy(img.astype(np.float32))
    return img

def get_c2w(camera):
    position = torch.tensor(camera['position'], dtype=torch.float32)
    target = torch.tensor(camera['look_at'], dtype=torch.float32)
    up = torch.tensor(camera.get('up', [0,1,0]), dtype=torch.float32)
     
    forward = target - position
    forward = forward / torch.norm(forward)
    # Ensure `up` is not parallel to `forward`
    if torch.abs(torch.dot(forward, up)) > 0.99:  # Too parallel, adjust up
        up = torch.tensor([1.0, 0.0, 0.0]) if torch.abs(forward[0]) < 0.99 else torch.tensor([0.0, 1.0, 0.0])
    """ right hand coordinate system """
    right = torch.cross(up, forward)
    right = right / torch.norm(right)
    up = torch.cross(forward, right)
    
    c2w = torch.eye(4)
    c2w[:3,:3] = torch.stack([right, up, forward], dim=1)
    c2w[:3,3] = position
    c2w = _cv_to_gl(c2w)
    c2w = c2w[:3,:4]
    return c2w

def get_c2w_from_robot_pose(camera_info, R_c2g, t_c2g):
    """ get camera to world matrix from robot pose """
    g2w = build_4x4(camera_info["rotation_matrix"], camera_info["position"])
    c2g = build_4x4(R_c2g, t_c2g)
    c2w = g2w @ c2g
    return torch.from_numpy(c2w[:3, :4]).float()

def load_metadata(colmap_camera, metadata_path, camera_metadata_path, gt_folder, cfg, debug, debug_num, split, turntable_center, turntable_axis, R_c2g, t_c2g, start_idx, use_fixed_val=False, fixed_val_num=240):
    metadata, _, _= load_camera_turntable_light_metadata(metadata_path)
    
    if colmap_camera:
        camera_metadata = load_camera_metadata(camera_metadata_path)
    else:
        camera_metadata = load_camera_metadata_from_robotic_log(metadata_path, turntable_center, turntable_axis, R_c2g, t_c2g)
    
    # Filter out metadata entries with non-existent image files and filtered_puple_ids
    valid_metadata = []
    # Load unmatched_scan_ids from parent folder of gt_folder
    parent_folder = os.path.dirname(gt_folder)
    unmatched_scan_ids_path = os.path.join(parent_folder, "unmatched_scan_ids.json")
    if os.path.exists(unmatched_scan_ids_path):
        with open(unmatched_scan_ids_path, "r") as f:
            unmatched_scan_ids = json.load(f)
    else:
        unmatched_scan_ids = []
    # unmatched_scan_ids = []
    for item in metadata:
        # Add "masked_" prefix to filename
        file_name = item["filename"]
        # if not file_name.startswith("masked_"):
        #     file_name = "masked_" + file_name
        #     item["filename"] = file_name
        # file_name = item["filename"]
        img_path = os.path.join(gt_folder, file_name)
        overall_id = item.get("overall_id")
        
        # Skip if image file doesn't exist or is 0 bytes
        if not os.path.exists(img_path):
            print(f"Warning: Image file {img_path} does not exist, skipping from metadata...")
            continue
        
        # Skip if image file is 0 bytes
        if os.path.getsize(img_path) == 0:
            print(f"Warning: Image file {img_path} is 0 bytes, skipping from metadata...")
            continue
            
        # Skip if overall_id is in unmatched_scan_ids
        if int(overall_id) in unmatched_scan_ids:
            print(f"Warning: overall_id {overall_id} is in unmatched_scan_ids, skipping from metadata...")
            continue
            
        valid_metadata.append(item)
    
    metadata = valid_metadata
    # # Filter out images with overall_id >= 1000
    # metadata = [item for item in metadata if int(item.get("overall_id", 0)) < 1000]
    # print(f"After filtering overall_id >= 1000: {len(metadata)} images remaining")
    # print(f"Loaded {len(metadata)} valid images out of {len(metadata) + len([item for item in metadata if not os.path.exists(os.path.join(gt_folder, item['filename']))])} total metadata entries")
    
    total_images = len(metadata)
    
    # Split metadata into training and validation sets with fixed random seed
    torch.manual_seed(42)  # Fixed seed for reproducible splits
    if debug:
        indices = list(range(total_images))
    else:
        indices = torch.randperm(total_images)
    
    # Use 80% for training
    if debug:
        if debug_num == -1:
            debug_num = total_images
        selected_indices = indices[start_idx:start_idx+debug_num]
        # For debugging, still split into train/val but use a smaller subset
        split_idx = int(0.8 * debug_num)
        # if split == 'train':
        #     selected_indices = selected_indices[:split_idx]
        # else:
        #     selected_indices = selected_indices[split_idx:]
        selected_metadata = [metadata[i] for i in selected_indices]
        print(f"Debug mode: {len(selected_metadata)} images selected")
    else:
        if use_fixed_val:
            Shuffle_val = True
            # Use first fixed_val_num images for validation, rest for training
            # Split first, then permute each set separately
            if split == 'val':
                selected_metadata = metadata[0:fixed_val_num]
                print(f"Fixed validation set: {len(selected_metadata)} images (first {fixed_val_num})")
            else:
                # selected_metadata = metadata
                selected_metadata = metadata[fixed_val_num:]
                print(f"Training set: {len(selected_metadata)} images (after first {fixed_val_num})")
            # Permute the selected metadata (both val and train when Shuffle_val=True)
            if Shuffle_val or split != 'val':
                perm_indices = torch.randperm(len(selected_metadata)).tolist()
                selected_metadata = [selected_metadata[i] for i in perm_indices]
        else:
            split_idx = int(0.8 * total_images)
            if split == 'train':
                selected_indices = indices[:split_idx]
            else:
                selected_indices = indices[split_idx:]
                
            selected_metadata = [metadata[i] for i in selected_indices] 
        
    return selected_metadata, camera_metadata
        


class RealImageDenseDataset(IterableDataset):
    """Stage-2 training loader: preloads ALL images for the material into RAM
    once, then samples uniformly each iteration. No chunk swapping, no background thread.

    Per-pixel transforms (CCM clip, luminance>1e-7 mask, ray construction) are reused
    verbatim from the legacy chunked loader's _preload_given_metadata, so the (rays, rgbs,
    camera_ids, emitter_ids) multiset is bit-identical to a single chunk that covers
    all metadata.

    Caveats:
    - Multi-resolution (downsample_iter) is not supported (data preloaded once at
      full res); a warning is printed if downsample_iter != [-1, -1].
    - importance_sampling=True uses a global PDF instead of per-chunk PDF — this
      diverges from the chunk loader's behavior. Default importance_sampling=False
      is unaffected.
    """

    def __init__(self, cfg, gt_folder, split):
        self.cfg = cfg
        self.pixel = True
        self.rays_num = cfg.data.rays_num
        self.num_view_batch = cfg.renderer.camera.views_per_batch
        self.importance_sampling = cfg.data.importance_sampling
        self.use_single_chunk_sampling = cfg.data.use_single_chunk_sampling
        self.multi_resolution = cfg.data.multi_resolution
        self.downsample_iter = cfg.data.downsample_iter
        self.gt_folder = gt_folder
        self.debug = cfg.data.debug
        self.debug_num = cfg.data.debug_num
        self.intrinsics = cfg.renderer.camera.intrinsics
        self.cx = self.intrinsics['cx']
        self.cy = self.intrinsics['cy']
        self.distortion = self.intrinsics['distortion']
        self.focal = self.intrinsics['focal_length']
        self.img_hw = (self.intrinsics['height'], self.intrinsics['width'])
        self.ccm = np.array(cfg.data.ccm)
        self.R_c2g = cfg.renderer.camera.R_c2g
        self.t_c2g = cfg.renderer.camera.t_c2g
        self.turntable_center = cfg.renderer.emitter.turntable.center
        self.turntable_axis = cfg.renderer.emitter.turntable.axis
        self.colmap_camera = cfg.renderer.camera.colmap_camera
        self.start_idx = cfg.data.start_idx
        self.use_fixed_val = cfg.data.use_fixed_val
        self.hold_out_val_num = cfg.data.hold_out_val_num

        metadata_path = cfg.data.metadata_path
        camera_metadata_path = cfg.data.camera_metadata_path
        self.all_metadata, self.camera_metadata = load_metadata(
            self.colmap_camera,
            metadata_path,
            camera_metadata_path,
            gt_folder,
            cfg,
            self.debug,
            self.debug_num,
            split,
            self.turntable_center,
            self.turntable_axis,
            self.R_c2g,
            self.t_c2g,
            self.start_idx,
            self.use_fixed_val,
            self.hold_out_val_num,
        )

        if list(self.downsample_iter) != [-1, -1]:
            print(f"[RealImageDenseDataset] WARNING: downsample_iter={list(self.downsample_iter)} "
                  f"is ignored — data is preloaded once at full resolution.")

        directions = get_ray_directions(
            self.img_hw[0], self.img_hw[1], self.focal, self.cx, self.cy, self.distortion
        )

        # Reuse the thread-safe preloader. Pass our
        # instance via __get__ so it can read self.gt_folder / camera_metadata / etc.
        # which we mirror above with identical semantics.
        print(f"[RealImageDenseDataset] Preloading {len(self.all_metadata)} images into RAM...")
        self.rays, self.rgbs, self.camera_ids, self.emitter_ids, self.pdf = (
            self._preload_given_metadata(self.all_metadata, directions, downsample_scale=1)
        )
        gb = lambda t: t.element_size() * t.numel() / 1e9
        print(f"[RealImageDenseDataset] Loaded {self.rays.shape[0]:,} rays "
              f"(rays={gb(self.rays):.2f} GB, rgbs={gb(self.rgbs):.2f} GB, "
              f"cam_ids={gb(self.camera_ids):.2f} GB, emit_ids={gb(self.emitter_ids):.2f} GB, "
              f"pdf={gb(self.pdf):.2f} GB)")

        self.step = 0

    def set_step(self, step: int):
        # No-op: preloaded data is fixed. Kept for trainer-loop compatibility.
        self.step = step

    def __iter__(self):
        while True:
            N_total = self.rays.shape[0]
            if self.importance_sampling and self.pdf.numel() > 0:
                sample_idx = torch.multinomial(self.pdf, self.rays_num, replacement=True)
                pdf_vals = self.pdf[sample_idx] * N_total
            else:
                N = min(self.rays_num, N_total)
                sample_idx = torch.randint(0, N_total, (N,), dtype=torch.long)
                pdf_vals = torch.ones(N)

            yield {
                'rays':        self.rays[sample_idx],
                'rgbs':        self.rgbs[sample_idx],
                'emitter_ids': self.emitter_ids[sample_idx],
                'camera_ids':  self.camera_ids[sample_idx],
                'pdf':         pdf_vals,
                'gt_params':   torch.zeros(1),
            }

    def _preload_given_metadata(self, metadata_list, directions, downsample_scale=1):
        # This body mirrors your original implementation, except it uses
        # (metadata_list, directions) passed in, so it is safe for a background thread.
        all_rays = []
        all_rgbs = []
        all_emitter_ids = []
        all_camera_ids = []
        print(f"Loading chunk with {len(metadata_list)} images...")
        for img_data in metadata_list:
            # Get camera info using camera_id
            camera_info = self.camera_metadata[str(img_data["camera_id"])]
            camera_dict = {
                "position": camera_info["position"],
                "rotation_matrix": camera_info["rotation_matrix"],
            }

            # Generate rays for this camera
            # c2w = get_c2w_from_robot_pose(camera_dict, self.R_c2g, self.t_c2g)
            c2w = torch.from_numpy(build_4x4(camera_dict["rotation_matrix"], camera_dict["position"])).float()
            # c2w = _cv_to_gl(c2w)[:3, :4]
            c2w = c2w[:3, :4]
            rays_o, rays_d, dxdu, dydv = get_rays(directions, c2w, focal=self.intrinsics['focal_length'])
            rays = torch.cat([rays_o, rays_d, dxdu, dydv], dim=-1)

            # Load original RGB image (without gamma correction)
            file_name = img_data["filename"]
            img_path = os.path.join(self.gt_folder, file_name)

            if img_path.endswith('.png'):
                # Load PNG image (original linear RGB)
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"Warning: Could not load image {img_path}, skipping...")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img @ self.ccm
                img = img.clip(0, None)
                img = torch.from_numpy(img).float()
            elif img_path.endswith('.exr'):
                # Load EXR image (already linear)
                img = open_exr(img_path, self.img_hw)
            else:
                raise ValueError(f"Unsupported image format: {img_path}")

            img_flat = img.reshape(-1, 3)
            # Downsample the image if needed
            if downsample_scale > 1:
                # Reshape to image format for downsampling
                img_reshaped = img_flat.reshape(self.img_hw[0], self.img_hw[1], 3)
                # Downsample using average pooling
                img_downsampled = torch.nn.functional.avg_pool2d(
                    img_reshaped.permute(2, 0, 1).unsqueeze(0),  # [1, 3, H, W]
                    kernel_size=downsample_scale,
                    stride=downsample_scale
                ).squeeze(0).permute(1, 2, 0)  # [H', W', 3]
                img_flat = img_downsampled.reshape(-1, 3)

            # Get emitter ID directly from metadata
            emitter_id = img_data["emitter_id"]
            emitter_ids = torch.full((rays.shape[0],), emitter_id, dtype=torch.long)
            camera_ids = torch.full((rays.shape[0],), int(img_data["camera_id"]), dtype=torch.long)

            # Filter out low luminance rays
            luminance = (0.2126 * img_flat[..., 0] +
                         0.7152 * img_flat[..., 1] +
                         0.0722 * img_flat[..., 2])
            luminance_threshold = 1e-7
            valid_mask = luminance > luminance_threshold

            # Apply mask to filter out low luminance rays
            rays = rays[valid_mask]
            img_flat = img_flat[valid_mask]
            emitter_ids = emitter_ids[valid_mask]
            camera_ids = camera_ids[valid_mask]
            all_rays.append(rays)
            all_rgbs.append(img_flat)
            all_emitter_ids.append(emitter_ids)
            all_camera_ids.append(camera_ids)
        
        print(f"Finished loading chunk: {len(all_rays)} images processed")

        # Concatenate all data
        if len(all_rays) == 0:
            # Empty fallback
            rays = torch.empty(0, 12)
            rgbs = torch.empty(0, 3)
            emitter_ids = torch.empty(0, dtype=torch.long)
            camera_ids = torch.empty(0, dtype=torch.long)
            pdf = torch.empty(0)
            return (rays, rgbs, camera_ids, emitter_ids, pdf)

        rays = torch.cat(all_rays)
        rgbs = torch.cat(all_rgbs)
        emitter_ids = torch.cat(all_emitter_ids)
        camera_ids = torch.cat(all_camera_ids)

        # Calculate luminance for importance sampling
        luminance = (0.2126 * rgbs[..., 0] +
                     0.7152 * rgbs[..., 1] +
                     0.0722 * rgbs[..., 2])         # (N_tot,)
        luminance = torch.abs(luminance)
        pdf = luminance.detach() / (luminance.sum() + 1e-12)  # p_i  (no grad)

        # Randomly permute the data
        perm_indices = torch.randperm(rays.shape[0])
        rays = rays[perm_indices]
        rgbs = rgbs[perm_indices]
        emitter_ids = emitter_ids[perm_indices]
        camera_ids = camera_ids[perm_indices]
        pdf = pdf[perm_indices]
        return (rays, rgbs, camera_ids, emitter_ids, pdf)


class RealValDataset(Dataset):
    """ validation dataset that loads images from metadata, returns complete images """
    def __init__(self, cfg, gt_folder):
        self.cfg = cfg
        self.pixel = False
        self.gt_folder = gt_folder
        self.debug = cfg.data.debug
        self.debug_num = cfg.data.debug_num
        self.start_idx = cfg.data.start_idx
        self.intrinsics = cfg.renderer.camera.intrinsics
        self.cx = self.intrinsics['cx']
        self.cy = self.intrinsics['cy']
        self.distortion = self.intrinsics['distortion']
        self.focal = self.intrinsics['focal_length']
        self.img_hw = (self.intrinsics['height'], self.intrinsics['width'])
        # self.ccm = cfg.data.ccm
        self.ccm = np.array(cfg.data.ccm) 
        
        # get R_c2g and t_c2g from cfg
        self.R_c2g = cfg.renderer.camera.R_c2g
        self.t_c2g = cfg.renderer.camera.t_c2g
        self.colmap_camera = cfg.renderer.camera.colmap_camera # whether to use colmap camera or robotic log camera
        self.turntable_center = cfg.renderer.emitter.turntable.center
        self.turntable_axis = cfg.renderer.emitter.turntable.axis
        self.use_fixed_val = cfg.data.use_fixed_val
        self.hold_out_val_num = cfg.data.hold_out_val_num
        self.valid_num = cfg.data.valid_num
        # Load metadata from JSON file
        metadata_path = cfg.data.metadata_path
        camera_metadata_path = cfg.data.camera_metadata_path
        if cfg.data.valid_on_train_set:
            split = 'train'
        else:
            split = 'val'
        self.metadata, self.camera_metadata = load_metadata(self.colmap_camera, metadata_path, camera_metadata_path, gt_folder, cfg, self.debug, self.debug_num, split, self.turntable_center, self.turntable_axis, self.R_c2g, self.t_c2g, self.start_idx, self.use_fixed_val, self.hold_out_val_num)
        # Keep ALL val images so val metrics are computed on the full held-out
        # set; ``valid_num`` is consumed by the trainer to gate per-view image
        # saving (the first ``valid_num`` items, which are random because
        # load_metadata returns either a torch.randperm-ordered slice or a
        # post-permutation set when use_fixed_val=True).
        n_save = min(self.valid_num, len(self.metadata)) if self.valid_num > 0 else len(self.metadata)
        print(f"RealValDataset: {len(self.metadata)} val images for metrics; "
              f"saving images for first {n_save}")
        self.directions = get_ray_directions(self.img_hw[0], self.img_hw[1], self.focal, self.cx, self.cy, self.distortion)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_data = self.metadata[idx]
        # if idx == 59:
        #     print(f"Camera ID: {img_data['camera_id']}")
        #     print(f"Emitter ID: {img_data['emitter_id']}")
        #     print(f"Filename: {img_data['filename']}")
        #     print(f"Position: {camera_info['position']}")
        #     print(f"Rotation Matrix: {camera_info['rotation_matrix']}")
        #     print(f"C2W: {c2w}")
        #     print(f"Rays: {rays}")
        #     print(f"Img Flat: {img_flat}")
        #     print(f"Emitter IDs: {emitter_ids}")
        
        # Get camera info using camera_id
        camera_id = img_data["camera_id"]
        # overall_id = 0 # debug with the first camera
        camera_info = self.camera_metadata[str(camera_id)]
        camera_dict = {
            "position": camera_info["position"],
            "rotation_matrix": camera_info["rotation_matrix"],
        }
        
        # Generate rays for this camera
        # c2w = get_c2w_from_robot_pose(camera_dict, self.R_c2g, self.t_c2g)
        c2w = torch.from_numpy(build_4x4(camera_dict["rotation_matrix"], camera_dict["position"])).float()
        # c2w = _cv_to_gl(c2w)[:3, :4]
        c2w = c2w[:3, :4]
        rays_o, rays_d, dxdu, dydv = get_rays(self.directions, c2w, focal=self.focal)
        rays = torch.cat([rays_o, rays_d, dxdu, dydv], dim=-1)
        
        # Load original RGB image (without gamma correction)
        img_path = os.path.join(self.gt_folder, img_data["filename"])
                
        if img_path.endswith('.png'):
            # Load PNG image (original linear RGB)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float64)
            img = img @ self.ccm
            img = img.clip(0, None)  # match training loader: CCM on clipped sensor data yields unphysical negatives that NaN logrel's log_mapping
            img = torch.from_numpy(img).float()
        elif img_path.endswith('.exr'):
            # Load EXR image (already linear)
            img = open_exr(img_path, self.img_hw)
        else:
            raise ValueError(f"Unsupported image format: {img_path}")
        
        img_flat = img.reshape(-1, 3)
        
        # Get emitter ID directly from metadata
        emitter_ids = torch.full((rays.shape[0],), img_data["emitter_id"], dtype=torch.long)
        # emitter_ids = torch.full((rays.shape[0],), idx, dtype=torch.long)
        
        camera_ids = torch.full((rays.shape[0],), int(camera_id), dtype=torch.long)
        return {
            'rays': rays,
            'rgbs': img_flat,
            'emitter_ids': emitter_ids,
            'gt_params': torch.zeros(1),
            'camera_ids': camera_ids,
        }

