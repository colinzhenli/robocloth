import torch
import torch.nn.functional as NF
from torchvision import transforms as TF
from torch.utils.data import Dataset, IterableDataset
import json
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
import cv2
import math
import matplotlib.pyplot as plt

def load_pbr_texture_stack(pbr_folder):
    def read_img(fname):
        path = os.path.join(pbr_folder, fname)
        if path.endswith(".exr"):
            exr = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # H × W × C, float32
            # Convert BGR to RGB for OpenCV
            exr = exr[..., ::-1]

            tensor = torch.from_numpy(exr.copy())         # torch.float32
            # Check if it's only two channels and unsqueeze if needed
            if len(tensor.shape) == 2:
                tensor = tensor.unsqueeze(2)  # Add channel dimension if missing
            return tensor.permute(2, 0, 1)                # [C,H,W]
        else:
            img = Image.open(path).convert('RGB')
            tensor = T.ToTensor()(img).float()
            return tensor if tensor.max() <= 1.0 else tensor / 255.0

    # Collect PBR texture data
    try:
        # Read all texture maps based on the image
        col_1 = read_img("fabric_pattern_07_col_1_4k.jpg")  # [3,H,W] - Color map
        ao = read_img("fabric_pattern_07_ao_4k.jpg")        # [3,H,W] - Ambient occlusion
        arm = read_img("fabric_pattern_07_arm_4k.jpg")      # [3,H,W] - (A, Roughness, Metalness)
        rough = read_img("fabric_pattern_07_rough_4k.exr")  # [1,H,W] - Roughness map (EXR)
        nor_dx = read_img("fabric_pattern_07_nor_dx_4k.exr") # [1,H,W] - Normal map X
        nor_gl = read_img("fabric_pattern_07_nor_gl_4k.exr") # [1,H,W] - Normal map GL
        
        
        # Combine all available channels into a texture
        # [Color (3) + AO (3) + ARM (3) + Roughness (1) + Normal DX (1) + Normal GL (1)] = 12 channels
        tex = torch.cat([
            col_1,                # RGB color (3 channels) 0:3
            ao,                   # Ambient occlusion (3 channels) 3:6
            arm,                  # ARM texture (3 channels) 6:9
            rough,                # Roughness map (1 channel) 9:10
            nor_dx,               # Normal map X (3 channel) 10:13
            nor_gl                # Normal map GL (3 channel) 13:16
        ], dim=0)  # [16,H,W]
        
        return tex.permute(1, 2, 0).contiguous()  # [H,W,6] => [U,V,6]
    except Exception as e:
        print(f"Error loading PBR textures: {e}")
        # Return a default texture if loading fails
        H, W = 1024, 1024
        return torch.ones(H, W, 6)

def get_ray_directions(H, W, focal):
    """ get camera ray direction
    Args:
        H,W: height and width
        focal: focal length
    """
    x_coords = torch.linspace(0.5, W - 0.5, W)
    y_coords = torch.linspace(0.5, H - 0.5, H)
    j, i = torch.meshgrid([y_coords, x_coords])
    directions = \
        torch.stack([-(i-W/2)/focal, -(j-H/2)/focal, torch.ones_like(i)], -1) 

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
    c2w = c2w[:3,:4]
    return c2w

class SphereIterableDataset(IterableDataset):
    """ training dataset, return random view and pixel-level rays"""
    def __init__(self, cfg, gt_folder, split):
        self.cfg = cfg
        self.pixel = True
        self.rays_num = cfg.data.rays_num
        self.num_view_batch = cfg.renderer.camera.views_per_batch

        # Load metadata
        self.gt_folder = gt_folder
        self.img_hw = cfg.renderer.resolution
        h, w = self.img_hw
        self.camera_angle_x = cfg.renderer.camera.camera_angle_x
        self.focal = (0.5 * w / np.tan(0.5 * self.camera_angle_x)).item()
        self.directions = get_ray_directions(h, w, self.focal)
        self.number_of_views = cfg.renderer.camera.number_of_views
        self.distance = cfg.renderer.camera.distance
        self.get_camera_dicts()
        self.all_rays, self.all_rgbs = self.preload_rays_and_rgbs()
        # self.pbr_texture = load_pbr_texture_stack(self.cfg.data.pbr_path)

    
    def get_camera_rotation_dicts(self):
        camera_dicts = []
        look_at = self.cfg.renderer.camera.look_at
        up = self.cfg.renderer.camera.up
        dist = self.distance
        n_steps = self.number_of_views
        phi = np.pi
        thetas = np.linspace(0, 2*np.pi, n_steps, endpoint=False)
        for theta in thetas:
            x = dist * np.sin(theta) * np.cos(phi)
            y = dist * np.sin(theta) * np.sin(phi)
            z = dist * np.cos(theta)
            camera_dicts.append({"position": [x, y, z], "look_at": look_at, "up": up})
        return camera_dicts

    def get_camera_dicts(self):
        """Populate self.camera_dict with views on the +Y hemisphere (y > 0)."""
        self.camera_dict = []

        look_at = self.cfg.renderer.camera.look_at
        up      = self.cfg.renderer.camera.up
        dist    = self.distance

        n_phi   = max(4, int(np.sqrt(max(self.number_of_views - 1, 1))))
        n_theta = max(2, int(np.ceil((self.number_of_views - 1) / n_phi)) + 1)

        # θ in (0, π/2)  ⇒  y > 0, exclude equator (θ = π/2) and pole (θ = 0)
        thetas = np.linspace(0.0, np.pi / 2.0, n_theta, endpoint=True)[1:-1]
        phis   = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

        theta_grid, phi_grid = np.meshgrid(thetas, phis, indexing="ij")
        for theta, phi in zip(theta_grid.ravel(), phi_grid.ravel()):
            x = dist * np.sin(theta) * np.cos(phi)
            z = dist * np.sin(theta) * np.sin(phi)
            y = dist * np.cos(theta)             # guaranteed y > 0

            self.camera_dict.append({
                "position": [x, y, z],
                "look_at":  look_at,
                "up":       up
            })

        # Add the single north-pole view (θ = 0, y = +dist)
        self.camera_dict.append({
            "position": [0.0, dist, 0.0],
            "look_at":  look_at,
            "up":       up,
            "phi":      phi
        })

        # Store how many views we actually generated
        self.total = len(self.camera_dict)

    def preload_rays_and_rgbs(self):
        all_rays = []
        all_rgbs = []
        for i, cam in enumerate(self.camera_dict):
            c2w = get_c2w(cam)
            rays_o, rays_d, dxdu, dydv = get_rays(self.directions, c2w, focal=self.focal)
            all_rays.append(torch.cat([rays_o, rays_d, dxdu, dydv], dim=-1))
            if self.gt_folder is not None:
                img = open_exr(os.path.join(self.gt_folder, f'output_view_{i}.exr'), self.img_hw).reshape(-1, 3)
                all_rgbs.append(img)
        return torch.cat(all_rays), torch.cat(all_rgbs) if all_rgbs else None

    def __iter__(self):
        rays_per_view = self.img_hw[0] * self.img_hw[1]

        while True:
            # for idx in torch.randperm(len(self.metadata)):
            view_indices = torch.randint(0, self.total, (self.num_view_batch,))
            total_indices = []
            for view_idx in view_indices:
                base = view_idx.item() * rays_per_view
                rand_idx = torch.randint(base, base + rays_per_view, (self.rays_num // self.num_view_batch,))
                # rand_idx = torch.randint(base, base + rays_per_view, (self.rays_num // self.num_view_batch,))
                total_indices.append(rand_idx)

            ray_idx = torch.cat(total_indices, dim=0)

            rays = self.all_rays[ray_idx]
            if self.all_rgbs is not None:
                rgbs = self.all_rgbs[ray_idx]
            else:
                rgbs = torch.zeros_like(rays[...,:3])

            yield {
                'rays': rays,
                'rgbs': rgbs,
                'gt_params': torch.zeros(1),
            }

class SphereImageDataset(IterableDataset):
    """ training dataset that loads images from metadata, returns sampled rays with emitter IDs"""
    def __init__(self, cfg, gt_folder, split):
        self.cfg = cfg
        self.pixel = True
        self.rays_num = cfg.data.rays_num
        self.num_view_batch = cfg.renderer.camera.views_per_batch
        self.importance_sampling = cfg.data.importance_sampling
        self.use_single_chunk_sampling = cfg.data.use_single_chunk_sampling
        self.multi_resolution = cfg.data.multi_resolution
        self.downsample_iter = cfg.data.downsample_iter
        # Load metadata
        self.gt_folder = gt_folder
        self.img_hw = cfg.renderer.resolution
        self.debug = cfg.data.debug
        h, w = self.img_hw
        self.camera_angle_x = cfg.renderer.camera.camera_angle_x
        self.focal = (0.5 * w / np.tan(0.5 * self.camera_angle_x)).item()
        
        # Load metadata from JSON file
        metadata_path = os.path.join(gt_folder, "metadata.json")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load camera metadata from separate JSON file
        camera_metadata_path = os.path.join(gt_folder, "camera_metadata.json")
        with open(camera_metadata_path, 'r') as f:
            self.camera_metadata = json.load(f)
        
        # Load emitter metadata from separate JSON file
        emitter_metadata_path = os.path.join(gt_folder, "emitter_metadata.json")
        with open(emitter_metadata_path, 'r') as f:
            self.emitter_metadata = json.load(f)
        
        self.total_images = len(self.metadata)
        
        # Split metadata into training and validation sets with fixed random seed
        torch.manual_seed(42)  # Fixed seed for reproducible splits
        indices = torch.randperm(self.total_images)
        
        # Use 80% for training
        split_idx = int(0.8 * self.total_images)
        selected_indices = indices[:split_idx]
        
        # Filter metadata based on split
        self.metadata = [self.metadata[i] for i in selected_indices] # used 10 images for training debug
        if self.debug:
            self.metadata = self.metadata[:10]
        #self.metadata = [item for idx, item in enumerate(self.metadata) if idx % 10 == 0]#temporal modification
        self.set_step(0)
        self.directions = get_ray_directions(h, w, self.focal)
        self.all_rays, self.all_rgbs, self.all_emitter_ids, self.all_pdf = self.preload_rays_and_rgbs(downsample_scale=1)   
        
    def preload_rays_and_rgbs(self, downsample_scale=1):
        all_rays = []
        all_rgbs = []
        all_emitter_ids = []
        
        for img_data in tqdm(self.metadata, desc="Loading images and rays"):
            # Get camera info using camera_id
            camera_id = img_data["camera_id"]
            camera_info = self.camera_metadata[camera_id]
            camera_dict = {
                "position": camera_info["position"],
                "look_at": camera_info["look_at"],
                "up": camera_info["up"]
            }
            
            # Generate rays for this camera
            c2w = get_c2w(camera_dict)
            rays_o, rays_d, dxdu, dydv = get_rays(self.directions, c2w, focal=self.focal)
            rays = torch.cat([rays_o, rays_d, dxdu, dydv], dim=-1)
            # Load original RGB image (without gamma correction)
            original_filename = img_data["filename"].replace(".png", "_original.png")
            img_path = os.path.join(self.gt_folder, original_filename)
            
            if img_path.endswith('.png'):
                # Load PNG image (original linear RGB)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = torch.from_numpy(img).float() / 255.0
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
            all_rays.append(rays)
            all_rgbs.append(img_flat)
            all_emitter_ids.append(emitter_ids)    
        
        # Concatenate all data
        rays = torch.cat(all_rays)
        rgbs = torch.cat(all_rgbs)
        emitter_ids = torch.cat(all_emitter_ids)
        # Calculate luminance for importance sampling
        luminance = (0.2126 * rgbs[..., 0] +
                    0.7152 * rgbs[..., 1] +
                    0.0722 * rgbs[..., 2])         # (N_tot,)
        luminance = torch.abs(luminance)
        pdf = luminance.detach() / luminance.sum()               # p_i  (no grad)

        
        # Randomly permute the data
        perm_indices = torch.randperm(rays.shape[0])
        rays = rays[perm_indices]
        rgbs = rgbs[perm_indices]
        emitter_ids = emitter_ids[perm_indices]
        
        return (rays, rgbs, emitter_ids, pdf)

    def sampler(self, rgbs_gt):
        """Set the importance sampler to use for ray sampling"""
        if self.importance_sampling:
            # --------------------------------------------------------------
            # 5-A  Build luminance-pdf and draw importance samples
            # --------------------------------------------------------------
            # pdf = luminance.detach()
            pdf = self.all_pdf
            if not torch.isfinite(pdf).all():                               # all-black fallback
                pdf = torch.full_like(pdf, 1.0 / pdf.numel())

            N_sample = self.rays_num
            # Use chunked sampling to avoid memory issues with large datasets
            chunk_size = min(15000000, len(pdf))  # Process in chunks of 10M or less
            sample_idx = []
            
            if len(pdf) <= chunk_size:
                # Small enough to sample directly
                sample_idx = torch.multinomial(pdf, N_sample, replacement=True)
            else:
                # Option 1: Sample from all chunks (original behavior)
                # Option 2: Sample from one random chunk only (to reduce cost)
                use_single_chunk = getattr(self, 'use_single_chunk_sampling', False)
                
                if use_single_chunk:
                    # Randomly select one chunk and sample all rays from it
                    num_chunks = (len(pdf) + chunk_size - 1) // chunk_size  # Ceiling division
                    selected_chunk_idx = torch.randint(0, num_chunks, (1,)).item()
                    
                    start_idx = selected_chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, len(pdf))
                    chunk_pdf = pdf[start_idx:end_idx]
                    
                    # Skip if chunk has all zero pdf values
                    if chunk_pdf.sum() == 0:
                        # Fallback to uniform sampling from this chunk
                        chunk_indices = torch.randint(0, len(chunk_pdf), (N_sample,))
                    else:
                        chunk_indices = torch.multinomial(chunk_pdf, N_sample, replacement=True)
                    
                    # Adjust indices to global indexing
                    sample_idx = chunk_indices + start_idx
                else:
                    # Original behavior: sample from all chunks
                    num_chunks = (len(pdf) + chunk_size - 1) // chunk_size  # Ceiling division
                    samples_per_chunk = N_sample // num_chunks
                    remaining_samples = N_sample % num_chunks  # Extra samples for last chunk
                    
                    # Loop over each chunk
                    for chunk_idx in range(num_chunks):
                        start_idx = chunk_idx * chunk_size
                        end_idx = min(start_idx + chunk_size, len(pdf))
                        chunk_pdf = pdf[start_idx:end_idx]
                        
                        # Skip chunks where all pdf values are 0
                        if chunk_pdf.sum() == 0:
                            continue
                        
                        # Determine number of samples for this chunk
                        if chunk_idx == num_chunks - 1:
                            # Last chunk gets remaining samples
                            chunk_samples = samples_per_chunk + remaining_samples
                        else:
                            chunk_samples = samples_per_chunk
                        
                        if chunk_samples > 0:
                            chunk_indices = torch.multinomial(chunk_pdf, chunk_samples, replacement=True)
                            # Adjust indices to global indexing
                            global_indices = chunk_indices + start_idx
                            sample_idx.append(global_indices)
                    
                    sample_idx = torch.cat(sample_idx) if sample_idx else torch.empty(0, dtype=torch.long)
                    
                    # If we still need more samples, fill remaining with uniform sampling
                    if len(sample_idx) < N_sample:
                        remaining = N_sample - len(sample_idx)
                        uniform_indices = torch.randint(0, len(pdf), (remaining,))
                        sample_idx = torch.cat([sample_idx, uniform_indices])
            
            return sample_idx, pdf[sample_idx] * len(pdf)
        else:
            # Uniform random sampling
            N_sample = self.rays_num
            sample_idx = torch.randint(0, len(rgbs_gt), (N_sample,), dtype=torch.long)
            pdf = torch.full((N_sample,), 1.0)
            return sample_idx, pdf

    def set_step(self, step):
        a, b = self.downsample_iter[0], self.downsample_iter[1]
        
        # Determine downsample scale based on thresholds
        if a >= 0 and step < a:
            downsample_scale = 4  # Use scale 4 before threshold a
        elif b >= 0 and step < b:
            downsample_scale = 2  # Use scale 2 before threshold b
        else:
            downsample_scale = 1  # Use scale 1 after both thresholds
        
        # Reload data if downsample scale changed
        if not hasattr(self, '_current_downsample_scale') or self._current_downsample_scale != downsample_scale:
            self._current_downsample_scale = downsample_scale
            # Clear GPU memory if attributes exist
            if hasattr(self, 'directions'):
                del self.directions
            if hasattr(self, 'all_rays'):
                del self.all_rays
            if hasattr(self, 'all_rgbs'):
                del self.all_rgbs
            if hasattr(self, 'all_emitter_ids'):
                del self.all_emitter_ids
            if hasattr(self, 'all_pdf'):
                del self.all_pdf
            # Update directions with proper downsampling
            print(f"Loading rays with downsample scale {downsample_scale}...")
            h, w = self.img_hw
            h_down, w_down = h // downsample_scale, w // downsample_scale
            self.directions = get_ray_directions(h_down, w_down, self.focal) 
            self.all_rays, self.all_rgbs, self.all_emitter_ids, self.all_pdf = self.preload_rays_and_rgbs(downsample_scale=downsample_scale)

    def __iter__(self):
        while True:
            if self.sampler is not None:
            #if False:
                # Use importance sampler
                ray_indices, pdf = self.sampler(self.all_rgbs)
            else:
                # Fallback to uniform sampling
                ray_indices = torch.randint(0, len(self.all_rays), (self.rays_num,))
                pdf = torch.ones(self.rays_num) / len(self.all_rays)

            rays = self.all_rays[ray_indices]
            rgbs = self.all_rgbs[ray_indices]
            emitter_ids = self.all_emitter_ids[ray_indices]

            yield {
                'rays': rays,
                'rgbs': rgbs,
                'emitter_ids': emitter_ids,
                'pdf': pdf,
                'gt_params': torch.zeros(1),
            }


class SphereValDataset(Dataset):
    """ validation dataset that loads images from metadata, returns complete images """
    def __init__(self, cfg, gt_folder):
        self.cfg = cfg
        self.pixel = False
        self.gt_folder = gt_folder
        self.img_hw = cfg.renderer.resolution
        self.debug = cfg.data.debug
        h, w = self.img_hw
        self.camera_angle_x = cfg.renderer.camera.camera_angle_x
        self.focal = (0.5 * w / np.tan(0.5 * self.camera_angle_x)).item()
        self.directions = get_ray_directions(h, w, self.focal)
        
        # Load metadata from JSON file
        metadata_path = os.path.join(gt_folder, "metadata.json")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load camera metadata from separate JSON file
        camera_metadata_path = os.path.join(gt_folder, "camera_metadata.json")
        with open(camera_metadata_path, 'r') as f:
            self.camera_metadata = json.load(f)
        
        # Load emitter metadata from separate JSON file
        emitter_metadata_path = os.path.join(gt_folder, "emitter_metadata.json")
        with open(emitter_metadata_path, 'r') as f:
            self.emitter_metadata = json.load(f)
        
        self.total_images = len(self.metadata)
        
        # Split metadata into training and validation sets with fixed random seed
        torch.manual_seed(42)  # Fixed seed for reproducible splits
        indices = torch.randperm(self.total_images)
        
        # Use 20% for validation
        split_idx = int(0.8 * self.total_images)
        selected_indices = indices[split_idx:]
        
        # Filter metadata based on split
        self.metadata = [self.metadata[i] for i in selected_indices]
        if self.debug:
            self.metadata = self.metadata[:2]
        #self.metadata = [item for idx, item in enumerate(self.metadata) if idx % 10 == 0]#temporal modification

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_data = self.metadata[idx]
        
        # Get camera info using camera_id
        camera_id = img_data["camera_id"]
        # camera_id = 0 # debug with the first camera
        camera_info = self.camera_metadata[camera_id]
        camera_dict = {
            "position": camera_info["position"],
            "look_at": camera_info["look_at"],
            "up": camera_info["up"]
        }
        
        # Generate rays for this camera
        c2w = get_c2w(camera_dict)
        rays_o, rays_d, dxdu, dydv = get_rays(self.directions, c2w, focal=self.focal)
        rays = torch.cat([rays_o, rays_d, dxdu, dydv], dim=-1)
        
        # Load original RGB image (without gamma correction)
        original_filename = img_data["filename"].replace(".png", "_original.png")
        img_path = os.path.join(self.gt_folder, original_filename)
        
        if img_path.endswith('.png'):
            # Load PNG image (original linear RGB)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).float() / 255.0
        elif img_path.endswith('.exr'):
            # Load EXR image (already linear)
            img = open_exr(img_path, self.img_hw)
        else:
            raise ValueError(f"Unsupported image format: {img_path}")
        
        img_flat = img.reshape(-1, 3)
        
        # Get emitter ID directly from metadata
        emitter_ids = torch.full((rays.shape[0],), img_data["emitter_id"], dtype=torch.long)
        return {
            'rays': rays,
            'rgbs': img_flat,
            'emitter_ids': emitter_ids,
            'gt_params': torch.zeros(1),
        }
        
class SphereTestDataset(Dataset):
    """  test dataset, return all views """
    def __init__(self, cfg, gt_folder, split):
        self.cfg = cfg
        self.gt_folder = gt_folder
        self.img_hw = cfg.renderer.resolution
        h, w = self.img_hw
        self.number_of_views = cfg.renderer.camera.number_of_view_test
        self.distance = cfg.renderer.camera.distance
        self.camera_angle_x = cfg.renderer.camera.camera_angle_x
        self.focal = (0.5 * w / np.tan(0.5 * self.camera_angle_x)).item()
        self.directions = get_ray_directions(h, w, self.focal)
        self.get_camera_rotation_dicts()
        self.metadata = self._load_metadata(f"metadata/dual_{split}.txt")

    def _load_metadata(self, path):
        metadata = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    values = list(map(float, line.strip().split()))
                    if len(values) == 2:
                        r, m = values
                        metadata.append((r, m))
                    elif len(values) == 4:  # Support for dual parameter sets
                        r1, m1, r2, m2 = values
                        metadata.append((r1, m1, r2, m2))
        return metadata

    def get_camera_rotation_dicts(self):
        # Initialize camera dicts list  
        self.camera_dict = []
        
        look_at = self.cfg.renderer.camera.look_at
        up = self.cfg.renderer.camera.up
        dist = self.distance
        phi = np.pi
        n_steps = self.number_of_views  # Can be adjusted for more/fewer views
        self.total = n_steps
        thetas = np.linspace(0, 2*np.pi, n_steps, endpoint=False)
        for theta in thetas:
            x = dist * np.sin(theta) * np.cos(phi)  # Using cos(0)=1 for fixed phi
            y = dist * np.sin(theta) * np.sin(phi)  # Using sin(0)=0 for fixed phi
            z = dist * np.cos(theta)
            
            # Create camera dict for this position
            camera_dict = {
                "position": [x, y, z],
                "look_at": look_at,
                "up": up
            }
            
            self.camera_dict.append(camera_dict)
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        params = self.metadata[idx]
        if len(params) == 2:
            r, m = params
            gt_params = {'roughness': torch.tensor(r, dtype=torch.float32),
                         'metallic': torch.tensor(m, dtype=torch.float32)}
        else:  # Handle dual parameter case
            r1, m1, r2, m2 = params
            gt_params = {'roughness': torch.tensor([r1, r2], dtype=torch.float32),
                         'metallic': torch.tensor([m1, m2], dtype=torch.float32)}
            
        all_views = []
        for view_idx, cam_dict in enumerate(self.camera_dict):
            c2w = get_c2w(cam_dict)
            rays_o, rays_d, dxdu, dydv = get_rays(self.directions, c2w, focal=self.focal)
            rays = torch.cat([rays_o, rays_d, dxdu, dydv], dim=-1)
            if self.gt_folder:
                img_path = os.path.join(self.gt_folder, f'output_{idx}_view_{view_idx}.exr')
                if os.path.exists(img_path):
                    img = open_exr(img_path, self.img_hw).reshape(-1, 3)
                else:
                    img = torch.zeros_like(rays[..., :3])
            else:
                img = torch.zeros_like(rays[..., :3])
            
            view_data = {
                'rays': rays,
                'rgbs': img,
                'gt_params': gt_params
            }
            all_views.append(view_data)
        
        return all_views
