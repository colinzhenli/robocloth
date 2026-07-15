import torch
import torch.nn as nn
import torch.nn.functional as NF
import numpy as np
import math
import imageio
import cv2
from openexr_numpy import imread, imwrite
import json
import os
from utils.io import read_light_transforms

class EnvMapEmitter(nn.Module):
    """ Environment map emitter using HDRI """
    def __init__(self, envmap_path):
        """
        Args:
            envmap_path: Path to the .exr or .hdr environment map
        """
        super(EnvMapEmitter, self).__init__()
        
        # Load environment map (assumed to be in lat-long format)
        envmap = imageio.imread(envmap_path).astype('float32')[:,:,:3]  # Shape: (H, W, 3)
        envmap = torch.from_numpy(envmap).permute(2, 0, 1)  # Convert to (3, H, W)
        
        self.register_buffer('envmap', envmap)
        self.H, self.W = envmap.shape[1:]  # Get resolution

    def sample_emitter(self, sample, position):
        """
        Sample a direction from the environment map
        Args:
            sample: Bx2 uniform samples for spherical sampling
            position: Bx3 surface positions (unused)
        Returns:
            wi: Bx3 sampled directions
            pdf: Bx1 sampling pdf
            idx: B dummy indices (-1)
        """
        # Convert uniform samples to spherical coordinates
        phi = 2 * math.pi * sample[..., 0]  # Azimuth
        theta = torch.acos(1 - 2 * sample[..., 1])  # Elevation
        
        sin_theta = torch.sin(theta)
        x = sin_theta * torch.cos(phi)
        y = sin_theta * torch.sin(phi)
        z = torch.cos(theta)
        
        wi = torch.stack([x, y, z], dim=-1)  # Direction vectors
        
        # Compute PDF (uniform for now)
        pdf = torch.full((position.shape[0], 1), 1.0 / (4 * math.pi), device=position.device)
        
        idx = torch.full((position.shape[0],), -1, dtype=torch.long, device=position.device)
        
        return wi, pdf, idx

    def eval_emitter(self, position, light_dir):
        """
        Evaluate environment map radiance along given directions
        Args:
            position: Bx3 intersection points (unused)
            light_dir: Bx3 light directions
        Returns:
            Le: Bx3 radiance
            pdf: Bx1 pdf
            valid: B valid samples (always True for envmap)
        """
        # Convert direction to lat-long coordinates
        phi = torch.atan2(light_dir[..., 2], light_dir[..., 0])  # [-π, π]
        theta = torch.asin(-light_dir[..., 1])  # [-π/2, π/2]

        # Normalize to [0, 1] texture coordinates
        u = (phi / (2 * math.pi)) + 0.5
        v = theta / math.pi + 0.5

        # Convert to pixel indices
        u_idx = (u * (self.W - 1)).long().clamp(0, self.W - 1)
        v_idx = (v * (self.H - 1)).long().clamp(0, self.H - 1)

        # Sample radiance from environment map
        Le = self.envmap[:, v_idx, u_idx].permute(1, 0)  # (B, 3)

        # Compute PDF (assuming uniform distribution for now)
        pdf = torch.full((position.shape[0], 1), 1.0 / (4 * math.pi), device=position.device)

        return Le, pdf, torch.ones_like(pdf, dtype=torch.bool)  # Always valid

    
class DynamicPointEmitter(nn.Module):
    def __init__(self, ray_num, dist=4.0, num_lights=8, camera_phi=None, theta_angle=60.0, random_positions=True, random_intensities=False, different_per_point=False):
        """
        Args:
            dist: Radius of the sphere
            num_lights: Number of lights to sample
            fix_seed: Whether to fix the seed
        """
        super(DynamicPointEmitter, self).__init__()

        self.dist = dist
        self.num_lights = num_lights
        self.camera_phi = camera_phi
        self.theta_angle = theta_angle
        self.random_positions = random_positions
        self.ray_num = ray_num
        self.random_intensities = random_intensities
        self.different_per_point = different_per_point
        if self.different_per_point:
            theta = torch.pi/2 * torch.rand(ray_num, num_lights, device="cuda")
            phi = 2 * torch.pi * torch.rand(ray_num, num_lights, device="cuda")
            x = dist * torch.sin(theta) * torch.cos(phi)
            z = dist * torch.sin(theta) * torch.sin(phi)
            y = dist * torch.cos(theta)
            light_positions = torch.stack([x, y, z], dim=-1)
            if self.random_intensities:
                light_intensities = torch.rand(num_lights, 1, device="cuda") * 49.0 + 1.0  # Uniform [1.0, 50.0]
            else:
                light_intensities = torch.full((num_lights, 1), 50.0/num_lights, device="cuda")  # Fixed maximum intensity
            self.register_buffer('light_positions', light_positions)  # [B, N, 3]
            self.register_buffer('light_intensities', light_intensities)  # [B, N, 1]
            return

        if self.camera_phi is None:
            self.fixed_theta = False
        else:
            self.fixed_theta = True
        if self.random_positions:
            # randomly sample during training
            theta = torch.pi/2 * torch.rand(num_lights, device="cuda")
            phi   = 2 * torch.pi * torch.rand(num_lights, device="cuda")
        else:
            if self.fixed_theta:
                theta = np.deg2rad(theta_angle)
                theta = torch.full((num_lights,), theta, device='cuda')  # Fixed theta for all lights
                
                if num_lights == 1:
                    phi = torch.tensor([torch.pi], device='cuda')
                    phi = phi + camera_phi
                else:
                    # Uniformly distribute phi values for multiple lights
                    phi = torch.linspace(0, 2 * torch.pi, num_lights, endpoint=False, device='cuda')
                    phi = phi + camera_phi
            else:
                torch.manual_seed(0)
                # θ ∼ uniform on (0, π/2) to ensure y > 0 (cos(θ) > 0)
                theta = torch.pi/2 * torch.rand(num_lights, device="cuda")
                # φ uniform on (0, 2π)
                phi   = 2 * torch.pi * torch.rand(num_lights, device="cuda")

        x = dist * torch.sin(theta) * torch.cos(phi)
        z = dist * torch.sin(theta) * torch.sin(phi)
        y = dist * torch.cos(theta)   

        positions = torch.stack([x, y, z], dim=1)  # [N, 3]
        if self.random_intensities:
            intensities = torch.rand(num_lights, 1, device='cuda') * 49.0 + 1.0  # Uniform [1.0, 50.0]
        else:
            intensities = torch.full((num_lights, 1), 50.0/num_lights, device='cuda')  # Fixed maximum intensity

        self.register_buffer('light_positions', positions)  # [N, 3]
        self.register_buffer('light_intensities', intensities)  # [N, 1]

    def sample_emitter(self, position):
        """
        Deterministic sampling: For each position, sample toward every light.

        Args:
            sample1: (unused)
            sample2: (unused)
            position: (B, 3) surface positions

        Returns:
            wi: (B, N, 3) directions toward lights
            pdf: (B, N, 1) uniform pdf
            light_pos: (B, N, 3) selected light positions
            idx: (B, N) selected light indices
        """
        B = position.shape[0]
        if self.different_per_point:
            N = self.light_positions.shape[1]
        else:
            N = self.light_positions.shape[0]

        position_expand = position.unsqueeze(1).expand(B, N, 3)  # [B, N, 3]
        if self.different_per_point:
            # Select the first B positions from self.light_positions for per-point emitters
            light_pos_expand = self.light_positions[:B]  # [B, N, 3]
        else:
            light_pos_expand = self.light_positions.unsqueeze(0).expand(B, N, 3)  # [B, N, 3]

        vec = light_pos_expand - position_expand  # [B, N, 3]
        wi = NF.normalize(vec, dim=-1).reshape(-1, 3)  # [B*N, 3]

        pdf = torch.full((B*N, 1), 1.0 / N, device=position.device)

        idx = torch.arange(N, device=position.device).unsqueeze(0).expand(B, N).reshape(-1)  # [B*N]

        light_pos = light_pos_expand.reshape(-1, 3)

        return wi, pdf, light_pos, idx

    def eval_emitter(self, position, idx):
        """
        Evaluate radiance from selected lights.

        Args:
            position: (B, 3) surface points
            idx: (B,) selected light indices

        Returns:
            Le: (B, 3) radiance
            pdf: (B, 1) pdf
            valid: (B,) valid mask
        """
        B = position.shape[0]
        intensities = self.light_intensities.expand(-1, 3)  # [B, 3]
        pdf = torch.full((B, 1), 1.0 / self.light_positions.shape[0], device=position.device)
        valid = torch.ones(B, dtype=torch.bool, device=position.device)

        return intensities, pdf, valid
    
class PresetPointEmitter(nn.Module):
    def __init__(self, read_from_metadata=False, metadata_path=None, positions=None, intensities=None):
        """
        Args:
            positions: (N, 3) tensor of light positions
            intensities: (N, 1) tensor of light intensities
        """
        super(PresetPointEmitter, self).__init__()
        if read_from_metadata:
            # Read positions and intensities from metadata
            import json
            import os
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                # Extract emitter metadata
                positions = torch.tensor([em['position'] for em in metadata], device='cuda')
                intensities = torch.tensor([em['intensity'] for em in metadata], device='cuda')
                self.positions = positions
                self.intensities = intensities
            else:
                raise FileNotFoundError(f"Metadata file not found at {metadata_path}")  
        else:
            self.positions = positions
            self.intensities = intensities

        self.register_buffer('light_positions', positions)  # [N, 3]
        self.register_buffer('light_intensities', intensities)  # [N, 3]

    def sample_emitter(self, position, idx):
        """
        Deterministic sampling: For each position, sample toward one specific light using idx.

        Args:
            position: (B, 3) surface positions
            idx: (B,) selected light indices
        Returns:
            wi: (B, 3) directions toward selected lights
            pdf: (B, 1) uniform pdf
            light_pos: (B, 3) selected light positions
            idx: (B,) selected light indices
        """
        B = position.shape[0]
        
        # Select specific light positions based on idx
        light_pos = self.light_positions[idx]  # [B, 3]
        
        vec = light_pos - position  # [B, 3]
        wi = NF.normalize(vec, dim=-1)  # [B, 3]
        
        pdf = torch.full((B, 1), 1.0, device=position.device)
        
        return wi, pdf, light_pos, idx

    def eval_emitter(self, position, idx):
        """
        Evaluate radiance from selected lights.

        Args:
            position: (B, 3) surface points
            idx: (B,) selected light indices

        Returns:
            Le: (B, 3) radiance
            pdf: (B, 1) pdf
            valid: (B,) valid mask
        """
        B = position.shape[0]
        # Select intensities based on idx for each position
        intensities = self.light_intensities[idx]  #  [B, 3]
        pdf = torch.full((B, 1), 1.0 / self.light_positions.shape[0], device=position.device)
        valid = torch.ones(B, dtype=torch.bool, device=position.device)

        return intensities, pdf, valid
    
class RealAreaEmitter(nn.Module):
    def __init__(self, cfg, json_path):
        """
        Args:
            positions: (N, 3) light center
            radius: (N, 1) light radius
            radiance: (N, 1) largest radiance
        """
        super(RealAreaEmitter, self).__init__()
        # Extract configuration parameters
        radius = cfg.get('radius', 0.007)
        fwhm_deg = cfg.get('fwhm_deg', 115.0)
        # self.light_radiance = nn.Parameter(torch.tensor(cfg.get('radiance'), dtype=torch.float32, device='cuda'))
        self.register_buffer('light_radiance', torch.tensor(cfg.get('radiance'), dtype=torch.float32, device='cuda'))
        print(f"RealAreaEmitter: Light radiance: {cfg.get('radiance')}")

        theta_half = math.radians(fwhm_deg * 0.5)
        m = math.log(0.5) / math.log(max(1e-8, math.cos(theta_half)))
        self.register_buffer('m', torch.tensor(m, dtype=torch.float32, device='cuda'))
        self.register_buffer('light_radius', torch.tensor(radius, dtype=torch.float32, device='cuda'))

        self.l2w = self._compute_l2w(cfg, json_path)
        self.register_buffer('light_positions', self.l2w[:, :3, 3])  # [N, 3] - translation part
        # Add a small tilt along z-axis
        tilt_angle = math.radians(0)  # 5 degree tilt, adjust as needed
        light_normal_local = torch.tensor([0.0, -math.cos(tilt_angle), math.sin(tilt_angle)], dtype=torch.float32, device='cuda')
        light_normals_world = torch.matmul(self.l2w[:, :3, :3], light_normal_local)  # [N, 3]
        light_normals_world = light_normals_world / (light_normals_world.norm(dim=-1, keepdim=True) + 1e-12)
        self.register_buffer('light_normal', light_normals_world)
        
        # Load calibrated directional distribution if provided
        self.calibrated_directional_distribution = False
        direction_json = cfg.get('direction_json', '')
        if direction_json:
            self._load_calibration_table(direction_json)
    
    def _load_calibration_table(self, direction_json):
        """
        Load the directional distribution calibration table from JSON file.
        Args:
            direction_json: path to the JSON file containing the calibration table
        """
        import json
        
        with open(direction_json, 'r') as f:
            calibration_data = json.load(f)
        
        # Extract table data
        resolution = calibration_data['resolution_degrees']
        data_dict = calibration_data['data']
        max_cam_rad_ratio = calibration_data['max_cam_rad_ratio']
        
        # Convert dictionary to sorted arrays for interpolation
        angles = []
        ratios = []
        for angle_str, ratio in sorted(data_dict.items(), key=lambda x: float(x[0])):
            angles.append(float(angle_str))
            ratios.append(ratio)
        
        # Store as tensors
        self.register_buffer('calibration_angles', torch.tensor(angles, dtype=torch.float32, device='cuda'))
        self.register_buffer('calibration_ratios', torch.tensor(ratios, dtype=torch.float32, device='cuda'))
        self.register_buffer('calibration_resolution', torch.tensor(resolution, dtype=torch.float32, device='cuda'))
        self.register_buffer('max_cam_rad_ratio', torch.tensor(max_cam_rad_ratio, dtype=torch.float32, device='cuda'))
        
        self.calibrated_directional_distribution = True
        print(f"Loaded directional calibration table from {direction_json}")
        print(f"  Resolution: {resolution}°, Angles: {len(angles)}, Max ratio: {max_cam_rad_ratio:.6f}")
        
    def _compute_l2w(self, cfg, json_path):
        """
        Compute the world transform for each light.
        """
        # Compute light transformation matrix
        R_l2g = torch.tensor(cfg.get('R_l2g'), dtype=torch.float32, device='cuda')
        t_l2g = torch.tensor(cfg.get('t_l2g'), dtype=torch.float32, device='cuda')
        base2_to_base1 = torch.tensor(cfg.get('base2_to_base1'), dtype=torch.float32, device='cuda')
        g2b0 = read_light_transforms(json_path, cfg.turntable.center, cfg.turntable.axis, base2_to_base1) # [N, 4, 4]
        # g2b0 = base2_to_base1.unsqueeze(0) @ g2b
        l2g = torch.eye(4, device='cuda')
        l2g[:3, :3] = R_l2g
        l2g[:3, 3] = t_l2g
        l2w = g2b0 @ l2g
        return l2w
    
    def _update_poses_for_vis(self, turntable_center, steps):
        turntable_center = turntable_center.to(self.l2w.device)
        light_normal_local = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32, device='cuda')
        p0_world = self.l2w[:3, 3]   # [3]
        R0_world = self.l2w[:3, :3]  # [3,3]
        n0_world = (R0_world @ light_normal_local)  # [3]
        n0_world = n0_world / (n0_world.norm() + 1e-12)

        # Clockwise in right-handed (+Z out) means negative angles
        angles = torch.arange(steps, device='cuda', dtype=torch.float32) * (-2.0 * math.pi / steps)  # [60]

        c = torch.cos(angles)
        s = torch.sin(angles)
        # Batch of Rz(θ): shape [60, 3, 3]
        Rz = torch.zeros(steps, 3, 3, device='cuda', dtype=torch.float32)
        Rz[:, 0, 0] =  c
        Rz[:, 0, 1] = -s
        Rz[:, 1, 0] =  s
        Rz[:, 1, 1] =  c
        Rz[:, 2, 2] =  1.0

        # Rotate the position around the center: p' = Rz*(p0 - center) + center
        rel = p0_world - turntable_center  # [3]
        rel = rel.unsqueeze(-1)            # [3,1] for batch matmul
        pos_rot = (Rz @ rel).squeeze(-1) + turntable_center  # [60,3]

        # Rotate the normal as a direction: n' = Rz * n0
        n0 = n0_world.unsqueeze(-1)  # [3,1]
        nor_rot = (Rz @ n0).squeeze(-1)  # [60,3]
        nor_rot = nor_rot / (nor_rot.norm(dim=-1, keepdim=True) + 1e-12)

        # ---- Register buffers ----
        with torch.no_grad():
            self.light_positions = pos_rot
            self.light_normal = nor_rot

                
    def _directional_distribution(self, light_dir, light_id):
        """
        Compute directional radiance L(θ) for rays headed from the light to the surface.
        Args:
            light_dir: (B, 3) directions from surface -> light (so emission dir is -light_dir)
            light_id: ID of the light

        Returns:
            radiance: (B, 3) directional radiance (RGB) following L = L0 * cos^m(theta).
                      Clamped to zero for back-facing directions.
        """
        # Emission direction is from light -> surface
        v = -light_dir  # (B,3)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-12)

        # cos(theta) between light normal and emission direction
        cos_theta = torch.clamp((v * self.light_normal[light_id]).sum(dim=-1, keepdim=True), -1, 1)

        if self.calibrated_directional_distribution:
            # Use calibrated table with linear interpolation
            # Convert cos(theta) to degrees
            theta_rad = torch.acos(cos_theta)  # (B, 1)
            theta_deg = theta_rad * 180.0 / math.pi  # (B, 1)
            
            # Linear interpolation in the calibration table
            theta_deg_flat = theta_deg.squeeze(-1)  # (B,)
            
            # Clamp to valid range [0, 90]
            theta_deg_flat = torch.clamp(theta_deg_flat, 0.0, self.calibration_angles[-1])

            # Use searchsorted to find indices for interpolation
            indices = torch.searchsorted(self.calibration_angles, theta_deg_flat, right=False)
            indices = torch.clamp(indices, 1, len(self.calibration_angles) - 1)
            
            # Get lower and upper bounds for linear interpolation
            lower_idx = indices - 1
            upper_idx = indices
            
            lower_angle = self.calibration_angles[lower_idx]  # (B,)
            upper_angle = self.calibration_angles[upper_idx]  # (B,)
            lower_ratio = self.calibration_ratios[lower_idx]  # (B,)
            upper_ratio = self.calibration_ratios[upper_idx]  # (B,)
            
            # Linear interpolation weight
            weight = (theta_deg_flat - lower_angle) / (upper_angle - lower_angle + 1e-8)
            weight = torch.clamp(weight, 0.0, 1.0)
            
            interpolated_ratio = lower_ratio + weight * (upper_ratio - lower_ratio)  # (B,)
            
            # The calibration table stores relative ratios (normalized to max)
            Lshape = interpolated_ratio.unsqueeze(-1)  # (B, 1)
            L0 = self.light_radiance  # (3,) or (1, 3)
            radiance = Lshape * L0  # (B, 3)
        else:
            # Original cosine-power lobe model
            Lshape = cos_theta.pow(self.m)  # (B,1)
        L0 = self.light_radiance     # (B, 3)
        radiance = Lshape * L0          # (B,3)
        
        return radiance
    
    def sample_emitter(self, sample, position, light_id):
        """
        Sample a direction(position) from the area light (circular disk).
        Args:
            sample: Bx2 uniform samples for disk sampling
            position: Bx3 surface positions
            light_id: B light indices
        Returns:
            wi: Bx3 sampled directions
            pdf: Bx1 sampling pdf (area pdf)
            emit_position: Bx3 sampled positions
            emitter_normal: Bx3 sampled normals
        """
        B = position.shape[0]
        
        # Get light properties for the specified light indices
        light_pos = self.light_positions[light_id]  # (B, 3)
        light_r = self.light_radius.expand(B)  # (B,)
        light_n = self.light_normal[light_id]  # (B, 3)
        
        # Uniform sampling on disk using polar coordinates
        r_sample = torch.sqrt(sample[..., 0]) * light_r   # (B,)
        theta = 2.0 * math.pi * sample[..., 1]           # (B,)
        disk_x = r_sample * torch.cos(theta)             # (B,)
        disk_y = r_sample * torch.sin(theta)             # (B,)
        
        # Create orthonormal basis for each light plane
        up = torch.tensor([0.0, 1.0, 0.0], device=light_n.device, dtype=light_n.dtype)
        up = up.unsqueeze(0).expand(B, 3)  # (B, 3)
        
        # Check for near-parallel cases and use alternative up vector
        parallel_mask = torch.abs((light_n * up).sum(dim=-1)) > 0.9  # (B,)
        alt_up = torch.tensor([1.0, 0.0, 0.0], device=light_n.device, dtype=light_n.dtype)
        alt_up = alt_up.unsqueeze(0).expand(B, 3)  # (B, 3)
        up = torch.where(parallel_mask.unsqueeze(-1), alt_up, up)  # (B, 3)
        
        # Compute u_axis for each light
        u_axis = torch.cross(light_n, up, dim=-1)  # (B, 3)
        u_axis = u_axis / (torch.norm(u_axis, dim=-1, keepdim=True) + 1e-8)  # (B, 3)
        
        # Compute v_axis for each light
        v_axis = torch.cross(light_n, u_axis, dim=-1)  # (B, 3)
        v_axis = v_axis / (torch.norm(v_axis, dim=-1, keepdim=True) + 1e-8)  # (B, 3)
        
        # Compute sampled position on light disk
        emit_position = (
            light_pos 
            + disk_x.unsqueeze(-1) * u_axis 
            + disk_y.unsqueeze(-1) * v_axis
        )  # (B, 3)
        
        # Calculate direction from surface to light sample
        wi = emit_position - position  # (B, 3)
        distance = torch.norm(wi, dim=-1, keepdim=True)  # (B, 1)
        wi = wi / distance  # (B, 3)
        
        # Calculate area-based PDF
        # PDF = 1 / Area = 1 / (π r^2)
        light_area = math.pi * light_r * light_r  # (B,)
        pdf = 1.0 / light_area  # (B,)
        pdf = pdf.unsqueeze(-1)  # (B, 1)
        
        # Emitter normal (same for all sampled points)
        emitter_normal = light_n  # (B, 3)
        
        return wi, pdf, emit_position, emitter_normal

    def intersect(self, position, light_dir, light_id):
        """
        Intersect a ray with the area light (circular disk)
        Args:
            position: (B, 3) ray origins
            light_dir: (B, 3) ray directions (normalized)
            light_id: (B,) light indices
        Returns:
            t: (B,) intersection distances (negative if no intersection)
            hit: (B,) boolean mask for valid intersections
            hit_pos: (B, 3) intersection positions
            light_idx: (B,) light indices
        """
        B = position.shape[0]
        
        # Get light properties for the specified light indices
        light_pos = self.light_positions[light_id]  # (B, 3)
        light_r = self.light_radius.expand(B)  # (B,)
        light_n = self.light_normal[light_id]  # (B, 3)
        
        # Ray-plane intersection
        # Ray: p(t) = pos + t * dirs
        # Plane: (p - light_pos) · light_normal = 0
        # Substituting: (pos + t*dirs - light_pos) · light_normal = 0
        # Solving for t: t = (light_pos - pos) · light_normal / (dirs · light_normal)
        
        # Compute denominator (ray direction dot plane normal)
        denom = (light_dir * light_n).sum(dim=-1)  # (B,)

        # Check if ray is parallel to plane (denom ≈ 0)
        
        # Compute numerator
        to_light = light_pos - position  # (B, 3)
        numer = (to_light * light_n).sum(dim=-1)  # (B,)
        '''
        print("position",position)
        print("light_pos",light_pos)
        print("to_light",to_light)
        print("light_n",light_n)
        
        print("numer",numer)
        print("denom",denom)
        '''
        # Compute intersection distance
        t = numer / denom  # (B,)
        
        # Check if intersection is in front of ray origin
        
        # Compute intersection points
        hit_pos = position + t.unsqueeze(-1) * light_dir  # (B, 3)
        
        # Check if intersection point is within the circular disk
        # Create orthonormal basis for each light plane
        up = torch.tensor([0.0, 1.0, 0.0], device=light_n.device, dtype=light_n.dtype)
        up = up.unsqueeze(0).expand(B, 3)  # (B, 3)
        
        # Check for near-parallel cases and use alternative up vector
        parallel_mask = torch.abs((light_n * up).sum(dim=-1)) > 0.9  # (B,)
        alt_up = torch.tensor([1.0, 0.0, 0.0], device=light_n.device, dtype=light_n.dtype)
        alt_up = alt_up.unsqueeze(0).expand(B, 3)  # (B, 3)
        up = torch.where(parallel_mask.unsqueeze(-1), alt_up, up)  # (B, 3)
        
        # Compute u_axis for each light
        u_axis = torch.cross(light_n, up, dim=-1)  # (B, 3)
        u_axis = u_axis / (torch.norm(u_axis, dim=-1, keepdim=True) + 1e-8)  # (B, 3)
        
        # Compute v_axis for each light
        v_axis = torch.cross(light_n, u_axis, dim=-1)  # (B, 3)
        v_axis = v_axis / (torch.norm(v_axis, dim=-1, keepdim=True) + 1e-8)  # (B, 3)
        
        # Project intersection point onto the light plane coordinate system
        to_hit = hit_pos - light_pos  # (B, 3)
        u_coord = (to_hit * u_axis).sum(dim=-1)  # (B,)
        v_coord = (to_hit * v_axis).sum(dim=-1)  # (B,)
        
        # Check if within circular disk bounds (distance from center <= radius)
        dist_from_center = torch.sqrt(u_coord * u_coord + v_coord * v_coord)  # (B,)
        
        # Final hit mask: valid t, not parallel, and within disk
        
        # Set invalid distances to negative
        #t = torch.where(hit, t, torch.full_like(t, -1.0))
        hit=True
        
        # Light indices for each intersection
        light_idx = light_id  # (B,)
        
        return t, hit, hit_pos, light_idx
    
    def eval_emitter(self, position, light_dir, light_id):
        """
        Evaluate environment map radiance along given directions
        Args:
            position: Bx3 intersection points 
            light_dir: Bx3 light directions(from surface to light)
        Returns:
            Le: Bx3 radiance
            pdf: Bx1 pdf
            valid: B valid samples (always True for area light)
        """
        t, hit, hit_pos, light_idx=self.intersect(position,light_dir,light_id)
        '''
        print("t",t.shape)
        print("hit_pos",hit_pos.shape)
        print("light_dir",light_dir.shape)
        print("light_normal",self.light_normal[light_id].shape)
        '''
        B = position.shape[0]

        Le=self._directional_distribution(light_dir,light_id)
        # dA_dw=((position-hit_pos)*(position-hit_pos)).sum(dim=-1)/((-light_dir)*self.light_normal[light_id]).sum(dim=-1)
        pdf=1.0/(self.light_radius.expand(B)*self.light_radius.expand(B)*torch.pi)
        pdf=pdf.unsqueeze(-1)

        if torch.isnan(Le).any():
            print("Le is nan")
        return Le, pdf, torch.ones_like(pdf, dtype=torch.bool)  # Always valid
    
class MultiAreaEmitter(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            cfg: configuration object
            folder_path: root folder containing material subfolders (0, 1, 2...)
        """
        super(MultiAreaEmitter, self).__init__()
        from pathlib import Path
        
        # Extract configuration parameters
        radius = cfg.get('radius', 0.007)
        fwhm_deg = cfg.get('fwhm_deg', 115.0)
        # self.light_radiance = nn.Parameter(torch.tensor(cfg.get('radiance'), dtype=torch.float32, device='cuda'))
        self.register_buffer('light_radiance', torch.tensor(cfg.get('radiance'), dtype=torch.float32, device='cuda'))

        theta_half = math.radians(fwhm_deg * 0.5)
        m = math.log(0.5) / math.log(max(1e-8, math.cos(theta_half)))
        self.register_buffer('m', torch.tensor(m, dtype=torch.float32, device='cuda'))
        self.register_buffer('light_radius', torch.tensor(radius, dtype=torch.float32, device='cuda'))

        # Read training list from txt file and load scan_log.json for each
        folder_path = cfg.get('folder_path', '')
        root = Path(folder_path)
        training_list_path = cfg.training_list_path
        training_list = []
        with open(training_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    training_list.append(int(line))

        # Mirror MultiMaterialDenseDataset / _load_point_metadata: when continue-
        # training from a pre-swap checkpoint, read scan_log.json from the swap
        # partner's folder so emitter buffer shapes line up with the dataset and
        # latent_bank for any swapped slot that lands in the training list.
        # Driven by data.legacy_swap_indexing via the renderer/multiarea_emitter
        # yaml interpolation (cfg.legacy_swap_indexing = ${data.legacy_swap_indexing}).
        swap_partner = {}
        legacy_swap_indexing = bool(cfg.get('legacy_swap_indexing', False))
        if legacy_swap_indexing:
            rl_path = cfg.get('replace_list_path', None) or str(root / 'replace_list.json')
            if os.path.exists(rl_path):
                with open(rl_path) as _rl_f:
                    _rl = json.load(_rl_f)
                for _r in _rl.get('records', []):
                    if _r.get('backup_id') is None or _r.get('replaces') is None:
                        continue
                    swap_partner[int(_r['backup_id'])] = int(_r['replaces'])
                    swap_partner[int(_r['replaces'])] = int(_r['backup_id'])
                _affected = sum(1 for m in training_list if m in swap_partner)
                print(f"MultiAreaEmitter [legacy_swap_indexing] enabled: {_affected}/{len(training_list)} training-list slots remap to partner folder ({rl_path})")
            else:
                print(f"MultiAreaEmitter [legacy_swap_indexing] enabled but replace_list.json not found at {rl_path}; no remap applied")

        # mat_ids stays as the training_list slot ids (used by mat_id_to_idx for
        # downstream lookup); only the folder we read scan_log.json from changes.
        material_folders = [root / str(swap_partner.get(mid, mid)) for mid in training_list]
        mat_ids = training_list
        print(f"MultiAreaEmitter: Training list path: {training_list_path}")
        print(f"MultiAreaEmitter: Loaded {len(training_list)} materials: {training_list}")

        l2w_list = []
        for d in material_folders:
            json_path = d / "scan_log.json"
            
            # Compute l2w for this material [N_i, 4, 4]
            l2w_mat = self._compute_l2w(cfg, str(json_path)) 
            l2w_list.append(l2w_mat)

        if not l2w_list:
             raise ValueError(f"No valid material folders found in {folder_path}")

        # Build mapping from material_id -> list index (to handle non-contiguous IDs like 0,1,3,5)
        mat_id_to_idx = torch.zeros(max(mat_ids) + 1, dtype=torch.long, device='cuda')
        for idx, mat_id in enumerate(mat_ids):
            mat_id_to_idx[mat_id] = idx
        self.register_buffer('mat_id_to_idx', mat_id_to_idx)

        # Find max emitter count and pad to uniform size
        num_emitters_list = [l2w.shape[0] for l2w in l2w_list]
        N_max = max(num_emitters_list)
        M = len(l2w_list)
        
        print(f"MultiAreaEmitter: Emitters per material: {num_emitters_list}, padding to N_max={N_max}")
        
        # Pad each l2w to [N_max, 4, 4] and create validity mask
        padded_l2w_list = []
        valid_mask = torch.zeros(M, N_max, dtype=torch.bool, device='cuda')
        for i, l2w in enumerate(l2w_list):
            N_i = l2w.shape[0]
            valid_mask[i, :N_i] = True
            if N_i < N_max:
                # Pad with identity matrices
                padding = torch.eye(4, device='cuda', dtype=l2w.dtype).unsqueeze(0).expand(N_max - N_i, 4, 4).clone()
                l2w = torch.cat([l2w, padding], dim=0)
            padded_l2w_list.append(l2w)

        # Stack all materials: [M, N_max, 4, 4]
        self.l2w = torch.stack(padded_l2w_list, dim=0)
        
        # Register validity mask and emitter counts
        self.register_buffer('valid_emitter_mask', valid_mask)  # [M, N_max]
        self.register_buffer('num_emitters_per_material', torch.tensor(num_emitters_list, dtype=torch.long, device='cuda'))  # [M]
        
        # Register buffers with extra material dimension [M, N_max, 3]
        self.register_buffer('light_positions', self.l2w[..., :3, 3]) 
        
        # Add a small tilt along z-axis
        tilt_angle = math.radians(0)  # 5 degree tilt, adjust as needed
        light_normal_local = torch.tensor([0.0, -math.cos(tilt_angle), math.sin(tilt_angle)], dtype=torch.float32, device='cuda')
        
        # Rotate normal: [M, N_max, 3, 3] @ [3] -> [M, N_max, 3]
        light_normals_world = torch.matmul(self.l2w[..., :3, :3], light_normal_local)
        light_normals_world = light_normals_world / (light_normals_world.norm(dim=-1, keepdim=True) + 1e-12)
        self.register_buffer('light_normal', light_normals_world)
        
        # Load calibrated directional distribution if provided
        self.calibrated_directional_distribution = False
        direction_json = cfg.get('direction_json', '')
        if direction_json:
            self._load_calibration_table(direction_json)
    
    def _load_calibration_table(self, direction_json):
        """
        Load the directional distribution calibration table from JSON file.
        Args:
            direction_json: path to the JSON file containing the calibration table
        """
        import json
        
        with open(direction_json, 'r') as f:
            calibration_data = json.load(f)
        
        # Extract table data
        resolution = calibration_data['resolution_degrees']
        data_dict = calibration_data['data']
        max_cam_rad_ratio = calibration_data['max_cam_rad_ratio']
        
        # Convert dictionary to sorted arrays for interpolation
        angles = []
        ratios = []
        for angle_str, ratio in sorted(data_dict.items(), key=lambda x: float(x[0])):
            angles.append(float(angle_str))
            ratios.append(ratio)
        
        # Store as tensors
        self.register_buffer('calibration_angles', torch.tensor(angles, dtype=torch.float32, device='cuda'))
        self.register_buffer('calibration_ratios', torch.tensor(ratios, dtype=torch.float32, device='cuda'))
        self.register_buffer('calibration_resolution', torch.tensor(resolution, dtype=torch.float32, device='cuda'))
        self.register_buffer('max_cam_rad_ratio', torch.tensor(max_cam_rad_ratio, dtype=torch.float32, device='cuda'))
        
        self.calibrated_directional_distribution = True
        print(f"Loaded directional calibration table from {direction_json}")
        print(f"  Resolution: {resolution}°, Angles: {len(angles)}, Max ratio: {max_cam_rad_ratio:.6f}")
        
    def _compute_l2w(self, cfg, json_path):
        """
        Compute the world transform for each light.
        """
        # Compute light transformation matrix
        R_l2g = torch.tensor(cfg.get('R_l2g'), dtype=torch.float32, device='cuda')
        t_l2g = torch.tensor(cfg.get('t_l2g'), dtype=torch.float32, device='cuda')
        base2_to_base1 = torch.tensor(cfg.get('base2_to_base1'), dtype=torch.float32, device='cuda')
        g2b0 = read_light_transforms(json_path, cfg.turntable.center, cfg.turntable.axis, base2_to_base1) # [N, 4, 4]
        # g2b0 = base2_to_base1.unsqueeze(0) @ g2b
        l2g = torch.eye(4, device='cuda')
        l2g[:3, :3] = R_l2g
        l2g[:3, 3] = t_l2g
        l2w = g2b0 @ l2g
        return l2w
    
    def _update_poses_for_vis(self, turntable_center, steps):
        turntable_center = turntable_center.to(self.l2w.device)
        light_normal_local = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32, device='cuda')
        p0_world = self.l2w[:3, 3]   # [3]
        R0_world = self.l2w[:3, :3]  # [3,3]
        n0_world = (R0_world @ light_normal_local)  # [3]
        n0_world = n0_world / (n0_world.norm() + 1e-12)

        # Clockwise in right-handed (+Z out) means negative angles
        angles = torch.arange(steps, device='cuda', dtype=torch.float32) * (-2.0 * math.pi / steps)  # [60]

        c = torch.cos(angles)
        s = torch.sin(angles)
        # Batch of Rz(θ): shape [60, 3, 3]
        Rz = torch.zeros(steps, 3, 3, device='cuda', dtype=torch.float32)
        Rz[:, 0, 0] =  c
        Rz[:, 0, 1] = -s
        Rz[:, 1, 0] =  s
        Rz[:, 1, 1] =  c
        Rz[:, 2, 2] =  1.0

        # Rotate the position around the center: p' = Rz*(p0 - center) + center
        rel = p0_world - turntable_center  # [3]
        rel = rel.unsqueeze(-1)            # [3,1] for batch matmul
        pos_rot = (Rz @ rel).squeeze(-1) + turntable_center  # [60,3]

        # Rotate the normal as a direction: n' = Rz * n0
        n0 = n0_world.unsqueeze(-1)  # [3,1]
        nor_rot = (Rz @ n0).squeeze(-1)  # [60,3]
        nor_rot = nor_rot / (nor_rot.norm(dim=-1, keepdim=True) + 1e-12)

        # ---- Register buffers ----
        with torch.no_grad():
            self.light_positions = pos_rot
            self.light_normal = nor_rot

                
    def _directional_distribution(self, light_dir, light_id, material_id):
        """
        Compute directional radiance L(θ) for rays headed from the light to the surface.
        Args:
            light_dir: (B, 3) directions from surface -> light (so emission dir is -light_dir)
            light_id: ID of the light
            material_id: ID of the material

        Returns:
            radiance: (B, 3) directional radiance (RGB) following L = L0 * cos^m(theta).
                      Clamped to zero for back-facing directions.
        """
        # Emission direction is from light -> surface
        v = -light_dir  # (B,3)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-12)

        # cos(theta) between light normal and emission direction
        # light_normal is [M, N, 3], index with material_id and light_id
        # Map material_id to tensor index (handles non-contiguous IDs)
        idx = self.mat_id_to_idx[material_id]
        
        # Validate light_id is within valid range for each material
        max_light_ids = self.num_emitters_per_material[idx]  # (B,)
        invalid_mask = light_id >= max_light_ids
        if invalid_mask.any():
            invalid_indices = torch.where(invalid_mask)[0]
            raise ValueError(
                f"Invalid light_id detected: light_id[{invalid_indices.tolist()}]="
                f"{light_id[invalid_mask].tolist()} >= max_emitters="
                f"{max_light_ids[invalid_mask].tolist()} for material_id="
                f"{material_id[invalid_mask].tolist()}"
            )
        light_n = self.light_normal[idx, light_id] # (B, 3)
        cos_theta = torch.clamp((v * light_n).sum(dim=-1, keepdim=True), -1, 1)

        if self.calibrated_directional_distribution:
            # Use calibrated table with linear interpolation
            # Convert cos(theta) to degrees
            theta_rad = torch.acos(cos_theta)  # (B, 1)
            theta_deg = theta_rad * 180.0 / math.pi  # (B, 1)
            
            # Linear interpolation in the calibration table
            theta_deg_flat = theta_deg.squeeze(-1)  # (B,)
            
            # Clamp to valid range [0, 90]
            theta_deg_flat = torch.clamp(theta_deg_flat, 0.0, self.calibration_angles[-1])

            # Use searchsorted to find indices for interpolation
            indices = torch.searchsorted(self.calibration_angles, theta_deg_flat, right=False)
            indices = torch.clamp(indices, 1, len(self.calibration_angles) - 1)
            
            # Get lower and upper bounds for linear interpolation
            lower_idx = indices - 1
            upper_idx = indices
            
            lower_angle = self.calibration_angles[lower_idx]  # (B,)
            upper_angle = self.calibration_angles[upper_idx]  # (B,)
            lower_ratio = self.calibration_ratios[lower_idx]  # (B,)
            upper_ratio = self.calibration_ratios[upper_idx]  # (B,)
            
            # Linear interpolation weight
            weight = (theta_deg_flat - lower_angle) / (upper_angle - lower_angle + 1e-8)
            weight = torch.clamp(weight, 0.0, 1.0)
            
            interpolated_ratio = lower_ratio + weight * (upper_ratio - lower_ratio)  # (B,)
            
            # The calibration table stores relative ratios (normalized to max)
            Lshape = interpolated_ratio.unsqueeze(-1)  # (B, 1)
            L0 = self.light_radiance  # (3,) or (1, 3)
            radiance = Lshape * L0  # (B, 3)
            if torch.isnan(radiance).any():
                print("radiance is nan")
        else:
            # Original cosine-power lobe model
            Lshape = cos_theta.pow(self.m)  # (B,1)
        L0 = self.light_radiance     # (B, 3)
        radiance = Lshape * L0          # (B,3)
        
        return radiance
    
    def sample_emitter(self, sample, position, light_id, material_id):
        """
        Sample a direction(position) from the area light (circular disk).
        Args:
            sample: Bx2 uniform samples for disk sampling
            position: Bx3 surface positions
            light_id: B light indices
            material_id: B material indices
        Returns:
            wi: Bx3 sampled directions
            pdf: Bx1 sampling pdf (area pdf)
            emit_position: Bx3 sampled positions
            emitter_normal: Bx3 sampled normals
        """
        B = position.shape[0]
        
        # Map material_id to tensor index (handles non-contiguous IDs)
        idx = self.mat_id_to_idx[material_id]
        
        # Get light properties for the specified light indices
        # light_positions and light_normal are [M, N, 3]
        light_pos = self.light_positions[idx, light_id]  # (B, 3)
        light_r = self.light_radius.expand(B)  # (B,)
        light_n = self.light_normal[idx, light_id]  # (B, 3)
        
        # Uniform sampling on disk using polar coordinates
        r_sample = torch.sqrt(sample[..., 0]) * light_r   # (B,)
        theta = 2.0 * math.pi * sample[..., 1]           # (B,)
        disk_x = r_sample * torch.cos(theta)             # (B,)
        disk_y = r_sample * torch.sin(theta)             # (B,)
        
        # Create orthonormal basis for each light plane
        up = torch.tensor([0.0, 1.0, 0.0], device=light_n.device, dtype=light_n.dtype)
        up = up.unsqueeze(0).expand(B, 3)  # (B, 3)
        
        # Check for near-parallel cases and use alternative up vector
        parallel_mask = torch.abs((light_n * up).sum(dim=-1)) > 0.9  # (B,)
        alt_up = torch.tensor([1.0, 0.0, 0.0], device=light_n.device, dtype=light_n.dtype)
        alt_up = alt_up.unsqueeze(0).expand(B, 3)  # (B, 3)
        up = torch.where(parallel_mask.unsqueeze(-1), alt_up, up)  # (B, 3)
        
        # Compute u_axis for each light
        u_axis = torch.cross(light_n, up, dim=-1)  # (B, 3)
        u_axis = u_axis / (torch.norm(u_axis, dim=-1, keepdim=True) + 1e-8)  # (B, 3)
        
        # Compute v_axis for each light
        v_axis = torch.cross(light_n, u_axis, dim=-1)  # (B, 3)
        v_axis = v_axis / (torch.norm(v_axis, dim=-1, keepdim=True) + 1e-8)  # (B, 3)
        
        # Compute sampled position on light disk
        emit_position = (
            light_pos 
            + disk_x.unsqueeze(-1) * u_axis 
            + disk_y.unsqueeze(-1) * v_axis
        )  # (B, 3)
        
        # Calculate direction from surface to light sample
        wi = emit_position - position  # (B, 3)
        distance = torch.norm(wi, dim=-1, keepdim=True)  # (B, 1)
        wi = wi / distance  # (B, 3)
        
        # Calculate area-based PDF
        # PDF = 1 / Area = 1 / (π r^2)
        light_area = math.pi * light_r * light_r  # (B,)
        pdf = 1.0 / light_area  # (B,)
        pdf = pdf.unsqueeze(-1)  # (B, 1)
        
        # Emitter normal (same for all sampled points)
        emitter_normal = light_n  # (B, 3)
        
        return wi, pdf, emit_position, emitter_normal

    def intersect(self, position, light_dir, light_id, material_id):
        """
        Intersect a ray with the area light (circular disk)
        Args:
            position: (B, 3) ray origins
            light_dir: (B, 3) ray directions (normalized)
            light_id: (B,) light indices
            material_id: (B,) material indices
        Returns:
            t: (B,) intersection distances (negative if no intersection)
            hit: (B,) boolean mask for valid intersections
            hit_pos: (B, 3) intersection positions
            light_idx: (B,) light indices
        """
        B = position.shape[0]
        
        # Map material_id to tensor index (handles non-contiguous IDs)
        idx = self.mat_id_to_idx[material_id]
        
        # Get light properties for the specified light indices
        light_pos = self.light_positions[idx, light_id]  # (B, 3)
        light_r = self.light_radius.expand(B)  # (B,)
        light_n = self.light_normal[idx, light_id]  # (B, 3)
        
        # Ray-plane intersection
        # Ray: p(t) = pos + t * dirs
        # Plane: (p - light_pos) · light_normal = 0
        # Substituting: (pos + t*dirs - light_pos) · light_normal = 0
        # Solving for t: t = (light_pos - pos) · light_normal / (dirs · light_normal)
        
        # Compute denominator (ray direction dot plane normal)
        denom = (light_dir * light_n).sum(dim=-1)  # (B,)

        # Check if ray is parallel to plane (denom ≈ 0)
        
        # Compute numerator
        to_light = light_pos - position  # (B, 3)
        numer = (to_light * light_n).sum(dim=-1)  # (B,)
        '''
        print("position",position)
        print("light_pos",light_pos)
        print("to_light",to_light)
        print("light_n",light_n)
        
        print("numer",numer)
        print("denom",denom)
        '''
        # Compute intersection distance
        t = numer / denom  # (B,)
        
        # Check if intersection is in front of ray origin
        
        # Compute intersection points
        hit_pos = position + t.unsqueeze(-1) * light_dir  # (B, 3)
        
        # Check if intersection point is within the circular disk
        # Create orthonormal basis for each light plane
        up = torch.tensor([0.0, 1.0, 0.0], device=light_n.device, dtype=light_n.dtype)
        up = up.unsqueeze(0).expand(B, 3)  # (B, 3)
        
        # Check for near-parallel cases and use alternative up vector
        parallel_mask = torch.abs((light_n * up).sum(dim=-1)) > 0.9  # (B,)
        alt_up = torch.tensor([1.0, 0.0, 0.0], device=light_n.device, dtype=light_n.dtype)
        alt_up = alt_up.unsqueeze(0).expand(B, 3)  # (B, 3)
        up = torch.where(parallel_mask.unsqueeze(-1), alt_up, up)  # (B, 3)
        
        # Compute u_axis for each light
        u_axis = torch.cross(light_n, up, dim=-1)  # (B, 3)
        u_axis = u_axis / (torch.norm(u_axis, dim=-1, keepdim=True) + 1e-8)  # (B, 3)
        
        # Compute v_axis for each light
        v_axis = torch.cross(light_n, u_axis, dim=-1)  # (B, 3)
        v_axis = v_axis / (torch.norm(v_axis, dim=-1, keepdim=True) + 1e-8)  # (B, 3)
        
        # Project intersection point onto the light plane coordinate system
        to_hit = hit_pos - light_pos  # (B, 3)
        u_coord = (to_hit * u_axis).sum(dim=-1)  # (B,)
        v_coord = (to_hit * v_axis).sum(dim=-1)  # (B,)
        
        # Check if within circular disk bounds (distance from center <= radius)
        dist_from_center = torch.sqrt(u_coord * u_coord + v_coord * v_coord)  # (B,)
        
        # Final hit mask: valid t, not parallel, and within disk
        
        # Set invalid distances to negative
        #t = torch.where(hit, t, torch.full_like(t, -1.0))
        hit=True
        
        # Light indices for each intersection
        light_idx = light_id  # (B,)
        
        return t, hit, hit_pos, light_idx
    
    def eval_emitter(self, position, light_dir, light_id, material_id):
        """
        Evaluate environment map radiance along given directions
        Args:
            position: Bx3 intersection points 
            light_dir: Bx3 light directions(from surface to light)
            light_id: B light indices
            material_id: B material indices
        Returns:
            Le: Bx3 radiance
            pdf: Bx1 pdf
            valid: B valid samples (always True for area light)
        """
        t, hit, hit_pos, light_idx=self.intersect(position,light_dir,light_id, material_id)
        '''
        print("t",t.shape)
        print("hit_pos",hit_pos.shape)
        print("light_dir",light_dir.shape)
        print("light_normal",self.light_normal[light_id].shape)
        '''
        B = position.shape[0]

        Le=self._directional_distribution(light_dir,light_id, material_id)
        if torch.isnan(Le).any():
            print("Le is nan")
        # dA_dw=((position-hit_pos)*(position-hit_pos)).sum(dim=-1)/((-light_dir)*self.light_normal[light_id]).sum(dim=-1)
        pdf=1.0/(self.light_radius.expand(B)*self.light_radius.expand(B)*torch.pi)
        pdf=pdf.unsqueeze(-1)

        if torch.isnan(Le).any():
            print("Le is nan")
        return Le, pdf, torch.ones_like(pdf, dtype=torch.bool)  # Always valid
    
class ConstantEmitter(nn.Module):
    def __init__(self, cfg, json_path):
        """
        Args:
            positions: (N, 3) light center
            radius: (N, 1) light radius
            radiance: (N, 1) largest radiance
        """
        super(ConstantEmitter, self).__init__()
        # Extract configuration parameters
        radius = cfg.get('radius', 0.007)
        fwhm_deg = cfg.get('fwhm_deg', 115.0)
        # self.light_radiance = nn.Parameter(torch.tensor(cfg.get('radiance'), dtype=torch.float32, device='cuda'))
        self.register_buffer('light_radiance', torch.tensor(cfg.get('radiance'), dtype=torch.float32, device='cuda'))

        theta_half = math.radians(fwhm_deg * 0.5)
        m = math.log(0.5) / math.log(max(1e-8, math.cos(theta_half)))
        self.register_buffer('m', torch.tensor(m, dtype=torch.float32, device='cuda'))
        self.register_buffer('light_radius', torch.tensor(radius, dtype=torch.float32, device='cuda'))
        self.sampling = False

        self.l2w = self._compute_l2w(cfg, json_path)
        self.register_buffer('light_positions', self.l2w[:, :3, 3])  # [N, 3] - translation part
        # Add a small tilt along z-axis
        tilt_angle = math.radians(0)  # 5 degree tilt, adjust as needed
        light_normal_local = torch.tensor([0.0, -math.cos(tilt_angle), math.sin(tilt_angle)], dtype=torch.float32, device='cuda')
        light_normals_world = torch.matmul(self.l2w[:, :3, :3], light_normal_local)  # [N, 3]
        light_normals_world = light_normals_world / (light_normals_world.norm(dim=-1, keepdim=True) + 1e-12)
        self.register_buffer('light_normal', light_normals_world)
        
        # Load calibrated directional distribution if provided
        self.calibrated_directional_distribution = False
        direction_json = cfg.get('direction_json', '')
        if direction_json:
            self._load_calibration_table(direction_json)
    
    def _load_calibration_table(self, direction_json):
        """
        Load the directional distribution calibration table from JSON file.
        Args:
            direction_json: path to the JSON file containing the calibration table
        """
        import json
        
        with open(direction_json, 'r') as f:
            calibration_data = json.load(f)
        
        # Extract table data
        resolution = calibration_data['resolution_degrees']
        data_dict = calibration_data['data']
        max_cam_rad_ratio = calibration_data['max_cam_rad_ratio']
        
        # Convert dictionary to sorted arrays for interpolation
        angles = []
        ratios = []
        for angle_str, ratio in sorted(data_dict.items(), key=lambda x: float(x[0])):
            angles.append(float(angle_str))
            ratios.append(ratio)
        
        # Store as tensors
        self.register_buffer('calibration_angles', torch.tensor(angles, dtype=torch.float32, device='cuda'))
        self.register_buffer('calibration_ratios', torch.tensor(ratios, dtype=torch.float32, device='cuda'))
        self.register_buffer('calibration_resolution', torch.tensor(resolution, dtype=torch.float32, device='cuda'))
        self.register_buffer('max_cam_rad_ratio', torch.tensor(max_cam_rad_ratio, dtype=torch.float32, device='cuda'))
        
        self.calibrated_directional_distribution = True
        print(f"Loaded directional calibration table from {direction_json}")
        print(f"  Resolution: {resolution}°, Angles: {len(angles)}, Max ratio: {max_cam_rad_ratio:.6f}")
        
    def _compute_l2w(self, cfg, json_path):
        """
        Compute the world transform for each light.
        """
        # Compute light transformation matrix
        R_l2g = torch.tensor(cfg.get('R_l2g'), dtype=torch.float32, device='cuda')
        t_l2g = torch.tensor(cfg.get('t_l2g'), dtype=torch.float32, device='cuda')
        base2_to_base1 = torch.tensor(cfg.get('base2_to_base1'), dtype=torch.float32, device='cuda')
        g2b0 = read_light_transforms(json_path, cfg.turntable.center, cfg.turntable.axis, base2_to_base1) # [N, 4, 4]
        # g2b0 = base2_to_base1.unsqueeze(0) @ g2b
        l2g = torch.eye(4, device='cuda')
        l2g[:3, :3] = R_l2g
        l2g[:3, 3] = t_l2g
        l2w = g2b0 @ l2g
        return l2w
    
    def _update_poses_for_vis(self, turntable_center, steps):
        turntable_center = turntable_center.to(self.l2w.device)
        light_normal_local = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32, device='cuda')
        p0_world = self.l2w[:3, 3]   # [3]
        R0_world = self.l2w[:3, :3]  # [3,3]
        n0_world = (R0_world @ light_normal_local)  # [3]
        n0_world = n0_world / (n0_world.norm() + 1e-12)

        # Clockwise in right-handed (+Z out) means negative angles
        angles = torch.arange(steps, device='cuda', dtype=torch.float32) * (-2.0 * math.pi / steps)  # [60]

        c = torch.cos(angles)
        s = torch.sin(angles)
        # Batch of Rz(θ): shape [60, 3, 3]
        Rz = torch.zeros(steps, 3, 3, device='cuda', dtype=torch.float32)
        Rz[:, 0, 0] =  c
        Rz[:, 0, 1] = -s
        Rz[:, 1, 0] =  s
        Rz[:, 1, 1] =  c
        Rz[:, 2, 2] =  1.0

        # Rotate the position around the center: p' = Rz*(p0 - center) + center
        rel = p0_world - turntable_center  # [3]
        rel = rel.unsqueeze(-1)            # [3,1] for batch matmul
        pos_rot = (Rz @ rel).squeeze(-1) + turntable_center  # [60,3]

        # Rotate the normal as a direction: n' = Rz * n0
        n0 = n0_world.unsqueeze(-1)  # [3,1]
        nor_rot = (Rz @ n0).squeeze(-1)  # [60,3]
        nor_rot = nor_rot / (nor_rot.norm(dim=-1, keepdim=True) + 1e-12)

        # ---- Register buffers ----
        with torch.no_grad():
            self.light_positions = pos_rot
            self.light_normal = nor_rot

                
    def _directional_distribution(self, light_dir, light_id):
        """
        Compute directional radiance L(θ) for rays headed from the light to the surface.
        Args:
            light_dir: (B, 3) directions from surface -> light (so emission dir is -light_dir)
            light_id: ID of the light

        Returns:
            radiance: (B, 3) directional radiance (RGB) following L = L0 * cos^m(theta).
                      Clamped to zero for back-facing directions.
        """
        # Emission direction is from light -> surface
        v = -light_dir  # (B,3)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-12)

        # cos(theta) between light normal and emission direction
        cos_theta = torch.clamp((v * self.light_normal[light_id]).sum(dim=-1, keepdim=True), -1, 1)

        if self.calibrated_directional_distribution:
            # Use calibrated table with linear interpolation
            # Convert cos(theta) to degrees
            theta_rad = torch.acos(cos_theta)  # (B, 1)
            theta_deg = theta_rad * 180.0 / math.pi  # (B, 1)
            
            # Linear interpolation in the calibration table
            theta_deg_flat = theta_deg.squeeze(-1)  # (B,)
            
            # Clamp to valid range [0, 90]
            theta_deg_flat = torch.clamp(theta_deg_flat, 0.0, self.calibration_angles[-1])

            # Use searchsorted to find indices for interpolation
            indices = torch.searchsorted(self.calibration_angles, theta_deg_flat, right=False)
            indices = torch.clamp(indices, 1, len(self.calibration_angles) - 1)
            
            # Get lower and upper bounds for linear interpolation
            lower_idx = indices - 1
            upper_idx = indices
            
            lower_angle = self.calibration_angles[lower_idx]  # (B,)
            upper_angle = self.calibration_angles[upper_idx]  # (B,)
            lower_ratio = self.calibration_ratios[lower_idx]  # (B,)
            upper_ratio = self.calibration_ratios[upper_idx]  # (B,)
            
            # Linear interpolation weight
            weight = (theta_deg_flat - lower_angle) / (upper_angle - lower_angle + 1e-8)
            weight = torch.clamp(weight, 0.0, 1.0)
            
            interpolated_ratio = lower_ratio + weight * (upper_ratio - lower_ratio)  # (B,)
            
            # The calibration table stores relative ratios (normalized to max)
            Lshape = interpolated_ratio.unsqueeze(-1)  # (B, 1)
            L0 = self.light_radiance  # (3,) or (1, 3)
            radiance = Lshape * L0  # (B, 3)
        else:
            # Original cosine-power lobe model
            Lshape = cos_theta.pow(self.m)  # (B,1)
        L0 = self.light_radiance     # (B, 3)
        radiance = Lshape * L0          # (B,3)
        
        return radiance
    
    def sample_emitter(self, sample, position, light_id):
        """
        Sample a direction(position) from the area light (circular disk).
        Args:
            sample: Bx2 uniform samples for disk sampling
            position: Bx3 surface positions
            light_id: B light indices
        Returns:
            wi: Bx3 sampled directions
            pdf: Bx1 sampling pdf (area pdf)
            emit_position: Bx3 sampled positions
            emitter_normal: Bx3 sampled normals
        """
        B = position.shape[0]
        
        # Get light properties for the specified light indices
        light_pos = self.light_positions[light_id]  # (B, 3)
        light_r = self.light_radius.expand(B)  # (B,)
        light_n = self.light_normal[light_id]  # (B, 3)
        
        # Uniform sampling on disk using polar coordinates
        r_sample = torch.sqrt(sample[..., 0]) * light_r   # (B,)
        theta = 2.0 * math.pi * sample[..., 1]           # (B,)
        disk_x = r_sample * torch.cos(theta)             # (B,)
        disk_y = r_sample * torch.sin(theta)             # (B,)
        
        # Create orthonormal basis for each light plane
        up = torch.tensor([0.0, 1.0, 0.0], device=light_n.device, dtype=light_n.dtype)
        up = up.unsqueeze(0).expand(B, 3)  # (B, 3)
        
        # Check for near-parallel cases and use alternative up vector
        parallel_mask = torch.abs((light_n * up).sum(dim=-1)) > 0.9  # (B,)
        alt_up = torch.tensor([1.0, 0.0, 0.0], device=light_n.device, dtype=light_n.dtype)
        alt_up = alt_up.unsqueeze(0).expand(B, 3)  # (B, 3)
        up = torch.where(parallel_mask.unsqueeze(-1), alt_up, up)  # (B, 3)
        
        # Compute u_axis for each light
        u_axis = torch.cross(light_n, up, dim=-1)  # (B, 3)
        u_axis = u_axis / (torch.norm(u_axis, dim=-1, keepdim=True) + 1e-8)  # (B, 3)
        
        # Compute v_axis for each light
        v_axis = torch.cross(light_n, u_axis, dim=-1)  # (B, 3)
        v_axis = v_axis / (torch.norm(v_axis, dim=-1, keepdim=True) + 1e-8)  # (B, 3)
        
        # Compute sampled position on light disk
        if self.sampling:
            emit_position = (
                light_pos 
                + disk_x.unsqueeze(-1) * u_axis 
                + disk_y.unsqueeze(-1) * v_axis
            )  # (B, 3)
        else:
            emit_position = light_pos
        
        # Calculate direction from surface to light sample
        wi = emit_position - position  # (B,  3)
        distance = torch.norm(wi, dim=-1, keepdim=True)  # (B, 1)
        wi = wi / distance  # (B, 3)
        
        # Calculate area-based PDF
        # PDF = 1 / Area = 1 / (π r^2)
        light_area = math.pi * light_r * light_r  # (B,)
        pdf = 1.0 / light_area  # (B,)
        pdf = pdf.unsqueeze(-1)  # (B, 1)
        
        # Emitter normal (same for all sampled points)
        emitter_normal = light_n  # (B, 3)
        
        return wi, pdf, emit_position, emitter_normal

    def intersect(self, position, light_dir, light_id):
        """
        Intersect a ray with the area light (circular disk)
        Args:
            position: (B, 3) ray origins
            light_dir: (B, 3) ray directions (normalized)
            light_id: (B,) light indices
        Returns:
            t: (B,) intersection distances (negative if no intersection)
            hit: (B,) boolean mask for valid intersections
            hit_pos: (B, 3) intersection positions
            light_idx: (B,) light indices
        """
        B = position.shape[0]
        
        # Get light properties for the specified light indices
        light_pos = self.light_positions[light_id]  # (B, 3)
        light_r = self.light_radius.expand(B)  # (B,)
        light_n = self.light_normal[light_id]  # (B, 3)
        
        # Ray-plane intersection
        # Ray: p(t) = pos + t * dirs
        # Plane: (p - light_pos) · light_normal = 0
        # Substituting: (pos + t*dirs - light_pos) · light_normal = 0
        # Solving for t: t = (light_pos - pos) · light_normal / (dirs · light_normal)
        
        # Compute denominator (ray direction dot plane normal)
        denom = (light_dir * light_n).sum(dim=-1)  # (B,)

        # Check if ray is parallel to plane (denom ≈ 0)
        
        # Compute numerator
        to_light = light_pos - position  # (B, 3)
        numer = (to_light * light_n).sum(dim=-1)  # (B,)
        '''
        print("position",position)
        print("light_pos",light_pos)
        print("to_light",to_light)
        print("light_n",light_n)
        
        print("numer",numer)
        print("denom",denom)
        '''
        # Compute intersection distance
        t = numer / denom  # (B,)
        
        # Check if intersection is in front of ray origin
        
        # Compute intersection points
        hit_pos = position + t.unsqueeze(-1) * light_dir  # (B, 3)
        
        # Check if intersection point is within the circular disk
        # Create orthonormal basis for each light plane
        up = torch.tensor([0.0, 1.0, 0.0], device=light_n.device, dtype=light_n.dtype)
        up = up.unsqueeze(0).expand(B, 3)  # (B, 3)
        
        # Check for near-parallel cases and use alternative up vector
        parallel_mask = torch.abs((light_n * up).sum(dim=-1)) > 0.9  # (B,)
        alt_up = torch.tensor([1.0, 0.0, 0.0], device=light_n.device, dtype=light_n.dtype)
        alt_up = alt_up.unsqueeze(0).expand(B, 3)  # (B, 3)
        up = torch.where(parallel_mask.unsqueeze(-1), alt_up, up)  # (B, 3)
        
        # Compute u_axis for each light
        u_axis = torch.cross(light_n, up, dim=-1)  # (B, 3)
        u_axis = u_axis / (torch.norm(u_axis, dim=-1, keepdim=True) + 1e-8)  # (B, 3)
        
        # Compute v_axis for each light
        v_axis = torch.cross(light_n, u_axis, dim=-1)  # (B, 3)
        v_axis = v_axis / (torch.norm(v_axis, dim=-1, keepdim=True) + 1e-8)  # (B, 3)
        
        # Project intersection point onto the light plane coordinate system
        to_hit = hit_pos - light_pos  # (B, 3)
        u_coord = (to_hit * u_axis).sum(dim=-1)  # (B,)
        v_coord = (to_hit * v_axis).sum(dim=-1)  # (B,)
        
        # Check if within circular disk bounds (distance from center <= radius)
        dist_from_center = torch.sqrt(u_coord * u_coord + v_coord * v_coord)  # (B,)
        
        # Final hit mask: valid t, not parallel, and within disk
        
        # Set invalid distances to negative
        #t = torch.where(hit, t, torch.full_like(t, -1.0))
        hit=True
        
        # Light indices for each intersection
        light_idx = light_id  # (B,)
        
        return t, hit, hit_pos, light_idx
    
    def eval_emitter(self, position, light_dir, light_id):
        """
        Evaluate environment map radiance along given directions
        Args:
            position: Bx3 intersection points 
            light_dir: Bx3 light directions(from surface to light)
        Returns:
            Le: Bx3 radiance
            pdf: Bx1 pdf
            valid: B valid samples (always True for area light)
        """
        t, hit, hit_pos, light_idx=self.intersect(position,light_dir,light_id)
        '''
        print("t",t.shape)
        print("hit_pos",hit_pos.shape)
        print("light_dir",light_dir.shape)
        print("light_normal",self.light_normal[light_id].shape)
        '''
        B = position.shape[0]

        Le = self.light_radiance.unsqueeze(0).expand(B, 3)
        # dA_dw=((position-hit_pos)*(position-hit_pos)).sum(dim=-1)/((-light_dir)*self.light_normal[light_id]).sum(dim=-1)
        pdf=1.0/(self.light_radius.expand(B)*self.light_radius.expand(B)*torch.pi)
        pdf=pdf.unsqueeze(-1)

        if torch.isnan(Le).any():
            print("Le is nan")
        return Le, pdf, torch.ones_like(pdf, dtype=torch.bool)  # Always valid

class RotateAreaEmitter(nn.Module):
    """
    Area emitter that creates emitters rotating along z-axis around a center point.
    Similar to RealNovelViewDataset camera rotation, but for emitters.
    Each emitter_id corresponds to one position in the rotation trajectory.
    
    Two sets of emitters:
    1. First num_lights emitters: rotate around z-axis at fixed altitude_angle
    2. Second num_views emitters: fixed azimuth, altitude varies from 0 to 90 degrees
    """
    def __init__(self, cfg):
        """
        Args:
            cfg: configuration object with:
                - num_lights: number of emitters in rotation trajectory (first set)
                - num_views: number of emitters with varying altitude (second set)
                - altitude_angle: elevation angle in degrees for first set (from horizontal plane)
                - fixed_azimuth: azimuth angle in degrees for second set (default: 0)
                - dist: distance from rotation center
                - radius: emitter disk radius
                - fwhm_deg: full width at half maximum angle for cosine falloff
                - radiance: [R, G, B] base radiance
                - direction_json: (optional) path to angular distribution calibration
                - bbox_json or turntable.center: rotation center
        """
        super(RotateAreaEmitter, self).__init__()
        import json
        import os
        
        # Extract configuration parameters
        radius = cfg.get('radius', 0.007)
        fwhm_deg = cfg.get('fwhm_deg', 115.0)
        self.register_buffer('light_radiance', torch.tensor(cfg.get('radiance'), dtype=torch.float32, device='cuda'))

        theta_half = math.radians(fwhm_deg * 0.5)
        m = math.log(0.5) / math.log(max(1e-8, math.cos(theta_half)))
        self.register_buffer('m', torch.tensor(m, dtype=torch.float32, device='cuda'))
        self.register_buffer('light_radius', torch.tensor(radius, dtype=torch.float32, device='cuda'))

        # Get rotation parameters from config
        num_lights = cfg.get('num_lights', 40)
        num_views = num_lights
        altitude_angle = cfg.get('altitude_angle', 30.0)  # degrees from horizontal
        fixed_azimuth = cfg.get('fixed_azimuth', 0.0)  # degrees for second set
        distance = cfg.get('dist', 0.4)
        
        # Load rotation center from bbox_json (like camera) or use turntable center
        bbox_json_path = cfg.get('bbox_json', '')
        if bbox_json_path and os.path.exists(bbox_json_path):
            with open(bbox_json_path, 'r') as f:
                bbox_data = json.load(f)
            center = bbox_data['bbox_center']
            print(f"RotateAreaEmitter: Loading center from bbox.json: {center}")
        elif hasattr(cfg, 'turntable') and hasattr(cfg.turntable, 'center'):
            center = cfg.turntable.center
            print(f"RotateAreaEmitter: Using turntable center: {center}")
        else:
            center = [0.0, 0.0, 0.0]
            print(f"RotateAreaEmitter: Using default center: {center}")
        
        self.register_buffer('rotation_center', torch.tensor(center, dtype=torch.float32, device='cuda'))
        
        # Generate first set: emitters rotating around z-axis at fixed altitude
        light_positions, light_normals = self._generate_rotating_emitters(
            num_lights, altitude_angle, distance, center
        )
        
        # Generate second set: emitters with fixed azimuth, varying altitude from 0 to 90
        if num_views > 0:
            lift_positions, lift_normals = self._generate_lifting_emitters(
                num_views, fixed_azimuth, distance, center
            )
            # Concatenate both sets
            light_positions = torch.cat([light_positions, lift_positions], dim=0)
            light_normals = torch.cat([light_normals, lift_normals], dim=0)
        
        # Register buffers: [N, 3] where N = num_lights + num_views
        self.register_buffer('light_positions', light_positions)  # [N, 3]
        self.register_buffer('light_normal', light_normals)  # [N, 3]
        self.num_emitters = num_lights + num_views
        self.num_rotating = num_lights
        self.num_lifting = num_views
        
        print(f"RotateAreaEmitter: Created {self.num_emitters} emitters")
        print(f"  First set (rotating): {num_lights} emitters at altitude {altitude_angle}°")
        if num_views > 0:
            print(f"  Second set (lifting): {num_views} emitters at azimuth {fixed_azimuth}°, altitude 0-90°")
        print(f"  Distance: {distance}, Center: {center}")
        
        # Load calibrated directional distribution if provided
        self.calibrated_directional_distribution = False
        direction_json = cfg.get('direction_json', '')
        if direction_json:
            self._load_calibration_table(direction_json)
    
    def _generate_rotating_emitters(self, num_lights, altitude_angle, distance, center):
        """
        Generate emitter positions and normals rotating around z-axis.
        Similar to RealNovelViewDataset.get_camera_rotation_dicts().
        
        Args:
            num_lights: number of emitters
            altitude_angle: elevation angle in degrees (from horizontal plane)
            distance: distance from rotation center
            center: [x, y, z] rotation center
            
        Returns:
            positions: [N, 3] emitter positions
            normals: [N, 3] emitter normals (pointing toward center)
        """
        center = torch.tensor(center, dtype=torch.float32, device='cuda')
        altitude = math.radians(altitude_angle)  # angle from horizontal plane
        
        # Uniformly sample azimuth angles around the circle
        azimuths = torch.linspace(0, 2 * math.pi, num_lights + 1, device='cuda')[:-1]  # [N]
        
        # Spherical to Cartesian conversion (same as RealNovelViewDataset)
        # altitude is the angle from horizontal plane (elevation)
        # azimuth is the angle around z-axis
        cos_alt = math.cos(altitude)
        sin_alt = math.sin(altitude)
        
        x = distance * cos_alt * torch.cos(azimuths) + center[0]  # [N]
        y = distance * cos_alt * torch.sin(azimuths) + center[1]  # [N]
        z = distance * sin_alt + center[2]  # scalar, broadcast to [N]
        z = z * torch.ones_like(x)
        
        positions = torch.stack([x, y, z], dim=-1)  # [N, 3]
        
        # Compute normals: emitters point toward center (opposite of camera look direction)
        # Normal = normalize(center - position)
        to_center = center.unsqueeze(0) - positions  # [N, 3]
        normals = to_center / (torch.norm(to_center, dim=-1, keepdim=True) + 1e-12)  # [N, 3]
        
        return positions, normals
    
    def _generate_lifting_emitters(self, num_views, fixed_azimuth, distance, center):
        """
        Generate emitter positions and normals with fixed azimuth and varying altitude.
        Altitude varies from 0 to 90 degrees.
        
        Args:
            num_views: number of emitters
            fixed_azimuth: azimuth angle in degrees (fixed for all emitters)
            distance: distance from rotation center
            center: [x, y, z] rotation center
            
        Returns:
            positions: [N, 3] emitter positions
            normals: [N, 3] emitter normals (pointing toward center)
        """
        center = torch.tensor(center, dtype=torch.float32, device='cuda')
        azimuth = math.radians(fixed_azimuth)  # fixed azimuth angle
        
        # Uniformly sample altitude angles from 0 to 90 degrees
        altitudes = torch.linspace(0, math.pi / 2, num_views, device='cuda')  # [N] radians
        
        # Spherical to Cartesian conversion
        # altitude is the angle from horizontal plane (elevation)
        # azimuth is the angle around z-axis (fixed)
        cos_azimuth = math.cos(azimuth)
        sin_azimuth = math.sin(azimuth)
        
        x = distance * torch.cos(altitudes) * cos_azimuth + center[0]  # [N]
        y = distance * torch.cos(altitudes) * sin_azimuth + center[1]  # [N]
        z = distance * torch.sin(altitudes) + center[2]  # [N]
        
        positions = torch.stack([x, y, z], dim=-1)  # [N, 3]
        
        # Compute normals: emitters point toward center
        # Normal = normalize(center - position)
        to_center = center.unsqueeze(0) - positions  # [N, 3]
        normals = to_center / (torch.norm(to_center, dim=-1, keepdim=True) + 1e-12)  # [N, 3]
        
        return positions, normals
    
    def _load_calibration_table(self, direction_json):
        """
        Load the directional distribution calibration table from JSON file.
        Args:
            direction_json: path to the JSON file containing the calibration table
        """
        import json
        
        with open(direction_json, 'r') as f:
            calibration_data = json.load(f)
        
        # Extract table data
        resolution = calibration_data['resolution_degrees']
        data_dict = calibration_data['data']
        max_cam_rad_ratio = calibration_data['max_cam_rad_ratio']
        
        # Convert dictionary to sorted arrays for interpolation
        angles = []
        ratios = []
        for angle_str, ratio in sorted(data_dict.items(), key=lambda x: float(x[0])):
            angles.append(float(angle_str))
            ratios.append(ratio)
        
        # Store as tensors
        self.register_buffer('calibration_angles', torch.tensor(angles, dtype=torch.float32, device='cuda'))
        self.register_buffer('calibration_ratios', torch.tensor(ratios, dtype=torch.float32, device='cuda'))
        self.register_buffer('calibration_resolution', torch.tensor(resolution, dtype=torch.float32, device='cuda'))
        self.register_buffer('max_cam_rad_ratio', torch.tensor(max_cam_rad_ratio, dtype=torch.float32, device='cuda'))
        
        self.calibrated_directional_distribution = True
        print(f"Loaded directional calibration table from {direction_json}")
        print(f"  Resolution: {resolution}°, Angles: {len(angles)}, Max ratio: {max_cam_rad_ratio:.6f}")
                
    def _directional_distribution(self, light_dir, light_id):
        """
        Compute directional radiance L(θ) for rays headed from the light to the surface.
        Args:
            light_dir: (B, 3) directions from surface -> light (so emission dir is -light_dir)
            light_id: (B,) light indices

        Returns:
            radiance: (B, 3) directional radiance (RGB) following L = L0 * cos^m(theta).
                      Clamped to zero for back-facing directions.
        """
        # Emission direction is from light -> surface
        v = -light_dir  # (B,3)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-12)

        # cos(theta) between light normal and emission direction
        # light_normal is [N, 3], index with light_id
        light_n = self.light_normal[light_id]  # (B, 3)
        cos_theta = torch.clamp((v * light_n).sum(dim=-1, keepdim=True), -1, 1)

        if self.calibrated_directional_distribution:
            # Use calibrated table with linear interpolation
            # Convert cos(theta) to degrees
            theta_rad = torch.acos(cos_theta)  # (B, 1)
            theta_deg = theta_rad * 180.0 / math.pi  # (B, 1)
            
            # Linear interpolation in the calibration table
            theta_deg_flat = theta_deg.squeeze(-1)  # (B,)
            
            # Clamp to valid range [0, 90]
            theta_deg_flat = torch.clamp(theta_deg_flat, 0.0, self.calibration_angles[-1])

            # Use searchsorted to find indices for interpolation
            indices = torch.searchsorted(self.calibration_angles, theta_deg_flat, right=False)
            indices = torch.clamp(indices, 1, len(self.calibration_angles) - 1)
            
            # Get lower and upper bounds for linear interpolation
            lower_idx = indices - 1
            upper_idx = indices
            
            lower_angle = self.calibration_angles[lower_idx]  # (B,)
            upper_angle = self.calibration_angles[upper_idx]  # (B,)
            lower_ratio = self.calibration_ratios[lower_idx]  # (B,)
            upper_ratio = self.calibration_ratios[upper_idx]  # (B,)
            
            # Linear interpolation weight
            weight = (theta_deg_flat - lower_angle) / (upper_angle - lower_angle + 1e-8)
            weight = torch.clamp(weight, 0.0, 1.0)
            
            interpolated_ratio = lower_ratio + weight * (upper_ratio - lower_ratio)  # (B,)
            
            # The calibration table stores relative ratios (normalized to max)
            Lshape = interpolated_ratio.unsqueeze(-1)  # (B, 1)
            L0 = self.light_radiance  # (3,) or (1, 3)
            radiance = Lshape * L0  # (B, 3)
            if torch.isnan(radiance).any():
                print("radiance is nan")
        else:
            # Original cosine-power lobe model
            Lshape = cos_theta.pow(self.m)  # (B,1)
            L0 = self.light_radiance  # (3,)
            radiance = Lshape * L0  # (B,3)
        
        return radiance
    
    def sample_emitter(self, sample, position, light_id):
        """
        Sample a direction(position) from the area light (circular disk).
        Args:
            sample: Bx2 uniform samples for disk sampling
            position: Bx3 surface positions
            light_id: B light indices
        Returns:
            wi: Bx3 sampled directions
            pdf: Bx1 sampling pdf (area pdf)
            emit_position: Bx3 sampled positions
            emitter_normal: Bx3 sampled normals
        """
        B = position.shape[0]
        
        # Get light properties for the specified light indices
        # light_positions and light_normal are [N, 3]
        light_pos = self.light_positions[light_id]  # (B, 3)
        light_r = self.light_radius.expand(B)  # (B,)
        light_n = self.light_normal[light_id]  # (B, 3)
        
        # Uniform sampling on disk using polar coordinates
        r_sample = torch.sqrt(sample[..., 0]) * light_r   # (B,)
        theta = 2.0 * math.pi * sample[..., 1]           # (B,)
        disk_x = r_sample * torch.cos(theta)             # (B,)
        disk_y = r_sample * torch.sin(theta)             # (B,)
        
        # Create orthonormal basis for each light plane
        up = torch.tensor([0.0, 1.0, 0.0], device=light_n.device, dtype=light_n.dtype)
        up = up.unsqueeze(0).expand(B, 3)  # (B, 3)
        
        # Check for near-parallel cases and use alternative up vector
        parallel_mask = torch.abs((light_n * up).sum(dim=-1)) > 0.9  # (B,)
        alt_up = torch.tensor([1.0, 0.0, 0.0], device=light_n.device, dtype=light_n.dtype)
        alt_up = alt_up.unsqueeze(0).expand(B, 3)  # (B, 3)
        up = torch.where(parallel_mask.unsqueeze(-1), alt_up, up)  # (B, 3)
        
        # Compute u_axis for each light
        u_axis = torch.cross(light_n, up, dim=-1)  # (B, 3)
        u_axis = u_axis / (torch.norm(u_axis, dim=-1, keepdim=True) + 1e-8)  # (B, 3)
        
        # Compute v_axis for each light
        v_axis = torch.cross(light_n, u_axis, dim=-1)  # (B, 3)
        v_axis = v_axis / (torch.norm(v_axis, dim=-1, keepdim=True) + 1e-8)  # (B, 3)
        
        # Compute sampled position on light disk
        emit_position = (
            light_pos 
            + disk_x.unsqueeze(-1) * u_axis 
            + disk_y.unsqueeze(-1) * v_axis
        )  # (B, 3)
        
        # Calculate direction from surface to light sample
        wi = emit_position - position  # (B, 3)
        distance = torch.norm(wi, dim=-1, keepdim=True)  # (B, 1)
        wi = wi / distance  # (B, 3)
        
        # Calculate area-based PDF
        # PDF = 1 / Area = 1 / (π r^2)
        light_area = math.pi * light_r * light_r  # (B,)
        pdf = 1.0 / light_area  # (B,)
        pdf = pdf.unsqueeze(-1)  # (B, 1)
        
        # Emitter normal (same for all sampled points)
        emitter_normal = light_n  # (B, 3)
        
        return wi, pdf, emit_position, emitter_normal

    def intersect(self, position, light_dir, light_id):
        """
        Intersect a ray with the area light (circular disk)
        Args:
            position: (B, 3) ray origins
            light_dir: (B, 3) ray directions (normalized)
            light_id: (B,) light indices
        Returns:
            t: (B,) intersection distances (negative if no intersection)
            hit: (B,) boolean mask for valid intersections
            hit_pos: (B, 3) intersection positions
            light_idx: (B,) light indices
        """
        B = position.shape[0]
        
        # Get light properties for the specified light indices
        light_pos = self.light_positions[light_id]  # (B, 3)
        light_r = self.light_radius.expand(B)  # (B,)
        light_n = self.light_normal[light_id]  # (B, 3)
        
        # Ray-plane intersection
        # Ray: p(t) = pos + t * dirs
        # Plane: (p - light_pos) · light_normal = 0
        # Substituting: (pos + t*dirs - light_pos) · light_normal = 0
        # Solving for t: t = (light_pos - pos) · light_normal / (dirs · light_normal)
        
        # Compute denominator (ray direction dot plane normal)
        denom = (light_dir * light_n).sum(dim=-1)  # (B,)

        # Compute numerator
        to_light = light_pos - position  # (B, 3)
        numer = (to_light * light_n).sum(dim=-1)  # (B,)

        # Compute intersection distance
        t = numer / denom  # (B,)
        
        # Compute intersection points
        hit_pos = position + t.unsqueeze(-1) * light_dir  # (B, 3)
        
        # Check if intersection point is within the circular disk
        # Create orthonormal basis for each light plane
        up = torch.tensor([0.0, 1.0, 0.0], device=light_n.device, dtype=light_n.dtype)
        up = up.unsqueeze(0).expand(B, 3)  # (B, 3)
        
        # Check for near-parallel cases and use alternative up vector
        parallel_mask = torch.abs((light_n * up).sum(dim=-1)) > 0.9  # (B,)
        alt_up = torch.tensor([1.0, 0.0, 0.0], device=light_n.device, dtype=light_n.dtype)
        alt_up = alt_up.unsqueeze(0).expand(B, 3)  # (B, 3)
        up = torch.where(parallel_mask.unsqueeze(-1), alt_up, up)  # (B, 3)
        
        # Compute u_axis for each light
        u_axis = torch.cross(light_n, up, dim=-1)  # (B, 3)
        u_axis = u_axis / (torch.norm(u_axis, dim=-1, keepdim=True) + 1e-8)  # (B, 3)
        
        # Compute v_axis for each light
        v_axis = torch.cross(light_n, u_axis, dim=-1)  # (B, 3)
        v_axis = v_axis / (torch.norm(v_axis, dim=-1, keepdim=True) + 1e-8)  # (B, 3)
        
        # Project intersection point onto the light plane coordinate system
        to_hit = hit_pos - light_pos  # (B, 3)
        u_coord = (to_hit * u_axis).sum(dim=-1)  # (B,)
        v_coord = (to_hit * v_axis).sum(dim=-1)  # (B,)
        
        # Check if within circular disk bounds (distance from center <= radius)
        dist_from_center = torch.sqrt(u_coord * u_coord + v_coord * v_coord)  # (B,)
        
        hit = True  # For simplicity, always consider hit
        
        # Light indices for each intersection
        light_idx = light_id  # (B,)
        
        return t, hit, hit_pos, light_idx
    
    def eval_emitter(self, position, light_dir, light_id):
        """
        Evaluate emitter radiance along given directions
        Args:
            position: Bx3 intersection points 
            light_dir: Bx3 light directions (from surface to light)
            light_id: B light indices
        Returns:
            Le: Bx3 radiance
            pdf: Bx1 pdf
            valid: B valid samples (always True for area light)
        """
        t, hit, hit_pos, light_idx = self.intersect(position, light_dir, light_id)
        B = position.shape[0]

        Le = self._directional_distribution(light_dir, light_id)
        if torch.isnan(Le).any():
            print("Le is nan")
        
        pdf = 1.0 / (self.light_radius.expand(B) * self.light_radius.expand(B) * torch.pi)
        pdf = pdf.unsqueeze(-1)

        return Le, pdf, torch.ones_like(pdf, dtype=torch.bool)  # Always valid