"""
MERL BRDF Interface - GPU-accelerated BRDF lookup.

This implementation closely follows the reference MERL C++ code (BRDFRead.cpp)
for accurate BRDF value lookups using Rusinkiewicz parameterization.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import struct
from pathlib import Path
import math


class MERLInterface:
    """
    GPU-accelerated interface for MERL BRDF database.
    
    This class provides efficient BRDF lookups using the Rusinkiewicz parameterization,
    matching the reference MERL implementation.
    
    Args:
        brdf_dir: Path to directory containing .binary BRDF files
        device: Device to store tensors ('cuda' or 'cpu')
    """
    
    # MERL BRDF constants (from BRDFRead.cpp, lines 26-28)
    BRDF_SAMPLING_RES_THETA_H = 90
    BRDF_SAMPLING_RES_THETA_D = 90
    BRDF_SAMPLING_RES_PHI_D = 360  # But only 180 values stored due to reciprocity
    
    # Scaling factors from MERL (lines 29-31 in BRDFRead.cpp)
    RED_SCALE = 1.0 / 1500.0
    GREEN_SCALE = 1.15 / 1500.0
    BLUE_SCALE = 1.66 / 1500.0
    
    def __init__(self, brdf_dir, device='cuda', material_names=None):
        """
        Initialize MERL BRDF interface and load all materials from directory.

        Args:
            brdf_dir: Path to directory containing .binary BRDF files
            device: Device to store tensors ('cuda' or 'cpu')
            material_names: Optional iterable of material stem names (without
                ".binary"). When provided, only those materials are loaded and
                material indices map 0..len(material_names)-1 in the order the
                filtered .binary files appear after sorting.
        """
        self.device = torch.device(device)
        self.brdf_dir = Path(brdf_dir)

        if not self.brdf_dir.exists():
            raise FileNotFoundError(f"BRDF directory not found: {brdf_dir}")

        if not self.brdf_dir.is_dir():
            raise ValueError(f"Path is not a directory: {brdf_dir}")

        # Load all BRDF materials
        self._material_filter = set(material_names) if material_names is not None else None
        self.brdf_data, self.material_names, self.material_ids = self._load_all_brdfs()
        
        print(f"[MERLInterface] Loaded {len(self.material_names)} materials from {self.brdf_dir}")
        print(f"[MERLInterface] Device: {self.device}")
        print(f"[MERLInterface] Data shape: {self.brdf_data.shape}")
        print(f"[MERLInterface] Memory: {self.brdf_data.element_size() * self.brdf_data.nelement() / 1e6:.2f} MB")
        print(f"[MERLInterface] Materials: {', '.join(self.material_names)}")
    
    def _load_all_brdfs(self):
        """
        Load all BRDF data from binary files in the directory.
        
        Returns:
            brdf_data: torch.Tensor of shape (n_materials, n_samples * 3) on device
            material_names: List of material names
            material_ids: Dict mapping material name to index
        """
        # Find all .binary files
        brdf_files = sorted(self.brdf_dir.glob("*.binary"))

        if len(brdf_files) == 0:
            raise ValueError(f"No .binary files found in {self.brdf_dir}")

        if self._material_filter is not None:
            requested = self._material_filter
            brdf_files = [f for f in brdf_files if f.stem in requested]
            missing = requested - {f.stem for f in brdf_files}
            if missing:
                raise ValueError(
                    f"Requested materials not found in {self.brdf_dir}: "
                    f"{sorted(missing)}"
                )
            if len(brdf_files) == 0:
                raise ValueError(
                    f"Material filter matched no .binary files in {self.brdf_dir}"
                )
        
        material_names = []
        material_data_list = []
        
        n_samples_per_material = (self.BRDF_SAMPLING_RES_THETA_H * 
                                  self.BRDF_SAMPLING_RES_THETA_D * 
                                  self.BRDF_SAMPLING_RES_PHI_D // 2)
        
        for brdf_file in brdf_files:
            material_name = brdf_file.stem  # filename without extension
            material_names.append(material_name)
            
            # Load single BRDF file
            with open(brdf_file, 'rb') as f:
                # Read dimensions (3 int32 values)
                dims = struct.unpack('iii', f.read(12))
                n_theta_h, n_theta_d, n_phi_d = dims
                
                # Verify dimensions
                if n_theta_h * n_theta_d * n_phi_d != n_samples_per_material:
                    raise ValueError(
                        f"Dimensions mismatch in {brdf_file.name}: got {dims}, expected "
                        f"({self.BRDF_SAMPLING_RES_THETA_H}, "
                        f"{self.BRDF_SAMPLING_RES_THETA_D}, "
                        f"{self.BRDF_SAMPLING_RES_PHI_D // 2})"
                    )
                
                # Read BRDF data as doubles
                n_samples = n_theta_h * n_theta_d * n_phi_d
                brdf_data = np.fromfile(f, dtype=np.float64, count=n_samples * 3)
                
                if len(brdf_data) != n_samples * 3:
                    raise ValueError(
                        f"Incomplete data in {brdf_file.name}: expected {n_samples * 3}, got {len(brdf_data)}"
                    )
                
                material_data_list.append(brdf_data)
        
        # Stack all materials into a single tensor
        # Shape: (n_materials, n_samples * 3)
        # Data layout for each material: [R_0...R_n, G_0...G_n, B_0...B_n]
        all_data = np.stack(material_data_list, axis=0)
        brdf_data = torch.from_numpy(all_data).to(self.device)
        
        # Create material ID mapping
        material_ids = {name: idx for idx, name in enumerate(material_names)}
        
        return brdf_data, material_names, material_ids
    
    def _std_coords_to_half_diff_coords(self, theta_in, phi_in, theta_out, phi_out):
        """
        Convert standard coordinates to half-difference coordinates.
        
        This follows the MERL reference implementation (lines 80-127 in BRDFRead.cpp).
        
        Args:
            theta_in: [B] or scalar, incoming polar angle [0, pi/2]
            phi_in: [B] or scalar, incoming azimuthal angle [-pi, pi]
            theta_out: [B] or scalar, outgoing polar angle [0, pi/2]
            phi_out: [B] or scalar, outgoing azimuthal angle [-pi, pi]
        
        Returns:
            theta_h: [B] half-vector polar angle
            phi_h: [B] half-vector azimuthal angle
            theta_d: [B] difference polar angle
            phi_d: [B] difference azimuthal angle
        """
        # Compute in vector (lines 84-90)
        in_vec_z = torch.cos(theta_in)
        proj_in_vec = torch.sin(theta_in)
        in_vec_x = proj_in_vec * torch.cos(phi_in)
        in_vec_y = proj_in_vec * torch.sin(phi_in)
        in_vec = torch.stack([in_vec_x, in_vec_y, in_vec_z], dim=-1)
        in_vec = F.normalize(in_vec, dim=-1)
        
        # Compute out vector (lines 93-99)
        out_vec_z = torch.cos(theta_out)
        proj_out_vec = torch.sin(theta_out)
        out_vec_x = proj_out_vec * torch.cos(phi_out)
        out_vec_y = proj_out_vec * torch.sin(phi_out)
        out_vec = torch.stack([out_vec_x, out_vec_y, out_vec_z], dim=-1)
        out_vec = F.normalize(out_vec, dim=-1)
        
        # Compute halfway vector (lines 102-107)
        half_x = (in_vec[..., 0] + out_vec[..., 0]) / 2.0
        half_y = (in_vec[..., 1] + out_vec[..., 1]) / 2.0
        half_z = (in_vec[..., 2] + out_vec[..., 2]) / 2.0
        half = torch.stack([half_x, half_y, half_z], dim=-1)
        half = F.normalize(half, dim=-1)
        
        #print(f"half: {half[..., 1]}, {half[..., 0]}")
        # Compute theta_half, phi_half (lines 109-111)
        theta_h = torch.acos(torch.clamp(half[..., 2], -1.0, 1.0))
        phi_h = torch.atan2(half[..., 1], half[..., 0])
        
        # Rotate in vector by -phi_h around z-axis (normal) (line 120)
        normal = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=in_vec.dtype)
        temp = self._rotate_vector(in_vec, normal, -phi_h)
        
        # Rotate by -theta_h around y-axis (binormal) (line 121)
        bi_normal = torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=in_vec.dtype)
        diff = self._rotate_vector(temp, bi_normal, -theta_h)
        
        # Compute theta_diff, phi_diff (lines 123-125)
        theta_d = torch.acos(torch.clamp(diff[..., 2], -1.0, 1.0))
        phi_d = torch.atan2(diff[..., 1], diff[..., 0])
        
        return theta_h, phi_h, theta_d, phi_d
    
    def _rotate_vector(self, vector, axis, angle):
        """
        Rotate vector around an axis by an angle.
        
        This implements the rotate_vector function from BRDFRead.cpp (lines 53-76).
        
        Args:
            vector: [B, 3] or [3] vector to rotate
            axis: [3] rotation axis (normalized)
            angle: [B] or scalar, rotation angle in radians
        
        Returns:
            [B, 3] rotated vector
        """
        cos_ang = torch.cos(angle)
        sin_ang = torch.sin(angle)
        
        # Expand dimensions if needed
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        if axis.dim() == 1:
            axis = axis.unsqueeze(0).expand(vector.shape[0], -1)
        
        # Ensure angle has correct shape
        if not isinstance(angle, torch.Tensor):
            angle = torch.tensor(angle, device=self.device)
        if angle.dim() == 0:
            cos_ang = cos_ang.unsqueeze(0).expand(vector.shape[0])
            sin_ang = sin_ang.unsqueeze(0).expand(vector.shape[0])
        
        # out = vector * cos(angle) (lines 60-62)
        out = vector * cos_ang.unsqueeze(-1)
        
        # temp = axis · vector * (1 - cos(angle)) (lines 64-65)
        temp = (axis * vector).sum(dim=-1, keepdim=True) * (1.0 - cos_ang.unsqueeze(-1))
        
        # out += axis * temp (lines 67-69)
        out = out + axis * temp
        
        # cross = axis × vector (line 71)
        cross = torch.cross(axis, vector, dim=-1)
        
        # out += cross * sin(angle) (lines 73-75)
        out = out + cross * sin_ang.unsqueeze(-1)
        
        if squeeze_output:
            out = out.squeeze(0)
        
        return out
    
    def _theta_half_index(self, theta_h):
        """
        Lookup theta_half index with non-linear mapping.
        
        This implements the exact mapping from BRDFRead.cpp (lines 134-146).
        
        In:  [0 .. pi/2]
        Out: [0 .. 89]
        
        Args:
            theta_h: [B] theta_half values in radians
        
        Returns:
            [B] indices in range [0, 89]
        """
        theta_h = torch.clamp(theta_h, min=0.0)
        
        # Non-linear mapping (line 138-140)
        theta_h_deg = (theta_h / (math.pi / 2.0)) * self.BRDF_SAMPLING_RES_THETA_H
        temp = theta_h_deg * self.BRDF_SAMPLING_RES_THETA_H
        temp = torch.sqrt(temp)
        
        ret_val = temp.long()
        ret_val = torch.clamp(ret_val, 0, self.BRDF_SAMPLING_RES_THETA_H - 1)
        
        return ret_val
    
    def _theta_diff_index(self, theta_d):
        """
        Lookup theta_diff index with linear mapping.
        
        This implements BRDFRead.cpp lines 152-161.
        
        In:  [0 .. pi/2]
        Out: [0 .. 89]
        
        Args:
            theta_d: [B] theta_diff values in radians
        
        Returns:
            [B] indices in range [0, 89]
        """
        tmp = (theta_d / (math.pi * 0.5) * self.BRDF_SAMPLING_RES_THETA_D).long()
        tmp = torch.clamp(tmp, 0, self.BRDF_SAMPLING_RES_THETA_D - 1)
        return tmp
    
    def _phi_diff_index(self, phi_d):
        """
        Lookup phi_diff index.
        
        This implements BRDFRead.cpp lines 165-181, including reciprocity handling.
        
        Due to reciprocity, BRDF is unchanged under phi_diff -> phi_diff + pi.
        Only [0, pi] is stored (180 samples instead of 360).
        
        In:  phi_diff in [-pi .. pi]
        Out: [0 .. 179]
        
        Args:
            phi_d: [B] phi_diff values in radians
        
        Returns:
            [B] indices in range [0, 179]
        """
        # Handle reciprocity (lines 169-170)
        phi_d = torch.where(phi_d < 0.0, phi_d + math.pi, phi_d)
        
        # Map to index (line 174)
        tmp = (phi_d / math.pi * (self.BRDF_SAMPLING_RES_PHI_D // 2)).long()
        tmp = torch.clamp(tmp, 0, (self.BRDF_SAMPLING_RES_PHI_D // 2) - 1)
        
        return tmp
    
    def lookup(self, theta_in, phi_in, theta_out, phi_out, material_id):
        """
        Look up BRDF values for given incoming/outgoing angles and material.
        
        This implements the lookup_brdf_val function from BRDFRead.cpp (lines 185-211).
        
        Args:
            theta_in: [B] or scalar, incoming polar angle [0, pi/2]
            phi_in: [B] or scalar, incoming azimuthal angle [-pi, pi]
            theta_out: [B] or scalar, outgoing polar angle [0, pi/2]
            phi_out: [B] or scalar, outgoing azimuthal angle [-pi, pi]
            material_id: [B] or scalar, material index or tensor of indices
        
        Returns:
            rgb: [B, 3] or [3] BRDF RGB values
        """
        # Convert to half-difference coordinates (lines 192-193)
        theta_h, phi_h, theta_d, phi_d = self._std_coords_to_half_diff_coords(
            theta_in, phi_in, theta_out, phi_out)
        
        #print(f"theta_half: {theta_h}, fi_half: {phi_h}, theta_diff: {theta_d}, fi_diff: {phi_d}")
        
        # Get indices (lines 198-201)
        # Note: phi_half is ignored since isotropic BRDFs are assumed
        ind_phi = self._phi_diff_index(phi_d)
        ind_theta_d = self._theta_diff_index(theta_d)
        ind_theta_h = self._theta_half_index(theta_h)
        
        # Compute linear index (lines 198-201)
        # Note: PHI_D is divided by 2 because only half the range is stored due to reciprocity
        # ind = phi_diff_index(fi_diff) +
        #       theta_diff_index(theta_diff) * BRDF_SAMPLING_RES_PHI_D / 2 +
        #       theta_half_index(theta_half) * BRDF_SAMPLING_RES_PHI_D / 2 * BRDF_SAMPLING_RES_THETA_D
        ind = (ind_phi + 
               ind_theta_d * (self.BRDF_SAMPLING_RES_PHI_D // 2) +
               ind_theta_h * (self.BRDF_SAMPLING_RES_PHI_D // 2) * self.BRDF_SAMPLING_RES_THETA_D)
        #print(f"ind_phi: {ind_phi}, ind_theta_d: {ind_theta_d}, ind_theta_h: {ind_theta_h}")
        #print(f"ind: {ind}")
        
        # Convert material_id to tensor if needed
        if not isinstance(material_id, torch.Tensor):
            material_id = torch.tensor(material_id, device=self.device, dtype=torch.long)
        
        # Extract R, G, B from separate channel blocks (lines 205-207)
        # Data layout for each material: [R_0...R_n, G_0...G_n, B_0...B_n]
        n = self.BRDF_SAMPLING_RES_THETA_H * self.BRDF_SAMPLING_RES_THETA_D * (self.BRDF_SAMPLING_RES_PHI_D // 2)
        
        # Index into brdf_data using material_id
        red_val = self.brdf_data[material_id, ind] * self.RED_SCALE
        green_val = self.brdf_data[material_id, ind + n] * self.GREEN_SCALE
        blue_val = self.brdf_data[material_id, ind + 2 * n] * self.BLUE_SCALE
        
        rgb = torch.stack([red_val, green_val, blue_val], dim=-1)
        
        # Check for below-horizon values (lines 208-209)
        if (rgb < 0).any():
            print("Below horizon.", file=sys.stderr)
        
        return rgb,phi_d,theta_d,theta_h

    def lookup_angle(self, phi_d,theta_d,theta_h, material_id):
        """
        Look up BRDF values for given incoming/outgoing angles and material.
        
        This implements the lookup_brdf_val function from BRDFRead.cpp (lines 185-211).
        
        Args:
            theta_in: [B] or scalar, incoming polar angle [0, pi/2]
            phi_in: [B] or scalar, incoming azimuthal angle [-pi, pi]
            theta_out: [B] or scalar, outgoing polar angle [0, pi/2]
            phi_out: [B] or scalar, outgoing azimuthal angle [-pi, pi]
            material_id: [B] or scalar, material index or tensor of indices
        
        Returns:
            rgb: [B, 3] or [3] BRDF RGB values
        """
        print("material_id",material_id)
        ind_phi = self._phi_diff_index(phi_d)
        ind_theta_d = self._theta_diff_index(theta_d)
        ind_theta_h = self._theta_half_index(theta_h)
        
        # Compute linear index (lines 198-201)
        # Note: PHI_D is divided by 2 because only half the range is stored due to reciprocity
        # ind = phi_diff_index(fi_diff) +
        #       theta_diff_index(theta_diff) * BRDF_SAMPLING_RES_PHI_D / 2 +
        #       theta_half_index(theta_half) * BRDF_SAMPLING_RES_PHI_D / 2 * BRDF_SAMPLING_RES_THETA_D
        ind = (ind_phi + 
               ind_theta_d * (self.BRDF_SAMPLING_RES_PHI_D // 2) +
               ind_theta_h * (self.BRDF_SAMPLING_RES_PHI_D // 2) * self.BRDF_SAMPLING_RES_THETA_D)
        #print(f"ind_phi: {ind_phi}, ind_theta_d: {ind_theta_d}, ind_theta_h: {ind_theta_h}")
        #print(f"ind: {ind}")
        
        # Convert material_id to tensor if needed
        if not isinstance(material_id, torch.Tensor):
            material_id = torch.tensor(material_id, device=self.device, dtype=torch.long)
        
        # Extract R, G, B from separate channel blocks (lines 205-207)
        # Data layout for each material: [R_0...R_n, G_0...G_n, B_0...B_n]
        n = self.BRDF_SAMPLING_RES_THETA_H * self.BRDF_SAMPLING_RES_THETA_D * (self.BRDF_SAMPLING_RES_PHI_D // 2)
        
        # Index into brdf_data using material_id
        red_val = self.brdf_data[material_id, ind] * self.RED_SCALE
        green_val = self.brdf_data[material_id, ind + n] * self.GREEN_SCALE
        blue_val = self.brdf_data[material_id, ind + 2 * n] * self.BLUE_SCALE
        
        rgb = torch.stack([red_val, green_val, blue_val], dim=-1)
        
        # Check for below-horizon values (lines 208-209)
        if (rgb < 0).any():
            print("Below horizon.", file=sys.stderr)
        
        return rgb
    
    def lookup_wiwo(self, wi, wo, material_id):
        """
        Look up BRDF values for given incoming/outgoing direction vectors.
        
        This is a wrapper around lookup() that converts direction vectors to angles.
        
        Args:
            wi: [B, 3] incoming light directions (normalized, pointing toward surface)
            wo: [B, 3] outgoing view directions (normalized, pointing away from surface)
        
        Returns:
            rgb: [B, 3] BRDF RGB values
        """
        # Normalize directions
        wi = F.normalize(wi, dim=-1)
        wo = F.normalize(wo, dim=-1)
        
        # Convert wi to spherical coordinates
        theta_in = torch.acos(torch.clamp(wi[..., 2], -1.0, 1.0))
        phi_in = torch.atan2(wi[..., 1], wi[..., 0])
        
        # Convert wo to spherical coordinates
        theta_out = torch.acos(torch.clamp(wo[..., 2], -1.0, 1.0))
        phi_out = torch.atan2(wo[..., 1], wo[..., 0])
        
        # Call the main lookup function
        return self.lookup(theta_in, phi_in, theta_out, phi_out, material_id)
    
    def lookup_batched(self, wi, wo, batch_size=10000):
        """
        Look up BRDF values in batches to save memory.
        
        Args:
            wi: [N, 3] incoming light directions
            wo: [N, 3] outgoing view directions
            batch_size: Number of lookups per batch
        
        Returns:
            rgb: [N, 3] BRDF RGB values
        """
        n_total = wi.shape[0]
        rgb_results = []
        
        for i in range(0, n_total, batch_size):
            end_i = min(i + batch_size, n_total)
            wi_batch = wi[i:end_i]
            wo_batch = wo[i:end_i]
            rgb_batch = self.lookup_wiwo(wi_batch, wo_batch)
            rgb_results.append(rgb_batch)
        
        return torch.cat(rgb_results, dim=0)
    
    def get_brdf_value(self, theta_in, phi_in, theta_out, phi_out):
        """
        Look up BRDF value from spherical angles (convenience function).
        
        Args:
            theta_in: Polar angle of incoming direction [0, pi/2]
            phi_in: Azimuthal angle of incoming direction [-pi, pi]
            theta_out: Polar angle of outgoing direction [0, pi/2]
            phi_out: Azimuthal angle of outgoing direction [-pi, pi]
        
        Returns:
            rgb: [3] BRDF RGB value
        """
        # Convert angles to Cartesian directions
        wi_x = torch.sin(theta_in) * torch.cos(phi_in)
        wi_y = torch.sin(theta_in) * torch.sin(phi_in)
        wi_z = torch.cos(theta_in)
        wi = torch.stack([wi_x, wi_y, wi_z], dim=-1).unsqueeze(0)
        
        wo_x = torch.sin(theta_out) * torch.cos(phi_out)
        wo_y = torch.sin(theta_out) * torch.sin(phi_out)
        wo_z = torch.cos(theta_out)
        wo = torch.stack([wo_x, wo_y, wo_z], dim=-1).unsqueeze(0)
        
        # Move to device
        wi = wi.to(self.device)
        wo = wo.to(self.device)
        
        # Lookup
        rgb = self.lookup(wi, wo)
        
        return rgb.squeeze(0)

    def rangles_to_rvectors(self, theta_h, theta_d, phi_d):
        """
        Convert Rusinkiewicz angles to direction vectors.
        
        Args:
            theta_h: half vector elevation angle (tensor or scalar)
            theta_d: diff vector elevation angle (tensor or scalar)
            phi_d: diff vector azimuthal angle (tensor or scalar)
        
        Returns:
            Tensor of shape [..., 6] containing [hx, hy, hz, dx, dy, dz]
        """
        
        hx = torch.sin(theta_h) * torch.cos(torch.zeros_like(theta_h))
        hy = torch.sin(theta_h) * torch.sin(torch.zeros_like(theta_h))
        hz = torch.cos(theta_h)
        dx = torch.sin(theta_d) * torch.cos(phi_d)
        dy = torch.sin(theta_d) * torch.sin(phi_d)
        dz = torch.cos(theta_d)
        
        return torch.stack([hx, hy, hz, dx, dy, dz], dim=-1)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python MERL.py <path_to_brdf.binary>")
        print("\nExample:")
        print("  python MERL.py /path/to/alum-bronze.binary")
        sys.exit(1)
    
    brdf_path = sys.argv[1]
    
    # Initialize interface
    merl = MERLInterface(brdf_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test lookup with normal incidence
    print("\n" + "="*70)
    print("Testing BRDF Lookup")
    print("="*70)
    
    # Normal incidence
    wi = torch.tensor([[0.0, 0.0, 1.0]], device=merl.device)
    wo = torch.tensor([[0.0, 0.0, 1.0]], device=merl.device)
    rgb = merl.lookup(wi, wo)
    print(f"\nNormal incidence (θ_i=0, θ_o=0):")
    print(f"  RGB = [{rgb[0,0]:.6f}, {rgb[0,1]:.6f}, {rgb[0,2]:.6f}]")
    
    # 45 degree incidence
    angle = math.pi / 4
    wi = torch.tensor([[math.sin(angle), 0.0, math.cos(angle)]], device=merl.device)
    wo = torch.tensor([[0.0, 0.0, 1.0]], device=merl.device)
    rgb = merl.lookup(wi, wo)
    print(f"\n45° incidence (θ_i=45°, θ_o=0°):")
    print(f"  RGB = [{rgb[0,0]:.6f}, {rgb[0,1]:.6f}, {rgb[0,2]:.6f}]")
    
    # Batch test
    print(f"\nBatch lookup test (1000 random directions):")
    n_test = 1000
    wi_batch = torch.randn(n_test, 3, device=merl.device)
    wi_batch[..., 2] = torch.abs(wi_batch[..., 2])  # Above surface
    wo_batch = torch.randn(n_test, 3, device=merl.device)
    wo_batch[..., 2] = torch.abs(wo_batch[..., 2])  # Above surface
    
    rgb_batch = merl.lookup(wi_batch, wo_batch)
    print(f"  Mean RGB: [{rgb_batch[:, 0].mean():.6f}, {rgb_batch[:, 1].mean():.6f}, {rgb_batch[:, 2].mean():.6f}]")
    print(f"  Valid samples: {(rgb_batch >= 0).all(dim=1).sum().item()}/{n_test}")
    
    print("\n" + "="*70)




class MerlTorch:
    """
    PyTorch version of Merl class with multi-material support.
    
    Supports multiple materials and batched operations on GPU/CPU.
    All operations use PyTorch tensors for efficient computation.
    
    Args:
        merl_files: List of paths to .binary BRDF files, or a single file path
        device: Device to store tensors ('cuda' or 'cpu')
    """
    sampling_theta_h = 90
    sampling_theta_d = 90
    sampling_phi_d = 180
    
    scale = torch.tensor([1. / 1500, 1.15 / 1500, 1.66 / 1500])
    
    def __init__(self, merl_files, device='cuda'):
        """
        Initialize and load MERL BRDF file(s).
        
        Args:
            merl_files: Path to a single .binary file, or list of paths to multiple files
            device: Device to store tensors on ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        self.scale = self.scale.to(self.device)
        
        # Handle single file or list of files
        if isinstance(merl_files, (str, Path)):
            merl_files = [merl_files]
        
        self.material_names = []
        self.n_materials = len(merl_files)
        
        # Load all BRDF data
        brdf_list = []
        for merl_file in merl_files:
            merl_file = Path(merl_file)
            self.material_names.append(merl_file.stem)
            
            with open(merl_file, 'rb') as f:
                data = f.read()
                n = struct.unpack_from('3i', data)
                
                # Update sampling_phi_d if needed (for first material)
                if len(brdf_list) == 0:
                    self.sampling_phi_d = n[2]
                
                length = self.sampling_theta_h * self.sampling_theta_d * self.sampling_phi_d
                if n[0] * n[1] * n[2] != length:
                    raise IOError(f"Dimensions do not match in {merl_file}")
                
                brdf_data = struct.unpack_from(str(3 * length) + 'd', data, offset=struct.calcsize('3i'))
                brdf_np = np.array(brdf_data)
                brdf_list.append(brdf_np)
        
        # Stack all materials: shape (n_materials, 3 * length)
        self.brdf = torch.from_numpy(np.stack(brdf_list, axis=0)).float().to(self.device)
        
        print(f"[MerlTorch] Loaded {self.n_materials} materials")
        print(f"[MerlTorch] Device: {self.device}")
        print(f"[MerlTorch] BRDF shape: {self.brdf.shape}")
        print(f"[MerlTorch] Materials: {', '.join(self.material_names)}")
    
    def eval_raw(self, theta_h, theta_d, phi_d, material_id):
        """
        Lookup the BRDF value for given half diff coordinates.
        
        Args:
            theta_h: half vector elevation angle in radians (tensor or scalar)
            theta_d: diff vector elevation angle in radians (tensor or scalar)
            phi_d: diff vector azimuthal angle in radians (tensor or scalar)
            material_id: material index (tensor or scalar), shape should match other inputs
        
        Returns:
            BRDF values [R, G, B] in linear RGB (tensor with shape [..., 3])
        """
        return self._eval_idx(
            self._theta_h_idx(theta_h),
            self._theta_d_idx(theta_d),
            self._phi_d_idx(phi_d),
            material_id
        )
    
    def _filter_phi_d(self, phi_d):
        """Filter phi_d to valid range based on sampling resolution."""
        if self.sampling_phi_d == 180:
            phi_d = torch.where(phi_d <= 0, phi_d + math.pi, phi_d)
        elif self.sampling_phi_d == 360:
            phi_d = torch.where(phi_d >= 2 * math.pi, phi_d - 2 * math.pi, phi_d)
            phi_d = torch.where(phi_d < 0, phi_d + 2 * math.pi, phi_d)
        return phi_d
    
    def eval_interp(self, theta_h, theta_d, phi_d, material_id):
        """
        Lookup the BRDF value for given half diff coordinates with trilinear interpolation.
        
        Args:
            theta_h: half vector elevation angle in radians (tensor or scalar)
            theta_d: diff vector elevation angle in radians (tensor or scalar)
            phi_d: diff vector azimuthal angle in radians (tensor or scalar)
            material_id: material index (tensor or scalar), shape [...] matches angle inputs
        
        Returns:
            BRDF values [R, G, B] in linear RGB (tensor with shape [..., 3])
        """
        # Convert scalars to tensors
        if not isinstance(theta_h, torch.Tensor):
            theta_h = torch.tensor(theta_h, device=self.device, dtype=torch.float32)
        if not isinstance(theta_d, torch.Tensor):
            theta_d = torch.tensor(theta_d, device=self.device, dtype=torch.float32)
        if not isinstance(phi_d, torch.Tensor):
            phi_d = torch.tensor(phi_d, device=self.device, dtype=torch.float32)
        if not isinstance(material_id, torch.Tensor):
            material_id = torch.tensor(material_id, device=self.device, dtype=torch.long)
        
        original_shape = theta_h.shape
        
        # Flatten all inputs for batch processing
        theta_h = theta_h.reshape(-1)
        theta_d = theta_d.reshape(-1)
        phi_d = phi_d.reshape(-1)
        material_id = material_id.reshape(-1)
        
        phi_d = self._filter_phi_d(phi_d)
        
        idx_th_p = self._theta_h_idx(theta_h)
        idx_td_p = self._theta_d_idx(theta_d)
        idx_pd_p = self._phi_d_idx(phi_d)
        
        # Calculate the indexes for interpolation
        idx_th_p = torch.where(idx_th_p < self.sampling_theta_h - 1, idx_th_p, 
                               torch.tensor(self.sampling_theta_h - 2, device=self.device))
        idx_td_p = torch.where(idx_td_p < self.sampling_theta_d - 1, idx_td_p,
                               torch.tensor(self.sampling_theta_d - 2, device=self.device))
        
        # Get neighboring indices
        idx_th_0, idx_th_1 = idx_th_p, idx_th_p + 1
        idx_td_0, idx_td_1 = idx_td_p, idx_td_p + 1
        idx_pd_0, idx_pd_1 = idx_pd_p, idx_pd_p + 1
        
        # Calculate the weights
        th_0 = self._theta_h_from_idx(idx_th_0)
        th_1 = self._theta_h_from_idx(idx_th_1)
        td_0 = self._theta_d_from_idx(idx_td_0)
        td_1 = self._theta_d_from_idx(idx_td_1)
        pd_0 = self._phi_d_from_idx(idx_pd_0)
        pd_1 = self._phi_d_from_idx(idx_pd_1)
        
        weight_th_0 = torch.abs(th_1 - theta_h)
        weight_th_1 = torch.abs(theta_h - th_0)
        weight_td_0 = torch.abs(td_1 - theta_d)
        weight_td_1 = torch.abs(theta_d - td_0)
        weight_pd_0 = torch.abs(pd_1 - phi_d)
        weight_pd_1 = torch.abs(phi_d - pd_0)
        
        # Normalize weights
        weight_sum_th = weight_th_0 + weight_th_1
        weight_sum_td = weight_td_0 + weight_td_1
        weight_sum_pd = weight_pd_0 + weight_pd_1
        
        # Avoid division by zero
        weight_sum_th = torch.where(weight_sum_th > 0, weight_sum_th, torch.ones_like(weight_sum_th))
        weight_sum_td = torch.where(weight_sum_td > 0, weight_sum_td, torch.ones_like(weight_sum_td))
        weight_sum_pd = torch.where(weight_sum_pd > 0, weight_sum_pd, torch.ones_like(weight_sum_pd))
        
        weight_th_0 = weight_th_0 / weight_sum_th
        weight_th_1 = weight_th_1 / weight_sum_th
        weight_td_0 = weight_td_0 / weight_sum_td
        weight_td_1 = weight_td_1 / weight_sum_td
        weight_pd_0 = weight_pd_0 / weight_sum_pd
        weight_pd_1 = weight_pd_1 / weight_sum_pd
        
        # Handle phi wrapping
        idx_pd_1 = torch.where(idx_pd_1 < self.sampling_phi_d, idx_pd_1, 
                               torch.zeros_like(idx_pd_1))
        
        # Trilinear interpolation
        ret_val = torch.zeros(theta_h.shape[0], 3, device=self.device)
        
        for ith, wth in [(idx_th_0, weight_th_0), (idx_th_1, weight_th_1)]:
            for itd, wtd in [(idx_td_0, weight_td_0), (idx_td_1, weight_td_1)]:
                for ipd, wpd in [(idx_pd_0, weight_pd_0), (idx_pd_1, weight_pd_1)]:
                    val = self._eval_idx(ith, itd, ipd, material_id)
                    weight = (wth * wtd * wpd).unsqueeze(-1)
                    ret_val = ret_val + val * weight
        
        # Reshape back to original shape
        ret_val = ret_val.reshape(*original_shape, 3)
        
        return ret_val
    
    def _eval_idx(self, ith, itd, ipd, material_id):
        """
        Lookup the BRDF value for a given set of indexes.
        
        Args:
            ith: theta_h index (tensor)
            itd: theta_d index (tensor)
            ipd: phi_d index (tensor)
            material_id: material index (tensor)
        
        Returns:
            BRDF values [R, G, B] in linear RGB (tensor with shape [..., 3])
        """
        ind = ipd + self.sampling_phi_d * (itd + ith * self.sampling_theta_d)
        
        stride = self.sampling_theta_h * self.sampling_theta_d * self.sampling_phi_d
        
        # Get RGB values for each color channel
        # brdf shape: (n_materials, 3 * stride)
        # Layout: [R_0...R_n, G_0...G_n, B_0...B_n]
        
        ret = []
        for color in range(3):
            color_ind = ind + color * stride
            # Index into brdf: material_id selects material, color_ind selects value
            val = self.brdf[material_id, color_ind] * self.scale[color]
            ret.append(val)
        
        return torch.stack(ret, dim=-1)
    
    def _theta_h_from_idx(self, theta_h_idx):
        """
        Get the theta_h value corresponding to a given index.
        
        Args:
            theta_h_idx: Index for theta_h (tensor or scalar)
        
        Returns:
            A theta_h value in radians (tensor or scalar)
        """
        if not isinstance(theta_h_idx, torch.Tensor):
            theta_h_idx = torch.tensor(theta_h_idx, device=self.device, dtype=torch.float32)
        
        ret_val = theta_h_idx.float() / self.sampling_theta_h
        return ret_val * ret_val * math.pi / 2
    
    def _theta_h_idx(self, theta_h):
        """
        Get the index corresponding to a given theta_h value.
        
        Args:
            theta_h: Value for theta_h in radians (tensor or scalar)
        
        Returns:
            The corresponding index for the given theta_h (tensor)
        """
        if not isinstance(theta_h, torch.Tensor):
            theta_h = torch.tensor(theta_h, device=self.device, dtype=torch.float32)
        
        th = self.sampling_theta_h * torch.sqrt(theta_h / (math.pi / 2))
        floorth = torch.floor(th)
        return torch.clamp(floorth, 0, self.sampling_theta_h - 1).long()
    
    def _theta_d_from_idx(self, theta_d_idx):
        """
        Get the theta_d value corresponding to a given index.
        
        Args:
            theta_d_idx: Index for theta_d (tensor or scalar)
        
        Returns:
            A theta_d value in radians (tensor or scalar)
        """
        if not isinstance(theta_d_idx, torch.Tensor):
            theta_d_idx = torch.tensor(theta_d_idx, device=self.device, dtype=torch.float32)
        
        return theta_d_idx.float() / self.sampling_theta_d * math.pi / 2
    
    def _theta_d_idx(self, theta_d):
        """
        Get the index corresponding to a given theta_d value.
        
        Args:
            theta_d: Value for theta_d in radians (tensor or scalar)
        
        Returns:
            The corresponding index for the given theta_d (tensor)
        """
        if not isinstance(theta_d, torch.Tensor):
            theta_d = torch.tensor(theta_d, device=self.device, dtype=torch.float32)
        
        floortd = torch.floor(self.sampling_theta_d * theta_d / (math.pi / 2))
        return torch.clamp(floortd, 0, self.sampling_theta_d - 1).long()
    
    def _phi_d_from_idx(self, phi_d_idx):
        """
        Get the phi_d value corresponding to a given index.
        
        Args:
            phi_d_idx: Index for phi_d (tensor or scalar)
        
        Returns:
            A phi_d value in radians (tensor or scalar)
        """
        if not isinstance(phi_d_idx, torch.Tensor):
            phi_d_idx = torch.tensor(phi_d_idx, device=self.device, dtype=torch.float32)
        
        if self.sampling_phi_d == 180:
            return phi_d_idx.float() / self.sampling_phi_d * math.pi
        elif self.sampling_phi_d == 360:
            return phi_d_idx.float() / self.sampling_phi_d * math.pi * 2
    
    def _phi_d_idx(self, phi_d):
        """
        Get the index corresponding to a given phi_d value.
        
        Args:
            phi_d: Value for phi_d in radians (tensor or scalar)
        
        Returns:
            The corresponding index for the given phi_d (tensor)
        """
        if not isinstance(phi_d, torch.Tensor):
            phi_d = torch.tensor(phi_d, device=self.device, dtype=torch.float32)
        
        phi_d = self._filter_phi_d(phi_d)
        
        if self.sampling_phi_d == 180:
            floorpd = torch.floor(self.sampling_phi_d * phi_d / math.pi)
            return torch.clamp(floorpd, 0, self.sampling_phi_d - 1).long()
        elif self.sampling_phi_d == 360:
            floorpd = torch.floor(self.sampling_phi_d * phi_d / (2 * math.pi))
            return torch.clamp(floorpd, 0, self.sampling_phi_d - 1).long()
    
    