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
import random
import struct
import glob
from datasets.MERLInterface import MERLInterface
from utils.ops import rotate_to_canonical_frame


def _load_merl_material_list(cfg, split):
    """Read the MERL material-name list from cfg.data.training_list_path.

    Both ``train`` and ``val`` splits return the same training material set.
    This is intentional: MERL is trained as an auto-decoder, where each
    material has its own learned latent in ``point_latent_bank``. Validating
    on disjoint test materials would only query randomly-initialised latent
    slots (no gradients ever flow there), producing meaningless loss. The
    ``test_list.txt`` sibling file is kept on disk for downstream / external
    use (e.g. zero-shot eval with externally fitted latents).

    The ``split`` parameter is kept for signature symmetry and future use.
    Returns ``None`` when the key is unset, so callers fall back to loading
    every ``.binary`` file in the dataset folder.
    """
    del split  # both splits use the training list — see docstring
    list_path = getattr(cfg.data, 'training_list_path', None)
    if not list_path:
        return None
    with open(list_path, 'r') as f:
        names = [ln.strip() for ln in f if ln.strip()]
    if not names:
        raise ValueError(f"MERL material list at {list_path} is empty")
    return names


class MERLBRDFIterableDataset(IterableDataset):
    """
    Iterable dataset for MERL BRDF data using MERLInterface.
    
    In each iteration, samples material_id, wi, wo and calculates rgb using lookup_wiwo.
    
    Args:
        data_folder: Path to folder containing .binary BRDF files
        batch_size: Number of samples per batch (default: 1024)
        split: 'train' or 'val' (default: 'train')
        material_names: Optional list of specific materials to load (without .binary extension)
        device: Device to store tensors on (default: 'cuda' if available, else 'cpu')
    """
    
    def __init__(self, cfg, data_folder, batch_size=1024, split='train', material_names=None, device=None):
        self.cfg = cfg
        self.data_folder = Path(data_folder)
        self.batch_size = batch_size
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.split = split

        if material_names is None:
            material_names = _load_merl_material_list(cfg, split)

        print(f"\n{'='*60}")
        print(f"Loading MERL BRDF Dataset")
        print(f"{'='*60}")
        print(f"Data folder: {data_folder}")
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size}")
        print(f"Split: {split}")
        print(f"Material filter: {len(material_names) if material_names is not None else 'all'} materials")

        # Initialize MERLInterface
        self.merl_interface = MERLInterface(str(self.data_folder), device=self.device,
                                            material_names=material_names)

        # Get number of materials
        self.n_materials = len(self.merl_interface.material_names)

        print(f"Loaded {self.n_materials} materials")
        print(f"{'='*60}\n")



    def _sample_direction(self, batch_size):
        """
        Sample random direction vectors in the upper hemisphere (z > 0).
        
        Args:
            batch_size: Number of directions to sample
            
        Returns:
            directions: [batch_size, 3] normalized direction vectors
        """
        # Sample uniform on sphere, then keep only upper hemisphere
        directions = torch.randn(batch_size, 3, device=self.device)
        directions = NF.normalize(directions, dim=-1)
        # Ensure z > 0 (upper hemisphere)
        directions[..., 2] = torch.abs(directions[..., 2])
        # Renormalize to ensure unit length
        directions = NF.normalize(directions, dim=-1)
        return directions
    
    def __iter__(self):
        """Iterator yielding batches of material_id, wi, wo, and rgb."""
        if self.split == 'train':
            while True:
                # Sample random material IDs
                if self.cfg.model.stage == 2:
                    material_id = torch.full((self.batch_size,), 0, dtype=torch.long, device=self.device)
                else:
                    material_id = torch.randint(0, self.n_materials, (self.batch_size,), device=self.device)
                
                # Sample random incoming (wi) and outgoing (wo) directions
                wi = self._sample_direction(self.batch_size)
                wo = self._sample_direction(self.batch_size)

                wi,wo=rotate_to_canonical_frame(wi,wo)
                
                # Calculate rgb using lookup_wiwo
                rgb,_,_,_ = self.merl_interface.lookup_wiwo(wi, wo, material_id)
                
                yield {
                    'material_id': material_id,
                    'wi': wi,
                    'wo': wo,
                    'rgb': rgb,
                }
        else:
            # For validation, yield a single batch
            if self.cfg.model.stage == 2:
                material_id = torch.full((self.batch_size,), 0, dtype=torch.long, device=self.device)
            else:
                material_id = torch.randint(0, self.n_materials, (self.batch_size,), device=self.device)
                
            wi = self._sample_direction(self.batch_size)
            wo = self._sample_direction(self.batch_size)
            wi,wo=rotate_to_canonical_frame(wi,wo)

            rgb,_,_,_ = self.merl_interface.lookup_wiwo(wi, wo, material_id)
            
            yield {
                'material_id': material_id,
                'wi': wi,
                'wo': wo,
                'rgb': rgb,
            }

class MERLBRDFFixedDataset(IterableDataset):
    """
    Iterable dataset for MERL BRDF data using MERLInterface.
    
    In each iteration, samples material_id, wi, wo and calculates rgb using lookup_wiwo.
    
    Args:
        data_folder: Path to folder containing .binary BRDF files
        batch_size: Number of samples per batch (default: 1024)
        split: 'train' or 'val' (default: 'train')
        material_names: Optional list of specific materials to load (without .binary extension)
        device: Device to store tensors on (default: 'cuda' if available, else 'cpu')
    """
    
    def __init__(self, cfg, data_folder, batch_size=1024, split='train', material_names=None, device=None):
        self.cfg = cfg
        self.data_folder = Path(data_folder)
        self.batch_size = batch_size
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.split = split

        if material_names is None:
            material_names = _load_merl_material_list(cfg, split)

        print(f"\n{'='*60}")
        print(f"Loading MERL BRDF Dataset")
        print(f"{'='*60}")
        print(f"Data folder: {data_folder}")
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size}")
        print(f"Split: {split}")
        print(f"Material filter: {len(material_names) if material_names is not None else 'all'} materials")

        # Initialize MERLInterface
        self.merl_interface = MERLInterface(str(self.data_folder), device=self.device,
                                            material_names=material_names)

        # Get number of materials
        self.n_materials = len(self.merl_interface.material_names)

        print(f"Loaded {self.n_materials} materials")
        print(f"{'='*60}\n")
        self._generate_samples()
    
    def _sample_direction(self, batch_size):
        """
        Sample random direction vectors in the upper hemisphere (z > 0).
        
        Args:
            batch_size: Number of directions to sample
            
        Returns:
            directions: [batch_size, 3] normalized direction vectors
        """
        # Sample uniform on sphere, then keep only upper hemisphere
        directions = torch.randn(batch_size, 3, device=self.device)
        directions = NF.normalize(directions, dim=-1)
        # Ensure z > 0 (upper hemisphere)
        directions[..., 2] = torch.abs(directions[..., 2])
        # Renormalize to ensure unit length
        directions = NF.normalize(directions, dim=-1)
        return directions

    def _generate_samples(self):
        self.wi = self._sample_direction(self.batch_size)
        self.wo = self._sample_direction(self.batch_size)
        self.wi,self.wo=rotate_to_canonical_frame(self.wi,self.wo)
        
        # Calculate rgb using lookup_wiwo
        self.material_id = torch.full((self.batch_size,), 0, dtype=torch.long, device=self.device)
        self.rgb,_,_,_ = self.merl_interface.lookup_wiwo(self.wi, self.wo, self.material_id)
        
    
    def __iter__(self):
        """Iterator yielding batches of material_id, wi, wo, and rgb."""
        if self.split == 'train':
            while True:
                yield {
                    'material_id': self.material_id,
                    'wi': self.wi,
                    'wo': self.wo,
                    'rgb': self.rgb,
                }
        else:
            
            yield {
                'material_id': self.material_id,
                'wi': self.wi,
                'wo': self.wo,
                'rgb': self.rgb,
            }
    
class MERLBRDFFixedDataset_hd(IterableDataset):
    """
    Fixed dataset for MERL BRDF data using Rusinkiewicz parameterization.
    
    Randomly generates n (theta_h, theta_d, phi_d) angle sets and queries MERL BRDF to get RGB values.
    All data is stored in CUDA memory for fast training.
    Returns half vector, difference vector, BRDF value, and material_id.
    
    Args:
        cfg: Configuration object
        data_folder: Path to folder containing .binary BRDF files
        n_samples: Number of samples to generate and store (default: 1000000)
        material_id: Which material to use (default: 3)
        split: 'train' or 'val' (default: 'train')
        device: Device to store tensors on (default: 'cuda' if available, else 'cpu')
    """
    
    def __init__(self, cfg, data_folder, n_samples=1000000, material_id=0, split='train', device=None):
        self.cfg = cfg
        self.data_folder = Path(data_folder)
        self.n_samples = n_samples
        self.batch_size = n_samples
        self.material_id = material_id
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.split = split

        material_names = _load_merl_material_list(cfg, split)

        print(f"\n{'='*60}")
        print(f"Loading Fixed MERL BRDF Dataset (Half-Diff Parameterization) ({split})")
        print(f"{'='*60}")
        print(f"Data folder: {data_folder}")
        print(f"Device: {self.device}")
        print(f"Material ID: {material_id}")
        print(f"Number of samples: {n_samples:,}")
        print(f"Split: {split}")
        print(f"Material filter: {len(material_names) if material_names is not None else 'all'} materials")

        # Initialize MERLInterface
        self.merl_interface = MERLInterface(str(self.data_folder), device=self.device,
                                            material_names=material_names)
        
        # Generate and store fixed samples
        print(f"Generating {n_samples:,} random BRDF samples using Rusinkiewicz angles...")
        self._generate_samples()
        
    
    def _sample_direction(self, batch_size):
        """
        Sample random direction vectors in the upper hemisphere (z > 0).
        
        Args:
            batch_size: Number of directions to sample
            
        Returns:
            directions: [batch_size, 3] normalized direction vectors
        """
        # Sample uniform on sphere, then keep only upper hemisphere
        directions = torch.randn(batch_size, 3, device=self.device)
        directions = NF.normalize(directions, dim=-1)
        # Ensure z > 0 (upper hemisphere)
        directions[..., 2] = torch.abs(directions[..., 2])
        # Renormalize to ensure unit length
        directions = NF.normalize(directions, dim=-1)
        return directions
    
        # Calculate rotation angle: negative of wi's azimuthal angle
        # phi_wi = atan2(wi.y, wi.x)
        # We want to rotate by -phi_wi to make wi.x = 0
        phi_wi = torch.atan2(wi[:, 1], wi[:, 0])  # [B]
        
        # Rotation matrix around z-axis by angle -phi_wi:
        # [cos(-phi)  -sin(-phi)  0]   [cos(phi)   sin(phi)  0]
        # [sin(-phi)   cos(-phi)  0] = [-sin(phi)  cos(phi)  0]
        # [   0           0       1]   [   0          0      1]
        
        cos_phi = torch.cos(phi_wi)  # [B]
        sin_phi = torch.sin(phi_wi)  # [B]
        
        # Apply rotation to wi
        wi_rotated = torch.zeros_like(wi)
        wi_rotated[:, 0] = cos_phi * wi[:, 0] + sin_phi * wi[:, 1]
        wi_rotated[:, 1] = -sin_phi * wi[:, 0] + cos_phi * wi[:, 1]
        wi_rotated[:, 2] = wi[:, 2]
        
        # Apply same rotation to wo
        wo_rotated = torch.zeros_like(wo)
        wo_rotated[:, 0] = cos_phi * wo[:, 0] + sin_phi * wo[:, 1]
        wo_rotated[:, 1] = -sin_phi * wo[:, 0] + cos_phi * wo[:, 1]
        wo_rotated[:, 2] = wo[:, 2]
        
        return wi_rotated, wo_rotated
    
    def sample_rusinkiewicz_angles(self, batch_size):
        """
        Directly sample Rusinkiewicz half-difference angles.
        
        This samples the MERL parameterization directly instead of converting from wi/wo.
        Provides more direct control over the half-difference parameter space.
        
        Args:
            batch_size: Number of angle sets to sample
        
        Returns:
            theta_h: [batch_size] half vector elevation angle [0, π/2]
            theta_d: [batch_size] difference vector elevation angle [0, π/2]
            phi_d: [batch_size] difference vector azimuthal angle [0, 2π]
        """
        import math
        
        # Sample theta_h uniformly in [0, π/2]
        theta_h = torch.rand(batch_size, device=self.device) * (math.pi / 2)
        
        # Sample theta_d uniformly in [0, π/2]
        theta_d = torch.rand(batch_size, device=self.device) * (math.pi / 2)
        
        # Sample phi_d uniformly in [0, 2π]
        phi_d = torch.rand(batch_size, device=self.device) * (math.pi * 2.0)
        
        return theta_h, theta_d, phi_d

    def rusinkiewicz_to_vectors(self,theta_h, theta_d, phi_d):
        """
        使用 PyTorch 将 Rusinkiewicz 坐标转换为 wi 和 wo 矢量。
        支持批处理 (Batch processing)。
        
        参数:
            theta_h (Tensor): 半矢量与法线夹角，形状为 (...)
            theta_d (Tensor): 入射光与半矢量夹角，形状为 (...)
            phi_d   (Tensor): 入射光绕半矢量的方位角，形状为 (...)
            
        返回:
            wi, wo (Tensor): 形状为 (..., 3) 的单位矢量
        """
        # 1. 计算三角函数
        sh, ch = torch.sin(theta_h), torch.cos(theta_h)
        sd, cd = torch.sin(theta_d), torch.cos(theta_d)
        spd, cpd = torch.sin(phi_d), torch.cos(phi_d)

        # 2. 在半矢量局部空间 (H-frame) 构建 wi 和 wo
        # 此时 H = [0, 0, 1]
        # wi_local = [sin(td)cos(pd), sin(td)sin(pd), cos(td)]
        wi_x_local = sd * cpd
        wi_y_local = sd * spd
        wi_z_local = cd

        # wo 是 wi 绕 H(Z轴) 旋转180度，即 x,y 取反
        wo_x_local = -wi_x_local
        wo_y_local = -wi_y_local
        wo_z_local = wi_z_local

        # 3. 将局部坐标旋转到法线坐标系 (绕 Y 轴旋转 theta_h)
        # 旋转公式:
        # x' = x*cos(th) + z*sin(th)
        # y' = y
        # z' = -x*sin(th) + z*cos(th)
        
        wi_x = wi_x_local * ch + wi_z_local * sh
        wi_y = wi_y_local
        wi_z = -wi_x_local * sh + wi_z_local * ch

        wo_x = wo_x_local * ch + wo_z_local * sh
        wo_y = wo_y_local
        wo_z = -wo_x_local * sh + wo_z_local * ch

        # 4. 拼接成 (..., 3) 形状的张量
        wi = torch.stack([wi_x, wi_y, wi_z], dim=-1)
        wo = torch.stack([wo_x, wo_y, wo_z], dim=-1)

        return wi, wo
    
    '''
    def _generate_samples_wiwo(self):
        """Generate all samples at once and store in memory."""
        # Sample random Rusinkiewicz angles
        wi=self._sample_direction(self.batch_size)
        wo=self._sample_direction(self.batch_size)
        
        self.wi=wi
        self.wo=wo
        wi_rotated, wo_rotated = rotate_to_canonical_frame(wi, wo)
        # Calculate BRDF values using lookup_angle
        
        # Create material_id tensor
        material_ids = torch.full((self.n_samples,), self.material_id, dtype=torch.long, device=self.device)
        
        self.rgb,phi_d,theta_d,theta_h = self.merl_interface.lookup_wiwo(wi, wo, material_ids)
        self.rgb_rotated,phi_d_rotated,theta_d_rotated,theta_h_rotated = self.merl_interface.lookup_wiwo(wi_rotated, wo_rotated, material_ids)

        print("rgb_diff",torch.mean(self.rgb - self.rgb_rotated))
        print("phi_d_diff",torch.mean(phi_d - phi_d_rotated))
        print("theta_d_diff",torch.mean(theta_d - theta_d_rotated))
        print("theta_h_diff",torch.mean(theta_h - theta_h_rotated))
        
        print("phi_d",torch.max(phi_d),torch.min(phi_d))
        print("theta_d",torch.max(theta_d),torch.min(theta_d))
        print("theta_h",torch.max(theta_h),torch.min(theta_h))
        
        # Convert angles to vectors using rangles_to_rvectors
        # Returns [hx, hy, hz, dx, dy, dz] with shape [batch_size, 6]
        vectors = self.merl_interface.rangles_to_rvectors(theta_h, theta_d, phi_d)
        
        # Split into h_vec and d_vec
        self.h_vec = vectors[:, :3]  # [batch_size, 3]
        self.d_vec = vectors[:, 3:]  # [batch_size, 3]
        
        #h_norms = torch.linalg.norm(self.h_vec, dim=1)
        #print(f"Norms of h_vec: {h_norms}")
        
        # Store material_ids for consistency
        self.material_ids = material_ids
        self.material_ids_train = torch.full((self.n_samples,), 0, dtype=torch.long, device=self.device)
    '''
    
    def _generate_samples(self):
        """
        Generate samples by directly sampling Rusinkiewicz angles.
        
        Alternative to sampling wi/wo directions - samples the half-difference
        parameterization directly for more uniform coverage of BRDF space.
        """
        # Directly sample Rusinkiewicz angles
        theta_h, theta_d, phi_d = self.sample_rusinkiewicz_angles(self.batch_size)
        
        # Create material_id tensor
        material_ids = torch.full((self.n_samples,), self.material_id, dtype=torch.long, device=self.device)
        
        # Calculate BRDF values directly using lookup_angle
        self.rgb = self.merl_interface.lookup_angle(phi_d, theta_d, theta_h, material_ids)
        self.rgb=torch.clamp(self.rgb, min=0.0)
        print("Direct angle sampling:")
        print("  phi_d:", torch.max(phi_d).item(), torch.min(phi_d).item())
        print("  theta_d:", torch.max(theta_d).item(), torch.min(theta_d).item())
        print("  theta_h:", torch.max(theta_h).item(), torch.min(theta_h).item())
        
        self.wi,self.wo=self.rusinkiewicz_to_vectors(theta_h, theta_d, phi_d)
        self.wi,self.wo=rotate_to_canonical_frame(self.wi,self.wo)
        
        
        # Store material_ids for consistency
        self.material_ids = material_ids
        self.material_ids_train = torch.full((self.n_samples,), 0, dtype=torch.long, device=self.device)
    
    def __iter__(self):
        """Iterator yielding batches of material_id, wi, wo, and rgb."""
        if self.split == 'train':
            while True:
                yield {
                    'material_id': self.material_ids,
                    'wi': self.wi,
                    'wo': self.wo,
                    'rgb': self.rgb,
                }
        else:
            
            yield {
                'material_id': self.material_ids,
                'wi': self.wi,
                'wo': self.wo,
                'rgb': self.rgb,
            }


class MERLBRDFIterableDataset_hd(IterableDataset):
    """
    Iterable dataset for MERL BRDF data using Rusinkiewicz parameterization.
    
    In each iteration, samples theta_h, theta_d, phi_d and calculates BRDF values using lookup_angle.
    Returns half vector, difference vector, BRDF value, and material_id.
    
    Args:
        data_folder: Path to folder containing .binary BRDF files
        batch_size: Number of samples per batch (default: 1024)
        split: 'train' or 'val' (default: 'train')
        material_names: Optional list of specific materials to load (without .binary extension)
        device: Device to store tensors on (default: 'cuda' if available, else 'cpu')
    """
    
    def __init__(self, cfg, data_folder, batch_size=1024, split='train', material_names=None, device=None):
        self.cfg = cfg
        self.data_folder = Path(data_folder)
        self.batch_size = batch_size
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.split = split

        if material_names is None:
            material_names = _load_merl_material_list(cfg, split)

        print(f"\n{'='*60}")
        print(f"Loading MERL BRDF Dataset (Half-Diff Parameterization)")
        print(f"{'='*60}")
        print(f"Data folder: {data_folder}")
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size}")
        print(f"Split: {split}")
        print(f"Material filter: {len(material_names) if material_names is not None else 'all'} materials")

        # Initialize MERLInterface
        self.merl_interface = MERLInterface(str(self.data_folder), device=self.device,
                                            material_names=material_names)

        # Get number of materials
        self.n_materials = len(self.merl_interface.material_names)
        
        print(f"Loaded {self.n_materials} materials")
        print(f"{'='*60}\n")
    
    def sample_rusinkiewicz_angles(self, batch_size):
        """
        Directly sample Rusinkiewicz half-difference angles.
        
        This samples the MERL parameterization directly instead of converting from wi/wo.
        Provides more direct control over the half-difference parameter space.
        
        Args:
            batch_size: Number of angle sets to sample
        
        Returns:
            theta_h: [batch_size] half vector elevation angle [0, π/2]
            theta_d: [batch_size] difference vector elevation angle [0, π/2]
            phi_d: [batch_size] difference vector azimuthal angle [0, 2π]
        """
        import math
        
        # Sample theta_h uniformly in [0, π/2]
        theta_h = torch.rand(batch_size, device=self.device) * (math.pi / 2)
        
        # Sample theta_d uniformly in [0, π/2]
        theta_d = torch.rand(batch_size, device=self.device) * (math.pi / 2)
        
        # Sample phi_d uniformly in [0, 2π]
        phi_d = torch.rand(batch_size, device=self.device) * (math.pi * 2.0)
        
        return theta_h, theta_d, phi_d


    def _sample_direction(self, batch_size):
        """
        Sample random direction vectors in the upper hemisphere (z > 0).
        
        Args:
            batch_size: Number of directions to sample
            
        Returns:
            directions: [batch_size, 3] normalized direction vectors
        """
        # Sample uniform on sphere, then keep only upper hemisphere
        directions = torch.randn(batch_size, 3, device=self.device)
        directions = NF.normalize(directions, dim=-1)
        # Ensure z > 0 (upper hemisphere)
        directions[..., 2] = torch.abs(directions[..., 2])
        # Renormalize to ensure unit length
        directions = NF.normalize(directions, dim=-1)
        return directions

    def rusinkiewicz_to_vectors(self,theta_h, theta_d, phi_d):
        """
        使用 PyTorch 将 Rusinkiewicz 坐标转换为 wi 和 wo 矢量。
        支持批处理 (Batch processing)。
        
        参数:
            theta_h (Tensor): 半矢量与法线夹角，形状为 (...)
            theta_d (Tensor): 入射光与半矢量夹角，形状为 (...)
            phi_d   (Tensor): 入射光绕半矢量的方位角，形状为 (...)
            
        返回:
            wi, wo (Tensor): 形状为 (..., 3) 的单位矢量
        """
        # 1. 计算三角函数
        sh, ch = torch.sin(theta_h), torch.cos(theta_h)
        sd, cd = torch.sin(theta_d), torch.cos(theta_d)
        spd, cpd = torch.sin(phi_d), torch.cos(phi_d)

        # 2. 在半矢量局部空间 (H-frame) 构建 wi 和 wo
        # 此时 H = [0, 0, 1]
        # wi_local = [sin(td)cos(pd), sin(td)sin(pd), cos(td)]
        wi_x_local = sd * cpd
        wi_y_local = sd * spd
        wi_z_local = cd

        # wo 是 wi 绕 H(Z轴) 旋转180度，即 x,y 取反
        wo_x_local = -wi_x_local
        wo_y_local = -wi_y_local
        wo_z_local = wi_z_local

        # 3. 将局部坐标旋转到法线坐标系 (绕 Y 轴旋转 theta_h)
        # 旋转公式:
        # x' = x*cos(th) + z*sin(th)
        # y' = y
        # z' = -x*sin(th) + z*cos(th)
        
        wi_x = wi_x_local * ch + wi_z_local * sh
        wi_y = wi_y_local
        wi_z = -wi_x_local * sh + wi_z_local * ch

        wo_x = wo_x_local * ch + wo_z_local * sh
        wo_y = wo_y_local
        wo_z = -wo_x_local * sh + wo_z_local * ch

        # 4. 拼接成 (..., 3) 形状的张量
        wi = torch.stack([wi_x, wi_y, wi_z], dim=-1)
        wo = torch.stack([wo_x, wo_y, wo_z], dim=-1)

        return wi, wo
    
    def __iter__(self):
        if self.split == 'train':
            while True:
                # Sample random material IDs
                if self.cfg.model.stage == 2:
                    material_id = torch.full((self.batch_size,), 0, dtype=torch.long, device=self.device)
                else:
                    material_id = torch.randint(0, self.n_materials, (self.batch_size,), device=self.device)
                
                # Sample random Rusinkiewicz angles
                theta_h, theta_d, phi_d = self.sample_rusinkiewicz_angles(self.batch_size)
                rgb = self.merl_interface.lookup_angle(phi_d, theta_d, theta_h, material_id)
                rgb=torch.clamp(rgb, min=0.0)

                wi,wo=self.rusinkiewicz_to_vectors(theta_h, theta_d, phi_d)
                wi,wo=rotate_to_canonical_frame(wi,wo)
                
                yield {
                    'material_id': material_id,
                    'wi': wi,
                    'wo': wo,
                    'rgb': rgb,
                }
        else:
            if self.cfg.model.stage == 2:
                material_id = torch.full((self.batch_size,), 0, dtype=torch.long, device=self.device)
            else:
                material_id = torch.randint(0, self.n_materials, (self.batch_size,), device=self.device)
            
            # Sample random Rusinkiewicz angles
            theta_h, theta_d, phi_d = self.sample_rusinkiewicz_angles(self.batch_size)
            rgb = self.merl_interface.lookup_angle(phi_d, theta_d, theta_h, material_id)
            rgb=torch.clamp(rgb, min=0.0)
            wi,wo=self.rusinkiewicz_to_vectors(theta_h, theta_d, phi_d)
            wi,wo=rotate_to_canonical_frame(wi,wo)
            yield {
                'material_id': material_id,
                'wi': wi,
                'wo': wo,
                'rgb': rgb,
            }
    '''
    def __iter__wiwo(self):
        """Iterator yielding batches of h_vec, d_vec, rgb, and material_id."""
        if self.split == 'train':
            while True:
                # Sample random material IDs
                if self.cfg.model.stage == 2:
                    material_id = torch.full((self.batch_size,), 0, dtype=torch.long, device=self.device)
                else:
                    material_id = torch.randint(0, self.n_materials, (self.batch_size,), device=self.device)
                
                # Sample random Rusinkiewicz angles
                wi=self._sample_direction(self.batch_size)
                wo=self._sample_direction(self.batch_size)
                
                # Calculate BRDF values using lookup_angle
                rgb,phi_d,theta_d,theta_h = self.merl_interface.lookup_wiwo(wi, wo, material_id)
                
                # Convert angles to vectors using rangles_to_rvectors
                # Returns [hx, hy, hz, dx, dy, dz] with shape [batch_size, 6]
                vectors = self.merl_interface.rangles_to_rvectors(theta_h, theta_d, phi_d)
                
                # Split into h_vec and d_vec
                h_vec = vectors[:, :3]  # [batch_size, 3]
                d_vec = vectors[:, 3:]  # [batch_size, 3]
                
                yield {
                    'material_id': material_id,
                    'wi': h_vec,
                    'wo': d_vec,
                    'rgb': rgb,
                }
        else:
            # For validation, yield a single batch
            if self.cfg.model.stage == 2:
                material_id = torch.full((self.batch_size,), 0, dtype=torch.long, device=self.device)
            else:
                material_id = torch.randint(0, self.n_materials, (self.batch_size,), device=self.device)
            
            # Sample random Rusinkiewicz angles
            wi=self._sample_direction(self.batch_size)
            wo=self._sample_direction(self.batch_size)
            
            # Calculate BRDF values using lookup_angle
            rgb,phi_d,theta_d,theta_h = self.merl_interface.lookup_wiwo(wi, wo, material_id)
            
            # Convert angles to vectors
            vectors = self.merl_interface.rangles_to_rvectors(theta_h, theta_d, phi_d)
            h_vec = vectors[:, :3]
            d_vec = vectors[:, 3:]
            
            yield {
                'material_id': material_id,
                'wi': h_vec,
                'wo': d_vec,
                'rgb': rgb,
            }
    '''