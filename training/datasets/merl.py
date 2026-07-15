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
    
