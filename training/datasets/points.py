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
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.ops import rotate_to_canonical_frame


def _dense_pool_load_worker(args):
    """ProcessPoolExecutor worker: load one material in a subprocess.

    Mirrors bonn.py's `_pool_load_worker`. Builds a stub
    MultiMaterialDenseDataset (bypassing __init__) with only the attributes
    `_load_single_material` reads, then delegates to it. Keeps the existing
    loader logic as the single source of truth — no duplicated code.
    """
    (material_folder_str, material_id, is_val, val_ratio,
     filter_observations, filter_center, filter_half_width,
     filter_half_length, point_subsample_ratio) = args
    stub = MultiMaterialDenseDataset.__new__(MultiMaterialDenseDataset)
    stub.split = 'val' if is_val else 'train'
    stub.val_ratio = val_ratio
    stub.filter_observations = filter_observations
    stub.filter_center = filter_center
    stub.filter_half_width = filter_half_width
    stub.filter_half_length = filter_half_length
    stub.point_subsample_ratio = point_subsample_ratio
    return stub._load_single_material(Path(material_folder_str), material_id)

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

def get_ray_directions_for_pixels(pixel_coords, focal, cx, cy, distortion):
    """
    Get camera ray directions for specific pixel coordinates with radial distortion correction.
    
    Args:
        pixel_coords: (N, 2) tensor of [u, v] pixel coordinates
        focal: focal length
        cx, cy: principal point coordinates
        distortion: radial distortion coefficient k1
        
    Returns:
        directions: (N, 3) tensor of ray directions in camera space
    """
    u = pixel_coords[:, 0]  # (N,)
    v = pixel_coords[:, 1]  # (N,)
    
    # Convert to normalized coordinates relative to principal point
    x_norm = (u - cx) / focal
    y_norm = (v - cy) / focal
    
    # Apply radial distortion correction
    r_squared = x_norm**2 + y_norm**2
    distortion_factor = 1 + distortion * r_squared
    
    x_corrected = x_norm * distortion_factor
    y_corrected = y_norm * distortion_factor
    
    # Stack into direction vectors (OpenGL convention: -y, -z)
    directions = torch.stack([x_corrected, -y_corrected, -torch.ones_like(u)], dim=-1)  # (N, 3)
    
    return directions

class MultiMaterialDenseDataset(IterableDataset):
    """
    Memory-efficient dataset that loads ALL observations into RAM from structured NPZ files.
    No chunk switching — all data is available every iteration.

    Each material folder must contain:
      - observations_structured.npz with:
          xyz        (V, 3)    float32   — 3D point positions
          point_ids  (V,)      int32     — local point IDs
          rgbs       (K, V, 3) uint16    — raw sensor RGB (CCM applied at sample time)
          cam_pos    (K, 3)    float32   — camera positions
          light_pos  (K, 3)    float32   — light positions
      - scan_log.json          — for emitter_id lookup
      - rotated_camera.json    — for full c2w matrices (rotation + position)

    Memory layout (per material ~1.2 GB, dominated by rgbs):
      - Per-material: rgbs (K,V,3) kept as numpy uint16 (NOT flattened)
      - Per-material: xyz (V,3) float32, point_ids (V,) int32
      - Flat sampling index: (N_total, 3) int32 storing (mat_local_idx, k, v)
      - Lookup tables: cam_pos, emitter_ids per material
      - At sample time: index into dense arrays, compute rays + CCM vectorized

    Memory per material: ~1.2 GB (uint16 rgbs) + ~6 MB (xyz, point_ids, metadata)
    Sampling index: ~12 bytes per valid observation (3 × int32)
    """

    def __init__(self, cfg, root_folder, split='train', share_from=None):
        """
        Args:
            cfg: configuration object
            root_folder: path to folder containing material subfolders
            split: 'train' or 'val'
            share_from: optional MultiMaterialDenseDataset to reuse already-loaded
                numpy arrays from (typically the train instance). When provided,
                this instance does not re-read any npz files; it only recomputes
                mat_valid_k, mat_num_valid_obs, and mat_weights for its split.
        """
        self.cfg = cfg
        self.root_folder = root_folder
        self.split = split
        self.rays_num = cfg.data.rays_num

        # Color correction matrix (applied at sample time)
        self.ccm = torch.from_numpy(np.array(cfg.data.ccm)).double()  # (3, 3) float64

        # XY filter
        self.filter_observations = getattr(cfg.data, 'filter_observations', True)
        rect_cfg = cfg.renderer.mesh.rectangle
        self.filter_center = rect_cfg.center
        self.filter_half_width = rect_cfg.width / 4
        self.filter_half_length = rect_cfg.length / 4

        # Train/val split
        self.val_ratio = getattr(cfg.data, 'val_ratio', 0.1)

        # Point subsampling: keep a random fraction of points per material.
        # Seeded by material_id below so train/val instances pick the same subset
        # even when built independently (without share_from).
        self.point_subsample_ratio = float(getattr(cfg.data, 'point_subsample_ratio', 1.0))

        # Parallel material loading. 0 = serial; >0 = ProcessPoolExecutor with
        # this many workers (mirrors BonnDataset.num_load_workers).
        self.num_load_workers = int(getattr(cfg.data, 'num_load_workers', 0))

        # Read training list
        self.training_list_path = cfg.data.training_list_path
        self.training_list = []
        with open(self.training_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.training_list.append(int(line))

        # Optional: "pretend the bad↔backup material swap never happened" for
        # continue-training from a pre-swap checkpoint. For each swapped slot,
        # read data from the swap partner's folder (which physically holds this
        # slot's pre-swap data) so num_points and per-material offsets stay
        # identical to what the saved checkpoint expects.
        self.legacy_swap_indexing = bool(getattr(cfg.data, 'legacy_swap_indexing', False))
        self.swap_partner = {}
        if self.legacy_swap_indexing:
            import json as _json
            rl_path = getattr(cfg.data, 'replace_list_path', None) or str(Path(root_folder) / 'replace_list.json')
            if os.path.exists(rl_path):
                with open(rl_path) as _rl_f:
                    _rl = _json.load(_rl_f)
                for _r in _rl.get('records', []):
                    if _r.get('backup_id') is None or _r.get('replaces') is None:
                        continue
                    self.swap_partner[int(_r['backup_id'])] = int(_r['replaces'])
                    self.swap_partner[int(_r['replaces'])] = int(_r['backup_id'])
                _affected = sum(1 for m in self.training_list if m in self.swap_partner)
                print(f"[legacy_swap_indexing] enabled: {_affected}/{len(self.training_list)} training-list slots remap to partner folder (replace_list={rl_path})")
            else:
                print(f"[legacy_swap_indexing] enabled but replace_list.json not found at {rl_path}; no remap applied")

        # When a slot is swap-paired AND legacy mode is on, read data from the
        # partner's folder. material_id (used for downstream indexing) stays as
        # the training_list slot id — see _load_all_data which uses
        # self.training_list[i] instead of folder.name.
        self.material_folders = [
            Path(root_folder) / str(self.swap_partner.get(mid, mid))
            for mid in self.training_list
        ]

        print(f"\n{'='*60}")
        print(f"Loading MultiMaterial Dense Dataset ({split})")
        print(f"{'='*60}")
        print(f"Materials: {len(self.training_list)}")

        if share_from is not None:
            self._share_from(share_from)
        else:
            self._load_all_data()

        total_obs = sum(self.mat_num_valid_obs)
        # Memory estimate: dense arrays + lookup tables (no sample index!)
        dense_mem = sum(r.nbytes for r in self.mat_rgbs) + sum(x.nbytes for x in self.mat_xyz)
        lookup_mem = self.cam_positions.nelement() * self.cam_positions.element_size()
        total_mem = dense_mem + lookup_mem
        print(f"\nDataset ready! Total valid observations: {total_obs:,}")
        if share_from is not None:
            print(f"Sharing dense arrays with parent dataset (no extra RAM)")
        else:
            print(f"Dense data: {dense_mem / 1e9:.2f} GB, "
                  f"Lookup tables: {lookup_mem / 1e6:.1f} MB, "
                  f"Total: {total_mem / 1e9:.2f} GB")
        print(f"Using rejection sampling (no sample index)")
        print(f"{'='*60}\n")

    def _share_from(self, parent):
        """Reuse already-loaded numpy arrays from a parent dataset.

        Dense arrays (rgbs, xyz, point_ids, ...) are shared by reference — no copy.
        Only mat_valid_k, mat_num_valid_obs, and mat_weights are recomputed for
        this instance's split.
        """
        # Share dense arrays and lookup tables by reference (no copy)
        self.mat_rgbs = parent.mat_rgbs
        self.mat_xyz = parent.mat_xyz
        self.mat_point_ids = parent.mat_point_ids
        self.mat_emitter_lookup = parent.mat_emitter_lookup
        self.mat_material_ids = parent.mat_material_ids
        self.mat_valid_v = parent.mat_valid_v
        self.cam_positions = parent.cam_positions

        is_val = (self.split == 'val')

        # Recompute valid_k and observation counts for this split
        self.mat_valid_k = []
        self.mat_num_valid_obs = []
        for m in range(len(self.mat_rgbs)):
            rgbs_dense = self.mat_rgbs[m]
            K = rgbs_dense.shape[0]
            valid_v = self.mat_valid_v[m]

            split_idx = int(K * (1 - self.val_ratio))
            if is_val:
                valid_k = np.arange(split_idx, K)
            else:
                valid_k = np.arange(K)

            rgbs_sub = rgbs_dense[np.ix_(valid_k, valid_v)]
            valid_mask = rgbs_sub.sum(axis=2) > 0
            N_obs = int(valid_mask.sum())

            self.mat_valid_k.append(valid_k)
            self.mat_num_valid_obs.append(N_obs)

        # Material sampling weights — materials with zero valid obs get zero weight
        obs_arr = np.array(self.mat_num_valid_obs, dtype=np.float64)
        total_valid = obs_arr.sum()
        if total_valid > 0:
            self.mat_weights = obs_arr / total_valid
        else:
            # Pathological: no valid observations at all in this split.
            self.mat_weights = np.ones(len(obs_arr)) / max(len(obs_arr), 1)

    def _load_single_material(self, material_folder, material_id):
        """Load one material's structured NPZ + metadata.

        Returns a dict with all per-material arrays the aggregator needs, or
        ``None`` if the material should be skipped (missing file, no valid
        points/observations after filters). Pure read + numpy work so the
        function is safe to invoke from a ProcessPoolExecutor worker.
        """
        is_val = (self.split == 'val')
        structured_path = material_folder / 'observations_structured.npz'

        if not structured_path.exists():
            print(f"  Warning: {structured_path} not found, skipping material {material_id}")
            return None

        data = np.load(structured_path)
        xyz = data['xyz']                  # (V, 3) float32
        point_ids_np = data['point_ids']   # (V,) int32
        rgbs_dense = data['rgbs']          # (K, V, 3) uint16

        K, V, _ = rgbs_dense.shape

        scan_log_path = str(material_folder / "scan_log.json")
        camera_json_path = str(material_folder / "rotated_camera.json")

        metadata_list, _, _ = load_camera_turntable_light_metadata(scan_log_path)
        camera_metadata = load_camera_metadata(camera_json_path)

        sorted_metadata = sorted(metadata_list, key=lambda x: int(x['overall_id']))
        emitter_lookup = np.array(
            [int(entry['emitter_id']) for entry in sorted_metadata], dtype=np.int32)

        cam_pos_array = np.zeros((K, 3), dtype=np.float32)
        for cam_id_str, cam_info in camera_metadata.items():
            cam_id = int(cam_id_str)
            if cam_id < K:
                position = np.array(cam_info['position'], dtype=np.float32)
                rotation_matrix = np.array(cam_info['rotation_matrix'])
                c2w = build_4x4(rotation_matrix, position)
                cam_pos_array[cam_id] = c2w[:3, 3]

        if self.filter_observations:
            point_mask = ((np.abs(xyz[:, 0] - self.filter_center[0]) <= self.filter_half_width) &
                          (np.abs(xyz[:, 1] - self.filter_center[1]) <= self.filter_half_length))
            valid_v = np.where(point_mask)[0]
        else:
            valid_v = np.arange(V)

        if len(valid_v) == 0:
            return None

        if self.point_subsample_ratio < 1.0:
            N_sub = max(1, int(len(valid_v) * self.point_subsample_ratio))
            rng = np.random.default_rng(material_id)
            sel = np.sort(rng.choice(len(valid_v), size=N_sub, replace=False))
            valid_v = valid_v[sel]

            rgbs_dense = rgbs_dense[:, valid_v, :]
            xyz = xyz[valid_v]
            point_ids_np = np.arange(len(valid_v), dtype=np.int32)
            valid_v = np.arange(len(valid_v))

        split_idx = int(K * (1 - self.val_ratio))
        if is_val:
            valid_k = np.arange(split_idx, K)
        else:
            valid_k = np.arange(K)

        rgbs_sub = rgbs_dense[np.ix_(valid_k, valid_v)]
        valid_mask = rgbs_sub.sum(axis=2) > 0
        N_obs = int(valid_mask.sum())

        if N_obs == 0:
            return None

        density = float(valid_mask.mean())

        print(f"  Material {material_id}: {N_obs:,} valid obs "
              f"(K={len(valid_k)}, V_filtered={len(valid_v)}, "
              f"density={density*100:.1f}%, "
              f"rgbs={rgbs_dense.nbytes/1e9:.2f} GB)")

        return {
            'material_id': material_id,
            'rgbs_dense': rgbs_dense,
            'xyz': xyz,
            'point_ids_np': point_ids_np,
            'emitter_lookup': emitter_lookup,
            'valid_v': valid_v,
            'valid_k': valid_k,
            'N_obs': N_obs,
            'cam_pos_array': cam_pos_array,
            'K': K,
        }

    def _load_materials_parallel(self, tasks):
        """Load materials in parallel using ProcessPoolExecutor.

        Returns results in the same order as ``tasks`` (matching the serial
        loader). Failed loads (worker raised) are dropped with a warning;
        worker-returned ``None`` results are also dropped.
        """
        n_workers = min(self.num_load_workers, len(tasks))
        print(f"[{self.split}] Parallel load: {len(tasks)} materials, "
              f"{n_workers} workers (ProcessPoolExecutor)")

        is_val = (self.split == 'val')
        worker_args = [
            (str(material_folder), material_id, is_val, self.val_ratio,
             self.filter_observations, self.filter_center,
             self.filter_half_width, self.filter_half_length,
             self.point_subsample_ratio)
            for material_folder, material_id in tasks
        ]

        results = {}
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            future_to_mid = {
                ex.submit(_dense_pool_load_worker, wa): wa[1]
                for wa in worker_args
            }
            for fut in tqdm(as_completed(future_to_mid),
                            total=len(future_to_mid),
                            desc=f"Loading {self.split} (parallel)"):
                mid = future_to_mid[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    print(f"  [Warning] material {mid} worker raised: {exc}")
                    res = None
                if res is not None:
                    results[mid] = res

        # Preserve task order
        return [results[mid] for _, mid in tasks if mid in results]

    def _load_all_data(self):
        """Load structured NPZ data. Keeps rgbs in dense numpy uint16 format."""
        # Per-material dense storage (kept as numpy for memory efficiency)
        self.mat_rgbs = []       # list of (K, V, 3) numpy uint16
        self.mat_xyz = []        # list of (V, 3) numpy float32
        self.mat_point_ids = []  # list of (V,) numpy int32
        self.mat_emitter_lookup = []  # list of (K,) numpy int32
        self.mat_material_ids = []    # list of int (material_id)
        self.mat_valid_v = []    # list of numpy array (filtered point indices)
        self.mat_valid_k = []    # list of numpy array (valid camera indices for split)
        self.mat_num_valid_obs = []  # list of int (count of valid observations per material)

        cam_pos_list = []  # list of (K, 3) numpy float32
        max_K = 0

        # (folder, material_id) tasks; material_id uses training_list slot id so
        # legacy_swap_indexing remapping doesn't change ids seen downstream.
        tasks = [
            (folder, self.training_list[i])
            for i, folder in enumerate(self.material_folders)
        ]

        if self.num_load_workers > 0 and len(tasks) > 1:
            mat_results = self._load_materials_parallel(tasks)
        else:
            mat_results = []
            for folder, mid in tqdm(tasks, desc=f"Loading {self.split} data"):
                res = self._load_single_material(folder, mid)
                if res is not None:
                    mat_results.append(res)

        for res in mat_results:
            self.mat_rgbs.append(res['rgbs_dense'])
            self.mat_xyz.append(res['xyz'])
            self.mat_point_ids.append(res['point_ids_np'])
            self.mat_emitter_lookup.append(res['emitter_lookup'])
            self.mat_material_ids.append(res['material_id'])
            self.mat_valid_v.append(res['valid_v'])
            self.mat_valid_k.append(res['valid_k'])
            self.mat_num_valid_obs.append(res['N_obs'])
            cam_pos_list.append(res['cam_pos_array'])
            max_K = max(max_K, res['K'])

        # Precompute material sampling weights (proportional to valid obs count)
        total_valid = sum(self.mat_num_valid_obs)
        self.mat_weights = np.array(self.mat_num_valid_obs, dtype=np.float64) / total_valid

        # Camera position lookup table: (num_materials, max_K, 3)
        num_materials = len(cam_pos_list)
        self.cam_positions = torch.zeros(num_materials, max_K, 3, dtype=torch.float32)
        for i, cp in enumerate(cam_pos_list):
            self.cam_positions[i, :cp.shape[0]] = torch.from_numpy(cp)

    def _sample_batch_rejection(self, N):
        """
        Sample N valid observations using rejection sampling.
        Randomly picks (mat, k, v) tuples, rejects invalid ones (rgbs sum == 0),
        repeats until N valid samples are collected. Fully vectorized per round.

        Returns:
            dict with rays, rgbs, xyz, emitter_ids, camera_ids, material_ids, point_ids
        """
        num_mats = len(self.mat_rgbs)

        # Collect valid samples across rounds
        collected_xyz = []
        collected_rgbs = []
        collected_point_ids = []
        collected_emitter_ids = []
        collected_camera_ids = []
        collected_material_ids = []
        collected_cam_pos = []
        num_collected = 0

        while num_collected < N:
            remaining = N - num_collected
            # Oversample by 1.3x to account for invalid entries
            n_try = int(remaining * 1.3) + 64

            # Sample materials weighted by valid obs count
            mat_samples = np.random.choice(num_mats, size=n_try, p=self.mat_weights)

            # For each material, sample random (k, v) from valid_k x valid_v
            for m in range(num_mats):
                m_mask = mat_samples == m
                n_m = int(m_mask.sum())
                if n_m == 0:
                    continue

                valid_k = self.mat_valid_k[m]
                valid_v = self.mat_valid_v[m]
                k_rand = valid_k[np.random.randint(0, len(valid_k), size=n_m)]
                v_rand = valid_v[np.random.randint(0, len(valid_v), size=n_m)]

                # Rejection: check rgbs sum > 0
                rgbs_sampled = self.mat_rgbs[m][k_rand, v_rand]  # (n_m, 3) uint16
                valid = rgbs_sampled.sum(axis=1) > 0
                if not valid.any():
                    continue

                k_valid = k_rand[valid]
                v_valid = v_rand[valid]

                collected_xyz.append(self.mat_xyz[m][v_valid])
                collected_rgbs.append(rgbs_sampled[valid].astype(np.float64))
                collected_point_ids.append(self.mat_point_ids[m][v_valid].astype(np.int64))
                collected_emitter_ids.append(self.mat_emitter_lookup[m][k_valid].astype(np.int64))
                collected_camera_ids.append(k_valid.astype(np.int64))
                collected_material_ids.append(np.full(int(valid.sum()), self.mat_material_ids[m], dtype=np.int64))
                collected_cam_pos.append(self.cam_positions[m, k_valid].numpy())
                num_collected += int(valid.sum())

        # Concatenate and trim to exactly N
        xyz_out = torch.from_numpy(np.concatenate(collected_xyz, axis=0)[:N])
        rgbs_raw = torch.from_numpy(np.concatenate(collected_rgbs, axis=0)[:N])
        point_ids_out = torch.from_numpy(np.concatenate(collected_point_ids, axis=0)[:N])
        emitter_ids_out = torch.from_numpy(np.concatenate(collected_emitter_ids, axis=0)[:N])
        camera_ids_out = torch.from_numpy(np.concatenate(collected_camera_ids, axis=0)[:N])
        material_ids_out = torch.from_numpy(np.concatenate(collected_material_ids, axis=0)[:N])
        cam_pos = torch.from_numpy(np.concatenate(collected_cam_pos, axis=0)[:N])

        # Vectorized ray computation
        rays_d = xyz_out - cam_pos
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        rays = torch.cat([cam_pos, rays_d], dim=-1)  # (N, 6)

        # Apply CCM (float64 @ ccm, clip, float32)
        rgbs = (rgbs_raw @ self.ccm).clamp(min=0.0).float()  # (N, 3)

        return {
            'rays': rays,
            'rgbs': rgbs,
            'xyz': xyz_out,
            'emitter_ids': emitter_ids_out,
            'camera_ids': camera_ids_out,
            'material_ids': material_ids_out,
            'point_ids': point_ids_out,
            'gt_params': torch.zeros(1),
        }

    def set_step(self, step: int):
        """No-op for compatibility with training loop."""
        pass

    def __len__(self):
        total_obs = sum(self.mat_num_valid_obs)
        if self.split == 'val':
            return math.ceil(total_obs / self.rays_num)
        return 1000000

    def __iter__(self):
        if self.split == 'train':
            while True:
                yield self._sample_batch_rejection(self.rays_num)
        else:
            # Validation: iterate all valid observations deterministically
            total_obs = sum(self.mat_num_valid_obs)
            num_batches = math.ceil(total_obs / self.rays_num)
            for _ in range(num_batches):
                yield self._sample_batch_rejection(self.rays_num)


class MultiMaterialPointDataset(IterableDataset if True else Dataset):
    """
    Dataset for multiple materials loaded from point observations.
    Each material has its own folder containing:
    - hdr/ (images, not loaded)
    - scan_log.json (emitter metadata)
    - rotated_camera.json (camera poses)
    - point_metadata.json (contains num_points for this material)
    - observations/ (chunked observation files: observations_chunk_00.npz, observations_chunk_01.npz, ...)
      OR sparse/observations.npz (legacy single file format)

    Each observation chunk contains: [x, y, z, image_id, pixel_x, pixel_y, r, g, b, point_id]
    where point_id is the LOCAL point index (0 to num_points-1) within this material.

    This class uses double buffering to randomly load chunks per material in the background,
    keeping memory usage constant while providing fresh data every N iterations.
    Supports both training and validation splits from the same data.
    """
    
    # ------------------------------
    # Embedded helper classes
    # ------------------------------
    @dataclass
    class ChunkData:
        """Container for loaded chunk data."""
        rays: torch.Tensor
        rgbs: torch.Tensor
        xyz: torch.Tensor
        camera_ids: torch.Tensor
        emitter_ids: torch.Tensor
        material_ids: torch.Tensor
        point_ids: torch.Tensor  # LOCAL point IDs (per-material)
    
    class _DoubleBuffer:
        """Two RAM slots with a background thread that fills the inactive slot."""
        def __init__(self, build_chunk_fn):
            self.build_chunk_fn = build_chunk_fn           # fn()->ChunkData
            self.slots = [None, None]
            self.ready = [threading.Event(), threading.Event()]
            self.active = 0
            self._stop = False
            self._q = queue.Queue(maxsize=2)
            self._t = threading.Thread(target=self._worker, daemon=True)
            self._t.start()
        
        def _worker(self):
            while not self._stop:
                try:
                    slot_id = self._q.get(timeout=0.1)
                except queue.Empty:
                    continue
                data = self.build_chunk_fn()  # Build new chunk
                self.slots[slot_id] = data
                self.ready[slot_id].set()
        
        def request_fill(self, slot_id):
            self.ready[slot_id].clear()
            try:
                self._q.put_nowait(slot_id)
            except queue.Full:
                # Drop oldest request to keep moving
                try:
                    _ = self._q.get_nowait()
                except queue.Empty:
                    pass
                self._q.put_nowait(slot_id)
        
        def wait_initial(self):
            self.ready[self.active].wait()
        
        def try_swap(self):
            nxt = 1 - self.active
            if self.ready[nxt].is_set():
                old = self.active
                self.active = nxt
                self.ready[old].clear()  # Clear old slot's ready flag
                return True
            return False
        
        def current(self):
            self.ready[self.active].wait()
            return self.slots[self.active]
        
        def stop(self):
            self._stop = True
            self._t.join(timeout=1.0)
    
    def __init__(self, cfg, root_folder, split='train'):
        """
        Args:
            cfg: configuration object
            root_folder: path to folder containing material subfolders (0, 1, 2, ...)
            split: 'train' or 'val'
        """
        self.cfg = cfg
        self.root_folder = root_folder
        self.split = split
        self.rays_num = cfg.data.rays_num
        
        # Camera intrinsics
        self.intrinsics = cfg.renderer.camera.intrinsics
        self.focal = self.intrinsics['focal_length']
        self.cx = self.intrinsics['cx']
        self.cy = self.intrinsics['cy']
        self.distortion = self.intrinsics['distortion']
        self.img_hw = (self.intrinsics['height'], self.intrinsics['width'])
        
        # Color correction matrix
        self.ccm = np.array(cfg.data.ccm)
        
        # XY filter bounds from mesh.rectangle config (filter to half the region)
        self.filter_observations = getattr(cfg.data, 'filter_observations', True)
        rect_cfg = cfg.renderer.mesh.rectangle
        self.filter_center = rect_cfg.center  # [x, y, z]
        # Half of the original width/length gives the new region dimensions
        self.filter_half_width = rect_cfg.width / 4  # half of (width/2)
        self.filter_half_length = rect_cfg.length / 4  # half of (length/2)
        print(f"XY filter enabled: center=({self.filter_center[0]:.4f}, {self.filter_center[1]:.4f}), "
              f"half_width={self.filter_half_width:.4f}, half_length={self.filter_half_length:.4f}")
        
        # Train/val split ratio
        self.val_ratio = getattr(cfg.data, 'val_ratio', 0.1)
        
        # Double buffer settings
        self.switch_iters = getattr(cfg.data, 'switch_iters', 1000)  # How often to reload chunks
        self.chunk_size = getattr(cfg.data, 'chunk_size', 200)  # Number of materials to sample per chunk
        self.step = 0
        
        # Read training list from txt file
        self.training_list_path = cfg.data.training_list_path
        self.training_list = []
        with open(self.training_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    self.training_list.append(int(line))
        
        # Build material folders from training list
        self.material_folders = [Path(root_folder) / str(mid) for mid in self.training_list]
        
        print(f"\n{'='*60}")
        print(f"Loading MultiMaterial Dataset ({split})")
        print(f"{'='*60}")
        print(f"Training list path: {self.training_list_path}")
        print(f"Loaded {len(self.training_list)} materials: {self.training_list}")
        
        # Discover chunks and load metadata (lightweight)
        self._discover_chunks_and_metadata()
        
        # Initialize based on split
        if split == 'train':
            print(f"\nInitializing double buffer (chunk reload every {self.switch_iters} iters)...")
            self._dbuf = MultiMaterialPointDataset._DoubleBuffer(lambda: self._load_chunks(split='train')   )
            # Prefill two | 91745/335183748ctive) and slot 1 (next)
            self._dbuf.request_fill(0)
            self._dbuf.request_fill(1)
            self._dbuf.wait_initial()  # Ensure first active chunk exists
            print("Double buffer initialized!")
        else:
            # For validation, load all validation observations directly
            print(f"\nLoading validation data...")
            chunk_data = self._load_chunks(split='val', load_all=True)
            self.all_rays = chunk_data.rays
            self.all_rgbs = chunk_data.rgbs
            self.all_xyz = chunk_data.xyz
            self.all_camera_ids = chunk_data.camera_ids
            self.all_emitter_ids = chunk_data.emitter_ids
            self.all_material_ids = chunk_data.material_ids
            self.all_point_ids = chunk_data.point_ids
            print(f"Validation data loaded: {len(self.all_rays):,} observations")
            
            # Debug visualization: show points with material_id == 0
            visualize = False
            if visualize:
                import open3d as o3d
                mask = self.all_material_ids == 102
                xyz = self.all_xyz[mask].cpu().numpy()
                # Convert int16 RGB to float [0, 1] for open3d
                rgbs = np.clip(self.all_rgbs[mask].cpu().numpy().astype(np.float32), 0, 65535) / 65535.0
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)
                pcd.colors = o3d.utility.Vector3dVector(rgbs)
                o3d.io.write_point_cloud("/media/raid/cloth/output/visualizaitons/material_102_points.ply", pcd)
                print(f"Saved debug point cloud to material_102_points.ply ({len(xyz)} points)")
        
        print(f"\nDataset ready!")
        print(f"{'='*60}\n")
    
    def _discover_chunks_and_metadata(self):
        """Discover available chunks and load metadata (camera/emitter lookups) for all materials."""
        self.material_chunks = {}  # material_id -> list of chunk file paths
        self.camera_lookups = {}   # material_id -> {camera_id (int) -> c2w matrix}
        self.emitter_lookups = {}  # material_id -> tensor of emitter_ids indexed by overall_id
        
        print("\nDiscovering chunks and loading metadata...")
        for material_folder in tqdm(self.material_folders, desc="Scanning materials"):
            material_id = int(material_folder.name)
            
            # Discover chunk files in observations/ folder
            obs_folder = material_folder / "observations"
            chunk_files = sorted(obs_folder.glob("observations_chunk_*.npz"))
            self.material_chunks[material_id] = chunk_files
            print(f"  Material {material_id}: Found {len(chunk_files)} chunks in {obs_folder}")
            
            # Load metadata using existing utility functions (like real.py does)
            scan_log_path = str(material_folder / "scan_log.json")
            camera_json_path = str(material_folder / "rotated_camera.json")
            
            # load_camera_turntable_light_metadata returns:
            #   metadata: list of dicts with ['overall_id', 'camera_id', 'emitter_id', 'filename', 'turn_angle']
            #   camera_metadata: not used here (we use rotated_camera.json instead)
            #   (position already in meters)
            metadata_list, _, _ = load_camera_turntable_light_metadata(scan_log_path)
            
            # load_camera_metadata returns dict {str(camera_id) -> {'position': [...], 'rotation_matrix': [...]}}
            # (position already in meters)
            camera_metadata = load_camera_metadata(camera_json_path)
            
            # Build camera lookup: camera_id (int) -> c2w (4x4 torch tensor)
            camera_lookup = {}
            for cam_id_str, cam_info in camera_metadata.items():
                position = np.array(cam_info['position'])  # already in meters
                rotation_matrix = np.array(cam_info['rotation_matrix'])
                c2w = build_4x4(rotation_matrix, position)
                camera_lookup[int(cam_id_str)] = torch.from_numpy(c2w).float()
            self.camera_lookups[material_id] = camera_lookup
            
            # Build emitter lookup as tensor: index by overall_id to get emitter_id
            # Sort by overall_id to ensure correct indexing
            sorted_metadata = sorted(metadata_list, key=lambda x: int(x['overall_id']))
            emitter_ids = np.array([int(entry['emitter_id']) for entry in sorted_metadata])
            self.emitter_lookups[material_id] = torch.from_numpy(emitter_ids).long()
        
        print(f"Metadata loaded for {len(self.material_chunks)} materials")
    
    def _filter_observations_by_xy(self, observations: np.ndarray) -> np.ndarray:
        """
        Filter observations to keep only points within the XY region.
        
        Args:
            observations: (N, 10) array with [x, y, z, image_id, pixel_x, pixel_y, r, g, b, point_id]
            
        Returns:
            Filtered observations array
        """
        x = observations[:, 0]
        y = observations[:, 1]
        
        # Keep points within half of the original width/length from center
        x_mask = np.abs(x - self.filter_center[0]) <= self.filter_half_width
        y_mask = np.abs(y - self.filter_center[1]) <= self.filter_half_length
        
        xy_mask = x_mask & y_mask
        return observations[xy_mask]
    
    def set_step(self, step: int):
        """Called by training loop to track current step for chunk reloading."""
        self.step = step
        # Request next chunk build at boundaries; swap happens lazily in __iter__
        if hasattr(self, '_dbuf') and step > 0 and step % self.switch_iters == 0:
            next_slot = 1 - self._dbuf.active
            print(f"[Step {step}] Requesting new random chunks to load into slot {next_slot}")
            self._dbuf.request_fill(next_slot)
    
    def _load_chunks(self, split='train', load_all=False) -> "MultiMaterialPointDataset.ChunkData":
        """
        Load chunks and filter by split at CHUNK level.
        
        Args:
            split: 'train' or 'val' - determines which chunks to use
                   First (1-val_ratio) chunks for training, last val_ratio chunks for validation
            load_all: if True, load ALL chunks in the split (for validation); 
                      if False, load one random chunk from the split (for training)
        
        Returns:
            ChunkData with observations from the selected chunks
        """
        all_rays = []
        all_rgbs = []
        all_xyz = []
        all_emitter_ids = []
        all_camera_ids = []
        all_material_ids = []
        all_point_ids = []
        
        is_val = (split == 'val')
        desc = f"Loading {'all' if load_all else 'random'} chunks for {split}"
        print(f"\n[{'Main' if is_val else 'Background'}] {desc}...")
        
        # Select material folders: all for validation, random sample for training
        num_to_sample = min(self.chunk_size, len(self.material_folders))
        selected_folders = random.sample(self.material_folders, num_to_sample)
        
        for material_folder in (tqdm(selected_folders, desc=desc)):
            material_id = int(material_folder.name)
            
            # Get metadata (already loaded, thread-safe to read)
            camera_lookup = self.camera_lookups[material_id]
            emitter_lookup = self.emitter_lookups[material_id]
            
            # Split chunks: first (1-val_ratio) for training, last val_ratio for validation
            all_chunks = self.material_chunks[material_id]  # Already sorted
            n_chunks = len(all_chunks)
            split_idx = int(n_chunks * (1 - self.val_ratio))
            
            if is_val:
                split_chunks = all_chunks[split_idx:]  # Last val_ratio chunks for validation
            else:
                split_chunks = all_chunks  # Use all chunks for training
            
            if len(split_chunks) == 0:
                print(f"  Warning: Material {material_id} has no {split} chunks!")
                continue
            
            # Determine which chunks to actually load
            if load_all:
                chunks_to_load = split_chunks  # Load all chunks in this split
            else:
                chunks_to_load = [random.choice(split_chunks)]  # Random single chunk from this split
            
            for chunk_path in chunks_to_load:
                # Load observations from chunk with error handling
                try:
                    obs_data = np.load(chunk_path)
                    observations = obs_data['observations']  # (N, 10): [x, y, z, image_id, pixel_x, pixel_y, r, g, b, point_id]
                except (EOFError, IOError, ValueError, KeyError) as e:
                    print(f"  [Warning] Skipping corrupted chunk {chunk_path}: {type(e).__name__}: {e}")
                    continue
                
                # Filter by XY region
                original_count = len(observations)
                if self.filter_observations:
                    observations = self._filter_observations_by_xy(observations)
                
                if len(observations) == 0:
                    continue
                
                if not load_all:
                    print(f"  [Background] Material {material_id}: Loaded {chunk_path.name} ({len(observations):,}/{original_count:,} obs after XY filter)")
                
                # Process observations - vectorized
                xyz = torch.from_numpy(observations[:, :3]).float()
                image_ids = observations[:, 3].astype(np.int32) - 1 # Colmap starts at 1
                image_ids_t = torch.from_numpy(image_ids).long()
                pixel_coords = torch.from_numpy(observations[:, 4:6]).float()
                # Apply color correction matrix to RGB values
                rgbs_np = observations[:, 6:9].astype(np.float64) @ self.ccm
                rgbs_np = rgbs_np.clip(0, None)
                rgbs = torch.from_numpy(rgbs_np).float()
                point_ids = torch.from_numpy(observations[:, 9].astype(np.int64)).long()
                
                # Vectorized: emitter_ids and camera_ids via direct tensor indexing
                chunk_emitter_ids = emitter_lookup[image_ids_t]  # (N,)
                chunk_camera_ids = image_ids_t.clone()  # (N,)
                
                # Generate rays - requires loop over unique images (get_rays needs single c2w)
                rays_list = []
                valid_mask_list = []
                
                # Group by image_id for ray generation
                unique_image_ids = np.unique(image_ids)
                for img_id in unique_image_ids:
                    if img_id not in camera_lookup:
                        continue
                    
                    c2w_full = camera_lookup[img_id]
                    c2w = c2w_full[:3, :4]
                    
                    # Get mask for this image
                    mask = image_ids_t == img_id
                    pixels = pixel_coords[mask]
                    
                    directions = get_ray_directions_for_pixels(
                        pixels, self.focal, self.cx, self.cy, self.distortion
                    )
                    
                    rays_o, rays_d = get_rays(directions, c2w, focal=None)
                    rays = torch.cat([rays_o, rays_d], dim=-1)
                    
                    rays_list.append((mask, rays))
                    valid_mask_list.append(mask)
                
                # Combine rays back into original order
                if len(rays_list) > 0:
                    # Create output tensor and fill in rays at correct positions
                    valid_mask = torch.zeros(len(image_ids_t), dtype=torch.bool)
                    for mask, _ in rays_list:
                        valid_mask |= mask
                    
                    chunk_rays = torch.zeros(valid_mask.sum(), 6)
                    chunk_xyz = xyz[valid_mask]
                    chunk_rgbs = rgbs[valid_mask]
                    chunk_point_ids = point_ids[valid_mask]
                    chunk_emitter_ids = chunk_emitter_ids[valid_mask]
                    chunk_camera_ids = chunk_camera_ids[valid_mask]
                    
                    # Map original indices to valid indices
                    valid_indices = torch.where(valid_mask)[0]
                    idx_map = torch.full((len(image_ids_t),), -1, dtype=torch.long)
                    idx_map[valid_indices] = torch.arange(len(valid_indices))
                    
                    for mask, rays in rays_list:
                        mapped_idx = idx_map[mask]
                        chunk_rays[mapped_idx] = rays
                    
                    chunk_material_ids = torch.full((chunk_rays.shape[0],), material_id, dtype=torch.long)
                    
                    all_rays.append(chunk_rays)
                    all_rgbs.append(chunk_rgbs)
                    all_xyz.append(chunk_xyz)
                    all_emitter_ids.append(chunk_emitter_ids)
                    all_camera_ids.append(chunk_camera_ids)
                    all_material_ids.append(chunk_material_ids)
                    all_point_ids.append(chunk_point_ids)
        
        # Concatenate across all materials
        rays = torch.cat(all_rays, dim=0)
        rgbs = torch.cat(all_rgbs, dim=0)
        xyz = torch.cat(all_xyz, dim=0)
        emitter_ids = torch.cat(all_emitter_ids, dim=0)
        camera_ids = torch.cat(all_camera_ids, dim=0)
        material_ids = torch.cat(all_material_ids, dim=0)
        point_ids = torch.cat(all_point_ids, dim=0)
        
        
        print(f"[{'Main' if is_val else 'Background'}] Chunk built: {len(rays):,} total {split} observations")
        
        return MultiMaterialPointDataset.ChunkData(
            rays=rays, rgbs=rgbs, xyz=xyz,
            camera_ids=camera_ids, emitter_ids=emitter_ids, material_ids=material_ids,
            point_ids=point_ids
        )
    
    def __len__(self):
        """Return number of iterations (batches) in this split."""
        if hasattr(self, 'all_rays'):
            # For validation: return number of batches to iterate through all data once
            return math.ceil(len(self.all_rays) / self.rays_num)
        else:
            # For training with double buffer, return a large number
            return 1000000
    
    def __iter__(self):
        """Infinite iterator for training (samples random batches)."""
        # Training mode with double buffer
        if hasattr(self, '_dbuf'):
            while True:
                # Non-blocking swap if next slot ready
                if self._dbuf.try_swap():
                    print(f"[Step {self.step}] ✓ Switched to new chunk (slot {self._dbuf.active})")
                
                # Use current active chunk
                chunk = self._dbuf.current()
                if chunk is None or chunk.rays.numel() == 0:
                    time.sleep(0.01)
                    continue
                
                # Random sample rays_num rays from current chunk
                total_rays = chunk.rays.shape[0]
                sample_idx = torch.randint(0, total_rays, (self.rays_num,), dtype=torch.long)
                
                yield {
                    'rays': chunk.rays[sample_idx],
                    'rgbs': chunk.rgbs[sample_idx],
                    'xyz': chunk.xyz[sample_idx],
                    'emitter_ids': chunk.emitter_ids[sample_idx],
                    'camera_ids': chunk.camera_ids[sample_idx],
                    'material_ids': chunk.material_ids[sample_idx],
                    'point_ids': chunk.point_ids[sample_idx],
                    'gt_params': torch.zeros(1),
                }
        
        # Validation mode with static data - finite iterator through all data once
        else:
            total_rays = self.all_rays.shape[0]
            num_batches = math.ceil(total_rays / self.rays_num)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.rays_num
                end_idx = min(start_idx + self.rays_num, total_rays)
                
                yield {
                    'rays': self.all_rays[start_idx:end_idx],
                    'rgbs': self.all_rgbs[start_idx:end_idx],
                    'xyz': self.all_xyz[start_idx:end_idx],
                    'emitter_ids': self.all_emitter_ids[start_idx:end_idx],
                    'camera_ids': self.all_camera_ids[start_idx:end_idx],
                    'material_ids': self.all_material_ids[start_idx:end_idx],
                    'point_ids': self.all_point_ids[start_idx:end_idx],
                    'gt_params': torch.zeros(1),
                }

