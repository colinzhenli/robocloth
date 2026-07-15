"""
UBO2014 BTF Dataset loaders for stage-2 single-material overfitting.

The Bonn UBO2014 BTF dataset stores per-texel BRDF measurements as a
compressed 5-D tensor: (nL, nV, H, W, 3).  We use the ``btf-extractor``
package to read the binary ``.btf`` files and decode images on the fly.

Data characteristics (important for pipeline correctness):
  - Values are **LDR** float32 in roughly [0, 0.7].  No HDR tone-mapping
    is needed; simple gamma for display is sufficient.
  - Angles from btf-extractor are ``(theta_l, phi_l, theta_v, phi_v)``
    in **degrees** (inclination from +Z, azimuth from +X toward +Y).
  - The sample is a flat surface with normal = (0, 0, 1).  All directions
    are already in the **local tangent frame** — no world-to-local
    transform is needed.
  - Images are RGB float32, shape ``(H, W, 3)``.

Directory layout expected:
    btf_folder/
        <material_name>_W400xH400_L151xV151.btf   (or resampled variant)
"""

from __future__ import annotations

import math
import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Spherical → Cartesian   (physics convention: theta = inclination from +Z)
# ---------------------------------------------------------------------------

def _sph2cart(theta_deg, phi_deg):
    """Convert (theta, phi) in degrees to unit Cartesian (x, y, z).

    Convention matches UBO dome:
      theta = polar angle from +Z  (0° = straight up/down, 90° = horizon)
      phi   = azimuth from +X toward +Y
    Returns (x, y, z) as float32.
    """
    theta = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    phi = np.deg2rad(np.asarray(phi_deg, dtype=np.float64))
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def _world_to_local_np(v: np.ndarray, normal: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """Same tangent-frame convention as ``AnisotropicLatentTexturedModel.world_to_local``."""
    v = np.asarray(v, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    tangent = np.asarray(tangent, dtype=np.float64)
    t_len = np.linalg.norm(tangent, axis=-1, keepdims=True)
    tangent = tangent / (t_len + 1e-8)
    bitangent = np.cross(normal, tangent)
    local = np.stack(
        [
            np.sum(v * tangent, axis=-1),
            np.sum(v * bitangent, axis=-1),
            np.sum(v * normal, axis=-1),
        ],
        axis=-1,
    )
    return local.astype(np.float32)


class UBOBTFInterpolator:
    """Preloaded UBO2014 sampler for Mitsuba BSDF evaluation, fully on GPU.

    The 22,801 measured (light, view) direction pairs are stored as a
    ``(N, 6)`` float32 CUDA tensor; the per-pixel cache is stored as
    ``(N, H*W, C)`` float16 on CUDA (≈22 GB for the standard 400×400 panel,
    fits on a single L40S 48 GB). kNN is ``torch.cdist + topk``; inverse-
    distance weighting is also pure torch. No NumPy or scipy in the hot
    path.

    Memory budget for a 400×400 panel × 22,801 angles:
      - dirpairs   :   N×6×4    = 547 KB  (float32)
      - rgbs       :   N×H*W×3×2= 22 GB   (float16)
      - per-chunk  :   chunk×N×4 distances tensor (8192×22801×4 ≈ 747 MB)
    """

    color_space = "linear"
    channel_order = "BGR"

    def __init__(
        self,
        btf_path: "str | Path",
        k: int = 4,
        p: float = 4.0,
        chunk_size: int = 8192,
        device: "str | torch.device" = "cuda",
        cache_dtype: torch.dtype = torch.float16,
    ):
        from btf_extractor import Ubo2014

        self.btf_path = Path(btf_path)
        self.k = max(1, int(k))
        self.p = float(p)
        self.chunk_size = max(1, int(chunk_size))
        self.device = torch.device(device)
        self.cache_dtype = cache_dtype

        print(f"\n{'='*60}")
        print(f"UBOBTFInterpolator  |  {self.btf_path.stem}  |  device={self.device}")
        print(f"{'='*60}")

        self.btf = Ubo2014(str(self.btf_path))
        self.H, self.W, self.C = self.btf.img_shape
        self.n_pixels = self.H * self.W

        self.angles = sorted(self.btf.angles_set)
        self.n_angles = len(self.angles)
        self.k = min(self.k, self.n_angles)

        theta_l = np.array([a[0] for a in self.angles], dtype=np.float64)
        phi_l = np.array([a[1] for a in self.angles], dtype=np.float64)
        theta_v = np.array([a[2] for a in self.angles], dtype=np.float64)
        phi_v = np.array([a[3] for a in self.angles], dtype=np.float64)
        wi_np = _sph2cart(theta_l, phi_l)
        wo_np = _sph2cart(theta_v, phi_v)
        # (N, 6) float32 on GPU — the searchable direction-pair table.
        self.dirpairs = torch.from_numpy(
            np.concatenate([wi_np, wo_np], axis=-1)
        ).to(device=self.device, dtype=torch.float32)

        # (N, H*W, C) cache on GPU — radiance per (angle, pixel, channel).
        bytes_required = self.n_angles * self.n_pixels * self.C * torch.tensor(
            [], dtype=self.cache_dtype
        ).element_size()
        print(f"Loading {self.n_angles} BTF images into GPU memory...")
        print(
            f"  Image shape: {self.H}x{self.W}, "
            f"cache dtype={self.cache_dtype}, "
            f"size={bytes_required / 1e9:.2f} GB"
        )

        self.rgbs = torch.empty(
            (self.n_angles, self.n_pixels, self.C),
            dtype=self.cache_dtype,
            device=self.device,
        )
        for i, a in enumerate(tqdm(self.angles, desc="Loading BTF angles")):
            img = self.btf.angles_to_image(*a)  # BGR float32 (H, W, C)
            img = img[:, :, ::-1].copy()        # BGR -> RGB
            np.clip(img, 0.0, None, out=img)
            self.rgbs[i] = (
                torch.from_numpy(img.reshape(self.n_pixels, self.C))
                .to(device=self.device, dtype=self.cache_dtype)
            )

        print("Loading complete.")
        print(f"{'='*60}\n")

    def _xy_from_uv(self, uv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # uv: (..., 2) torch float on self.device, in [0, 1].
        u = uv[..., 0]
        v = uv[..., 1]
        # Match NumPy reference exactly: floor(uv*(W-1)) mod W, clamped.
        x = ((u * (self.W - 1)).long() % self.W).clamp_(0, self.W - 1)
        y = ((v * (self.H - 1)).long() % self.H).clamp_(0, self.H - 1)
        return x, y

    def _weighted_average(
        self, values: torch.Tensor, distances: torch.Tensor
    ) -> torch.Tensor:
        # values: (B, k, C) float32 on device; distances: (B, k) float32.
        if self.k == 1:
            return values[:, 0, :]

        exact = distances <= 1e-8
        has_exact = exact.any(dim=-1, keepdim=True)
        inverse = 1.0 / distances.clamp(min=1e-8) ** self.p
        weights = torch.where(has_exact, exact.to(values.dtype), inverse)
        return (values * weights.unsqueeze(-1)).sum(dim=-2) / weights.sum(
            dim=-1, keepdim=True
        )

    def _query_chunk(
        self, wi: torch.Tensor, wo: torch.Tensor, uv: torch.Tensor
    ) -> torch.Tensor:
        # wi/wo: (B, 3) float32, uv: (B, 2) float32, all on self.device.
        points = torch.cat([wi, wo], dim=-1)            # (B, 6)
        distances = torch.cdist(points, self.dirpairs)  # (B, N)
        nn_dist, nn_idx = torch.topk(distances, self.k, dim=-1, largest=False)
        # If k==1, topk returns shape (B, 1) so subsequent indexing still works.

        x, y = self._xy_from_uv(uv)
        pixel_indices = (y * self.W + x).long()                                # (B,)
        pix_exp = pixel_indices.view(-1, 1).expand(-1, self.k)                 # (B, k)
        rgb_neighbors = self.rgbs[nn_idx, pix_exp].to(torch.float32)           # (B, k, C)

        rgb = self._weighted_average(rgb_neighbors, nn_dist)                   # (B, C)
        return rgb.flip(dims=[-1])                                             # RGB -> BGR

    def eval_brdf(
        self,
        gt_params,
        pos,
        wi,
        wo,
        normal,
        uv,
        TBN=None,
        latent=None,
        batch_mask=None,
        footprint_vis=None,
        dp_du=None,
        dp_dv=None,
    ):
        """Same argument layout as ``AnisotropicLatentTexturedModel.eval_brdf``.

        Inputs may be torch tensors (on any device) or ndarrays; everything is
        moved to ``self.device`` as float32 before evaluation.
        """

        def _to_dev_f32(x):
            if isinstance(x, torch.Tensor):
                return x.detach().to(device=self.device, dtype=torch.float32)
            return torch.as_tensor(
                np.asarray(x, dtype=np.float32), device=self.device
            )

        wi_t = _to_dev_f32(wi)
        wo_t = _to_dev_f32(wo)
        uv_t = _to_dev_f32(uv)

        if wi_t.dim() == 1:
            wi_t = wi_t.unsqueeze(0)
            wo_t = wo_t.unsqueeze(0)
            uv_t = uv_t.unsqueeze(0)

        if wi_t.shape != wo_t.shape or wi_t.shape[:-1] != uv_t.shape[:-1]:
            raise ValueError(
                f"wi, wo, and uv batch shapes do not match: "
                f"{tuple(wi_t.shape)}, {tuple(wo_t.shape)}, {tuple(uv_t.shape)}"
            )
        if wi_t.shape[-1] != 3 or uv_t.shape[-1] != 2:
            raise ValueError(
                f"Expected wi/wo shape (*, 3) and uv shape (*, 2), got "
                f"{tuple(wi_t.shape)}, {tuple(uv_t.shape)}"
            )

        flat_wi = wi_t.reshape(-1, 3).contiguous()
        flat_wo = wo_t.reshape(-1, 3).contiguous()
        flat_uv = uv_t.reshape(-1, 2).contiguous()
        N = flat_wi.shape[0]

        result = torch.empty((N, self.C), dtype=torch.float32, device=self.device)
        for begin in range(0, N, self.chunk_size):
            end = min(begin + self.chunk_size, N)
            result[begin:end] = self._query_chunk(
                flat_wi[begin:end], flat_wo[begin:end], flat_uv[begin:end]
            )

        # Match the legacy BGR-then-swap: _query_chunk emits BGR, swap back to RGB.
        brdf = result[:, [2, 1, 0]]

        # No predicted normal in GT — fall back to geometric +z cosines.
        cos_theta_i_pred = flat_wi[:, 2:3].contiguous()
        cos_theta_o_pred = flat_wo[:, 2:3].contiguous()
        brdf = brdf.reshape((*wi_t.shape[:-1], self.C))
        return brdf, None, None, None, cos_theta_i_pred, cos_theta_o_pred
