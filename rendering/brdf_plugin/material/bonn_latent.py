"""Bonn-dataset BRDF rendering models.

The Bonn stage-2 checkpoints store a per-texel latent **bank** (an
``nn.Embedding`` of ``H*W`` points, row-major ``point_id = row * W + col``) —
exactly like the UBO models. The only difference from UBO is that UBO uses a
fixed ``400 x 400`` grid for every material, whereas **Bonn's spatial
resolution differs per material** (e.g. mat32 = 843x408, mat37 = 375x1674,
mat226 = 417x1663, mat288 = 408x728).

So these classes subclass the already-validated UBO renderers verbatim and only
override the grid size via :meth:`set_grid`, which the loader
(``mlp.create_anisotropic_model``) calls after reading ``H, W`` from
``bonn_point_metadata.json`` for the material being rendered. Everything else
(UV->bank bilinear sampling, predicted-frame extraction, decoder, cosine
convention) is inherited unchanged.

  - ``BonnLatentBRDF``    : neural MLP ``BRDFDecoder`` (Bonn/MERL/Ours stage-2
                            ckpts; 30-D latent = 24 brdf + 6 frame)
  - ``BonnPBRLatentBRDF`` : analytic Disney ``PBRDecoder`` (PBR stage-2 ckpt;
                            18-D latent = 12 disney + 6 frame)
"""
import torch.nn as nn

from brdf_plugin.material.ubo_latent import UBOLatentBRDF
from brdf_plugin.material.ubo_pbr import UBOPBRLatentBRDF


class _PerMaterialGrid:
    """Mixin: resize the latent bank to a per-material ``H x W`` grid.

    Called once, after construction and before ``load_state_dict``, so the
    embedding matches the checkpoint bank exactly (``H*W`` rows). Keeps the new
    embedding on the same device as the placeholder built in ``__init__``.
    """

    def set_grid(self, H: int, W: int):
        H, W = int(H), int(W)
        device = self.point_latent_bank.weight.device
        self._H = H
        self._W = W
        self.point_latent_bank = nn.Embedding(
            num_embeddings=H * W,
            embedding_dim=self.total_latent_dim,
            sparse=False,
        ).to(device)
        return self


class BonnLatentBRDF(_PerMaterialGrid, UBOLatentBRDF):
    """Neural Bonn model — UBOLatentBRDF with a per-material latent grid."""
    pass


class BonnPBRLatentBRDF(_PerMaterialGrid, UBOPBRLatentBRDF):
    """Disney-PBR Bonn model — UBOPBRLatentBRDF with a per-material latent grid."""
    pass
