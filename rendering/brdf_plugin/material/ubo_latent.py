from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as NF
from pytorch_lightning import LightningModule


def uv_to_bilinear_point_ids_and_weights(
    uv: torch.Tensor,
    height: int,
    width: int,
    flip_v: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Map UV in ``[0, 1]^2`` to four indices into a row-major flattened ``H × W`` bank
    (``point_id = row * width + col``) and bilinear weights.

    Args:
        uv: ``[..., 2]`` or ``[B, 2]`` with ``(u, v)`` in ``[0, 1]``.
        height: Bank rows ``H`` (same as BTF / latent grid height).
        width: Bank columns ``W``.
        flip_v: If True, use ``v' = 1 - v`` so image row 0 matches typical mesh UV.

    Returns:
        point_ids: ``[..., 4]`` int64, corner ids in order (y0,x0), (y0,x1), (y1,x0), (y1,x1).
        weights: ``[..., 4]`` float32, nonnegative, sum to 1 per last dimension.
    """
    if height < 1 or width < 1:
        raise ValueError(f"height and width must be >= 1, got H={height}, W={width}")

    orig_shape = uv.shape[:-1]
    u = uv[..., 0].reshape(-1).clamp(0.0, 1.0)
    v = uv[..., 1].reshape(-1).clamp(0.0, 1.0)
    if flip_v:
        v = 1.0 - v

    if width == 1:
        x = torch.zeros_like(u)
    else:
        x = u * float(width - 1)
    if height == 1:
        y = torch.zeros_like(v)
    else:
        y = v * float(height - 1)

    x0 = x.floor().long().clamp(0, width - 1)
    y0 = y.floor().long().clamp(0, height - 1)
    x1 = (x0 + 1).clamp(max=width - 1)
    y1 = (y0 + 1).clamp(max=height - 1)

    wx = (x - x0.float()).clamp(0.0, 1.0)
    wy = (y - y0.float()).clamp(0.0, 1.0)

    id00 = (y0 * width + x0).to(torch.int64)
    id01 = (y0 * width + x1).to(torch.int64)
    id10 = (y1 * width + x0).to(torch.int64)
    id11 = (y1 * width + x1).to(torch.int64)

    w00 = ((1.0 - wx) * (1.0 - wy)).to(uv.dtype)
    w01 = (wx * (1.0 - wy)).to(uv.dtype)
    w10 = ((1.0 - wx) * wy).to(uv.dtype)
    w11 = (wx * wy).to(uv.dtype)

    point_ids = torch.stack([id00, id01, id10, id11], dim=-1)
    weights = torch.stack([w00, w01, w10, w11], dim=-1)
    # Normalize (handles degenerate wx/wy when W or H is 1)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    point_ids = point_ids.reshape(*orig_shape, 4)
    weights = weights.reshape(*orig_shape, 4)
    return point_ids, weights


def uv_to_nearest_point_id(
    uv: torch.Tensor,
    height: int,
    width: int,
    flip_v: bool = True,
) -> torch.Tensor:
    """Nearest-texel index ``[...,]`` int64 for a flattened ``H × W`` bank."""
    if height < 1 or width < 1:
        raise ValueError(f"height and width must be >= 1, got H={height}, W={width}")

    u = uv[..., 0].clamp(0.0, 1.0)
    v = uv[..., 1].clamp(0.0, 1.0)
    if flip_v:
        v = 1.0 - v

    if width == 1:
        col = torch.zeros_like(u, dtype=torch.int64)
    else:
        col = torch.round(u * float(width - 1)).long().clamp(0, width - 1)
    if height == 1:
        row = torch.zeros_like(v, dtype=torch.int64)
    else:
        row = torch.round(v * float(height - 1)).long().clamp(0, height - 1)
    return (row * width + col).to(torch.int64)


class UBOLatentBRDF(LightningModule):
    """BRDF model for the Bonn UBO2014 BTF dataset.

    Single-material auto-decoder with per-texel latent codes stored in
    ``nn.Embedding`` (flattened ``H × W`` bank). UV on a mesh selects bank entries
    via bilinear (or nearest) interpolation over latent vectors.

    Key differences from BonnLatentBRDF:
      - No material_ids (single material only, no global offset mapping)
      - No xyz positions required for inference (UV drives the bank)
      - Latent bank size is determined from BTF spatial resolution or config
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.latent_dim = cfg.latent_dim
        self.predict_frame = getattr(cfg, "predict_frame", False)
        self.different_decoder = getattr(cfg, "different_decoder", False)
        self.brdf_latent_dim = self.latent_dim * 3 if self.different_decoder else self.latent_dim
        self.use_pos_enc = getattr(cfg, "use_pos_enc", True)

        self.learnable_factor = getattr(cfg, "learnable_factor", False)
        if self.learnable_factor:
            self.factor = nn.Parameter(torch.ones(3))

        self.uv_flip_v = getattr(cfg, "uv_flip_v", True)
        self.uv_interpolation = getattr(cfg, "uv_interpolation", "bilinear")



        self._H = 400
        self._W = 400
        total_points = self._H * self._W

        if self.predict_frame:
            self.total_latent_dim = self.brdf_latent_dim + 6
        else:
            self.total_latent_dim = self.brdf_latent_dim

        self.point_latent_bank = nn.Embedding(
            num_embeddings=total_points,
            embedding_dim=self.total_latent_dim,
            sparse=False,
        )

        nn.init.normal_(self.point_latent_bank.weight, mean=0.0, std=cfg.init_std)

        if self.predict_frame:
            with torch.no_grad():
                self.point_latent_bank.weight[:, -6:-3] = torch.tensor([0.0, 0.0, 1.0])
                self.point_latent_bank.weight[:, -3:] = torch.tensor([0.0, 1.0, 0.0])

        from brdf_plugin.material.anisotropicLatent import BRDFDecoder

        self.decoder = BRDFDecoder(
            cfg=cfg.decoder,
            latent_dim=self.latent_dim,
            use_pos_enc=self.use_pos_enc,
            different_decoder=self.different_decoder,
        )

        self.smooth_reg = getattr(cfg.decoder, "smooth_reg", False)
        self.smooth_reg_eps = getattr(cfg.decoder, "smooth_reg_eps", 0.01)
        self._last_smooth_reg_loss: Optional[torch.Tensor] = None
        print("UBOLatentBRDF initialisation complete!")

    # ------------------------------------------------------------------
    # Frame helpers (only used when predict_frame=True)
    # ------------------------------------------------------------------
    def extract_frame_from_latent(self, latent: torch.Tensor):
        predicted_normal = NF.normalize(latent[..., -6:-3], dim=-1)
        predicted_tangent = NF.normalize(latent[..., -3:], dim=-1)
        predicted_tangent = predicted_tangent - torch.sum(
            predicted_tangent * predicted_normal, dim=-1, keepdim=True
        ) * predicted_normal
        predicted_tangent = NF.normalize(predicted_tangent, dim=-1)
        return predicted_normal, predicted_tangent

    def world_to_local(self, v, normal, tangent):
        bitangent = torch.cross(normal, tangent, dim=-1)
        return torch.stack(
            [
                (v * tangent).sum(dim=-1),
                (v * bitangent).sum(dim=-1),
                (v * normal).sum(dim=-1),
            ],
            dim=-1,
        )

    # ------------------------------------------------------------------
    # Point bank ↔ UV (flattened H×W grid)
    # ------------------------------------------------------------------
    def sample_latent_from_uv(self, uv: torch.Tensor) -> torch.Tensor:
        """
        Sample ``[B, total_latent_dim]`` from ``point_latent_bank`` using ``uv`` ``[B, 2]``.

        Interpolation mode: ``self.uv_interpolation`` in ``('bilinear', 'nearest')``.
        """
        if uv.dim() != 2 or uv.shape[-1] != 2:
            raise ValueError(f"uv must be [B, 2], got shape {tuple(uv.shape)}")

        # NaN/Inf UV survives clamp(0,1) and can yield garbage indices → Embedding OOB / UB.
        uv = torch.nan_to_num(uv, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        device = uv.device
        max_id = self.point_latent_bank.num_embeddings - 1
        if max_id < 0:
            raise RuntimeError("point_latent_bank has num_embeddings == 0")

        if self.uv_interpolation == "nearest":
            pid = uv_to_nearest_point_id(uv, self._H, self._W, flip_v=self.uv_flip_v)
            pid = pid.to(device).clamp(0, max_id)
            return self.point_latent_bank(pid)

        point_ids, weights = uv_to_bilinear_point_ids_and_weights(
            uv, self._H, self._W, flip_v=self.uv_flip_v
        )
        point_ids = point_ids.to(device).clamp(0, max_id)
        weights = weights.to(device).unsqueeze(-1)
        lat = self.point_latent_bank(point_ids.view(-1, 4)).view(
            uv.shape[0], 4, self.total_latent_dim
        )
        return (lat * weights).sum(dim=1)

    # ------------------------------------------------------------------
    # BRDF evaluation (same call pattern as AnisotropicLatentTexturedModel)
    # ------------------------------------------------------------------
    '''
    def eval_brdf_fake(
        self,
        gt_params,
        pos: torch.Tensor,
        wi: torch.Tensor,
        wo: torch.Tensor,
        normal: torch.Tensor,
        uv: torch.Tensor,
        TBN: torch.Tensor,
        latent=None,
        batch_mask=None,
        footprint_vis=None,
        dp_du=None,
        dp_dv=None,
    ):
        """
        Evaluate BRDF (aligned with ``AnisotropicLatentTexturedModel.eval_brdf``).

        Latent is read from the flattened point bank using ``uv`` (bilinear / nearest).

        Args:
            gt_params: Unused (kept for API compatibility).
            pos: ``[B, 3]`` positions (unused for bank lookup; kept for API compatibility).
            wi: ``[B, 3]`` incoming light toward the surface (world).
            wo: ``[B, 3]`` outgoing direction toward the camera (world).
            normal: ``[B, 3]`` shading normal (world).
            uv: ``[B, 2]`` texture coordinates in ``[0, 1]``.
            TBN: ``[B, 3, 3]`` columns ``(tangent, bitangent, normal)`` when ``predict_frame`` is False.

        Returns:
            brdf: ``[B, 3]``
            predicted_normal: ``[B, 3]``
            pdf: ``[B, 1]`` cosine-weighted ``NoL / π`` (``NoL`` w.r.t. ``predicted_normal``).
            uv_offset: ``[B, 2]`` zeros (no neural geometry in this model).

        If ``cfg.decoder.smooth_reg`` is True, the scalar smoothness term is stored in
        ``self._last_smooth_reg_loss`` (not returned, to keep the 4-tuple contract).
        """
        del gt_params, pos, latent, batch_mask, footprint_vis, dp_du, dp_dv

        self._last_smooth_reg_loss = None
        uv_offset = torch.zeros_like(uv)

        latent = self.sample_latent_from_uv(uv)

        if self.predict_frame:
            predicted_normal, predicted_tangent = self.extract_frame_from_latent(latent)
        else:
            predicted_normal = NF.normalize(normal, dim=-1)
            predicted_tangent = NF.normalize(TBN[:, :, 0], dim=-1)

        wi_local = self.world_to_local(wi, predicted_normal, predicted_tangent)
        wo_local = self.world_to_local(wo, predicted_normal, predicted_tangent)
        local_normal = torch.zeros_like(wi_local)
        local_normal[..., 2] = 1.0

        enc_dir = self.decoder.encode_directions(wi_local, wo_local, local_normal)

        brdf_lat = latent[:, : self.brdf_latent_dim]
        brdf = self.decoder(enc_dir, brdf_lat)
        if not self.different_decoder and brdf.shape[-1] == 1:
            brdf = brdf.expand(-1, 3)

        if self.smooth_reg:
            eps = self.smooth_reg_eps
            rand_vec = torch.randn_like(wi_local)
            rand_vec = rand_vec - (rand_vec * wi_local).sum(-1, keepdim=True) * wi_local
            axis = NF.normalize(rand_vec, dim=-1)
            wi_perturbed = wi_local * math.cos(eps) + torch.cross(axis, wi_local, dim=-1) * math.sin(
                eps
            )
            enc_pert = self.decoder.encode_directions(wi_perturbed, wo_local, local_normal)
            brdf_pert = self.decoder(enc_pert, brdf_lat)
            if not self.different_decoder and brdf_pert.shape[-1] == 1:
                brdf_pert = brdf_pert.expand(-1, 3)
            self._last_smooth_reg_loss = ((brdf_pert - brdf) / eps).pow(2).mean()

        if self.learnable_factor:
            brdf = brdf * self.factor

        NoL = (wi * predicted_normal).sum(-1, keepdim=True).clamp(min=0.0)
        pdf = NoL / math.pi
        G = NoL
        brdf = brdf * G

        return brdf, predicted_normal, pdf, uv_offset
    '''
    def eval_brdf(
        self,
        gt_params,
        pos: torch.Tensor,
        wi: torch.Tensor,
        wo: torch.Tensor,
        normal: torch.Tensor,
        uv: torch.Tensor,
        TBN: torch.Tensor,
        latent=None,
        batch_mask=None,
        footprint_vis=None,
        dp_du=None,
        dp_dv=None,
    ):
        """Evaluate BRDF for given directions and point IDs.

        Args:
            wi: [B, 3] light directions (local frame for flat BTF sample)
            wo: [B, 3] view directions (local frame for flat BTF sample)
            point_ids: [B] texel indices

        Returns:
            brdf: [B, 3] BRDF values
            smooth_loss: scalar smoothness regularisation loss
        """

        latent = self.sample_latent_from_uv(uv)  # [B, total_latent_dim]

        # Inference-time override: force the geometric +z frame even though the model
        # was trained with predict_frame=True. Latent tensor still has shape
        # (brdf_dim + 6); we only skip the extract+rotate step.
        use_predicted_frame_at_eval = bool(getattr(self.cfg, "use_predicted_frame_at_eval", True))
        if self.predict_frame and use_predicted_frame_at_eval:
            predicted_normal, predicted_tangent = self.extract_frame_from_latent(latent)
            wi_local = self.world_to_local(wi, predicted_normal, predicted_tangent)
            wo_local = self.world_to_local(wo, predicted_normal, predicted_tangent)
        else:
            # No predicted frame — directions stay in Mitsuba's local shading frame.
            wi_local = wi
            wo_local = wo
            predicted_normal = torch.zeros_like(wi)
            predicted_normal[..., 2] = 1.0

        normal_local = torch.zeros_like(wi_local)
        normal_local[..., 2] = 1.0

        enc_dir = self.decoder.encode_directions(wi_local, wo_local, normal_local)

        brdf_lat = latent[:, :self.brdf_latent_dim]
        brdf = self.decoder(enc_dir, brdf_lat)
        if not self.different_decoder and brdf.shape[-1] == 1:
            brdf = brdf.expand(-1, 3)

        # Smoothness regularisation
        if self.smooth_reg:
            eps = self.smooth_reg_eps
            rand_vec = torch.randn_like(wi_local)
            rand_vec = rand_vec - (rand_vec * wi_local).sum(-1, keepdim=True) * wi_local
            axis = NF.normalize(rand_vec, dim=-1)
            wi_perturbed = wi_local * math.cos(eps) + torch.cross(axis, wi_local, dim=-1) * math.sin(eps)
            enc_pert = self.decoder.encode_directions(wi_perturbed, wo_local, normal_local)
            brdf_pert = self.decoder(enc_pert, brdf_lat)
            if not self.different_decoder and brdf_pert.shape[-1] == 1:
                brdf_pert = brdf_pert.expand(-1, 3)
            smooth_loss = ((brdf_pert - brdf) / eps).pow(2).mean()
        else:
            smooth_loss = torch.tensor(0.0, device=wi.device)

        if self.learnable_factor:
            brdf = brdf * self.factor

        NoL = (wi * predicted_normal).sum(-1, keepdim=True).clamp(min=0.0)
        pdf = NoL / math.pi
        G = NoL
        # If training applied cosine externally (apply_cosine_weight=True), the
        # decoder learned BRDF without cos and we multiply here.  If training
        # did NOT apply cosine externally (apply_cosine_weight=False, e.g. the
        # bonn pretrain run), the decoder output already contains cos and we
        # must NOT multiply again.  The flag is set by MLPBRDF.__init__ from the
        # `apply_cosine_at_eval` BSDF prop; default True matches the original
        # from-Real training convention.
        if getattr(self, "apply_cosine_at_eval", True):
            brdf = brdf * G

        # Clamp ≥ 0 (mirror of commit e896799 fix in anisotropicLatent.py).
        # The predicted-normal cosines feed mlp.py's back-face mask and the
        # brdf*cos multiplication; on curved geometry like the sphere these
        # can go negative at grazing angles because predicted_normal is a
        # UV-space texture, not the geometric normal — producing speckled
        # black pixels along the silhouette.
        cos_theta_i_pred = (wi * predicted_normal).sum(-1, keepdim=True).clamp(min=0.0)
        cos_theta_o_pred = (wo * predicted_normal).sum(-1, keepdim=True).clamp(min=0.0)
        return brdf, predicted_normal, pdf, torch.zeros_like(uv), cos_theta_i_pred, cos_theta_o_pred


class UBOLatentBRDFDebugCos(UBOLatentBRDF):
    """Debug-only material: BRDF = |cos(wi_model, predicted_normal)|.

    Renders the per-texel predicted-normal distribution by replacing the
    decoded BRDF with the absolute cosine between the model's ``wi``
    (= the light direction after the swap in MLPBRDF.eval_anisotropic_mlp)
    and the predicted normal. abs() is used so the back-from-light side is
    not flat black — magnitude on both sides is visible, and the cos=0
    contour shows up as a dark line wherever the predicted normal is
    perpendicular to wi. cos_theta_i / cos_theta_o returned to the BSDF mask
    are also abs() so dr.select does not clip either side.
    """

    def eval_brdf(
        self,
        gt_params,
        pos: torch.Tensor,
        wi: torch.Tensor,
        wo: torch.Tensor,
        normal: torch.Tensor,
        uv: torch.Tensor,
        TBN: torch.Tensor,
        latent=None,
        batch_mask=None,
        footprint_vis=None,
        dp_du=None,
        dp_dv=None,
    ):
        latent = self.sample_latent_from_uv(uv)
        use_predicted_frame_at_eval = bool(
            getattr(self.cfg, "use_predicted_frame_at_eval", True)
        )
        if self.predict_frame and use_predicted_frame_at_eval:
            predicted_normal, _ = self.extract_frame_from_latent(latent)
        else:
            predicted_normal = torch.zeros_like(wi)
            predicted_normal[..., 2] = 1.0

        cos_wi_n = (wi * predicted_normal).sum(-1, keepdim=True)
        cos_wo_n = (wo * predicted_normal).sum(-1, keepdim=True)

        brdf = cos_wi_n.abs().expand(-1, 3)
        pdf = cos_wi_n.abs() / math.pi

        return (
            brdf,
            predicted_normal,
            pdf,
            torch.zeros_like(uv),
            cos_wi_n.abs(),
            cos_wo_n.abs(),
        )


class UBOLatentBRDFDebugBinary(UBOLatentBRDF):
    """Debug-only material: binary mask of ``cos(wi_model, predicted_normal)``.

    BRDF = 1 (white) where ``cos(wi, pred_n) > 0``, 0 (black) otherwise.
    Renders the exact boundary where the predicted normal flips to point
    away from the light — the cos=0 contour is the visible black/white edge.
    cos_theta_i / cos_theta_o sent to the BSDF mask are abs() so dr.select
    keeps both regions visible.
    """

    def eval_brdf(
        self,
        gt_params,
        pos: torch.Tensor,
        wi: torch.Tensor,
        wo: torch.Tensor,
        normal: torch.Tensor,
        uv: torch.Tensor,
        TBN: torch.Tensor,
        latent=None,
        batch_mask=None,
        footprint_vis=None,
        dp_du=None,
        dp_dv=None,
    ):
        latent = self.sample_latent_from_uv(uv)
        use_predicted_frame_at_eval = bool(
            getattr(self.cfg, "use_predicted_frame_at_eval", True)
        )
        if self.predict_frame and use_predicted_frame_at_eval:
            predicted_normal, _ = self.extract_frame_from_latent(latent)
        else:
            predicted_normal = torch.zeros_like(wi)
            predicted_normal[..., 2] = 1.0

        cos_wi_n = (wi * predicted_normal).sum(-1, keepdim=True)
        cos_wo_n = (wo * predicted_normal).sum(-1, keepdim=True)

        positive = (cos_wi_n > 0).to(cos_wi_n.dtype)
        brdf = positive.expand(-1, 3)
        pdf = positive / math.pi

        return (
            brdf,
            predicted_normal,
            pdf,
            torch.zeros_like(uv),
            cos_wi_n.abs(),
            cos_wo_n.abs(),
        )