from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as NF
from pytorch_lightning import LightningModule
from brdf_plugin.utils.ops import components_from_spherical_harmonics, num_sh_bases, D_GGX, fresnelSchlick, G_Smith, G_Smith_aniso, D_GGX_aniso


from brdf_plugin.material.ubo_latent import (
    uv_to_bilinear_point_ids_and_weights,
    uv_to_nearest_point_id,
)


class PBRDecoder(nn.Module):
    """
    PBR decoder that maps a single latent (material properties) + directions -> BRDF value.
    Works in canonical/local space where normal=(0,0,1), tangent=(0,1,0).
    Supports both isotropic and anisotropic BRDF models.
    
    Latent structure:
        Isotropic: [color(3), albedo(1), roughness(1), metallic(1)] = 6 channels
        Anisotropic: [diffuse(3), ao(1), roughness(1), metallic(1), ior(1), aniso_strength(1), aniso_rot(1)] = 9 channels
        Disney (anisotropic + Disney lobes):
            [baseColor(3), ao(1), roughness(1), metallic(1), ior(1), aniso_strength(1), aniso_rot(1),
             specularTint(1), sheen(1), sheenTint(1)] = 12 channels
    """
    def __init__(
        self,
        cfg,
        latent_dim: int = None,  # Not used, kept for API compatibility
        soft_constraint: bool = True
    ):
        """
        Args:
            cfg: Configuration with anisotropic flag
            latent_dim: Not used, kept for API compatibility
            soft_constraint: If True, use sigmoid for clamping; else use hard clamp
        """
        super().__init__()
        self.anisotropic = getattr(cfg, 'anisotropic', False)
        self.disney = getattr(cfg, 'disney', False)
        self.soft_constraint = soft_constraint
        
        # Canonical space basis vectors
        # Normal = (0, 0, 1), Tangent = (0, 1, 0), Bitangent = (1, 0, 0)
        self.register_buffer('canonical_normal', torch.tensor([0.0, 0.0, 1.0]))
        self.register_buffer('canonical_tangent', torch.tensor([0.0, 1.0, 0.0]))
        self.register_buffer('canonical_bitangent', torch.tensor([1.0, 0.0, 0.0]))
        
    def compute_svbrdf_pdf(self, albedo, roughness, metallic, wi, wo, normal):
        """
        Compute isotropic SVBRDF and PDF.
        
        Args:
            albedo: [N, 1] Albedo value
            roughness: [N, 1] Roughness
            metallic: [N, 1] Metallic factor
            wi: [N, 3] Incident direction (local space)
            wo: [N, 3] Outgoing direction (local space)
            normal: [N, 3] Normal (local space)
        Returns:
            brdf [N, 1], pdf [N, 1]
        """
        h = NF.normalize(wi + wo, dim=-1)
        NoL = (wi * normal).sum(-1, keepdim=True).relu()
        NoV = (wo * normal).sum(-1, keepdim=True).relu()
        VoH = (wo * h).sum(-1, keepdim=True).relu()
        NoH = (normal * h).sum(-1, keepdim=True).relu()

        D = D_GGX(NoH, roughness)
        pdf_spec = D.data / (4 * VoH.clamp_min(1e-4)) * NoH
        pdf_diff = NoL / math.pi
        pdf = 0.5 * pdf_spec + 0.5 * pdf_diff

        kd = albedo * (1 - metallic)
        ks = 0.04 * (1 - metallic) + albedo * metallic

        G = G_Smith(NoV, NoL, roughness)
        F = fresnelSchlick(VoH, ks)
        brdf_diff = kd / math.pi
        brdf_spec = D * G * F / 4.0

        brdf = brdf_diff + brdf_spec

        return brdf, pdf

    def compute_anisotropic_svbrdf_pdf(self,
                                       diffuse, ao, ax, ay, metallic, ior,
                                       wi, wo,
                                       normal, tangent):
        """
        Compute anisotropic SVBRDF and PDF.
        
        Args:
            diffuse: [N, 3] Diffuse (base-color)
            ao: [N, 1] Ambient Occlusion
            ax, ay: [N, 1] Directional roughness (tangent/bitangent)
            metallic: [N, 1] Metallic factor
            ior: [N, 1] Specular IOR
            wi, wo: [N, 3] Incident & outgoing directions (local space)
            normal: [N, 3] Surface normal (local space)
            tangent: [N, 3] Tangent vector (local space)
        Returns:
            brdf [N, 3], pdf [N, 1]
        """
        B = NF.normalize(torch.cross(normal, tangent, dim=-1), dim=-1)
        h = NF.normalize(wi + wo, dim=-1)

        NoL = (wi * normal).sum(-1, keepdim=True).clamp_min(0.0)
        NoV = (wo * normal).sum(-1, keepdim=True).clamp_min(0.0)
        VoH = (wo * h).sum(-1, keepdim=True).clamp_min(1e-4)
        NoH = (normal * h).sum(-1, keepdim=True).clamp_min(1e-4)

        # Specular distribution and geometry
        D = D_GGX_aniso(h, normal, tangent, B, ax, ay)
        G = G_Smith_aniso(wi, wo, normal, tangent, B, ax, ay)

        # Fresnel base reflectance
        F0_dielectric = ((ior - 1) / (ior + 1)).pow(2)
        F0 = F0_dielectric * (1 - metallic) + diffuse * metallic
        F = fresnelSchlick(VoH, F0)

        # Diffuse and Specular reflectances
        kd = diffuse * (1 - metallic) * ao  # apply AO only to diffuse
        ks = 1.0  # standard GGX specular strength

        # BRDF computation
        brdf_spec = ks * (D * G * F) / (4.0 * NoL * NoV + 1e-6)
        brdf_diff = kd / math.pi
        brdf = brdf_spec + brdf_diff

        # PDF (half diffuse, half specular)
        pdf_spec = D * NoH / (4.0 * VoH)
        pdf_diff = NoL / math.pi
        pdf = 0.5 * (pdf_spec + pdf_diff)

        return brdf, pdf

    def compute_disney_anisotropic_svbrdf_pdf(self,
                                              baseColor, ao, ax, ay, metallic, ior,
                                              specularTint, sheen, sheenTint,
                                              wi, wo,
                                              normal, tangent):
        """
        Disney Principled BRDF (anisotropic) with specularTint, Burley diffuse, and sheen.
        Reference: Burley 2012, WDAS BRDF Explorer disney.brdf

        Args:
            baseColor: [N, 3] Base color (linear space)
            ao: [N, 1] Ambient Occlusion
            ax, ay: [N, 1] Directional roughness (tangent/bitangent)
            metallic: [N, 1] Metallic factor
            ior: [N, 1] Specular IOR (controls dielectric F0 magnitude)
            specularTint: [N, 1] How much dielectric F0 takes baseColor hue (0=white, 1=tinted)
            sheen: [N, 1] Sheen strength (fabric edge glow)
            sheenTint: [N, 1] Sheen color (0=white, 1=baseColor tint)
            wi, wo: [N, 3] Incident & outgoing directions (local space)
            normal: [N, 3] Surface normal (local space)
            tangent: [N, 3] Tangent vector (local space)
        Returns:
            brdf [N, 3], pdf [N, 1]
        """
        B = NF.normalize(torch.cross(normal, tangent, dim=-1), dim=-1)
        h = NF.normalize(wi + wo, dim=-1)

        NoL = (wi * normal).sum(-1, keepdim=True).clamp_min(0.0)
        NoV = (wo * normal).sum(-1, keepdim=True).clamp_min(0.0)
        VoH = (wo * h).sum(-1, keepdim=True).clamp_min(1e-4)
        NoH = (normal * h).sum(-1, keepdim=True).clamp_min(1e-4)
        LdotH = (wi * h).sum(-1, keepdim=True).clamp_min(0.0)

        # --- Disney specularTint: colored dielectric F0 ---
        Cdlum = 0.3 * baseColor[:, 0:1] + 0.6 * baseColor[:, 1:2] + 0.1 * baseColor[:, 2:3]
        Ctint = baseColor / Cdlum.clamp(min=1e-6)

        F0_dielectric = ((ior - 1) / (ior + 1)).pow(2)
        Cspec0 = torch.lerp(
            F0_dielectric * torch.lerp(torch.ones_like(baseColor), Ctint, specularTint),
            baseColor,
            metallic
        )
        F = fresnelSchlick(VoH, Cspec0)

        # --- Specular: D * G * F (same NDF/G as anisotropic path) ---
        D = D_GGX_aniso(h, normal, tangent, B, ax, ay)
        G = G_Smith_aniso(wi, wo, normal, tangent, B, ax, ay)
        brdf_spec = (D * G * F) / (4.0 * NoL * NoV + 1e-6)

        # --- Burley diffuse (roughness-dependent retroreflection) ---
        FL = (1.0 - NoL).pow(5)
        FV = (1.0 - NoV).pow(5)
        roughness = (ax + ay) * 0.5  # average roughness for diffuse term
        Fd90 = 0.5 + 2.0 * LdotH * LdotH * roughness
        Fd = (1.0 + (Fd90 - 1.0) * FL) * (1.0 + (Fd90 - 1.0) * FV)

        kd = baseColor * (1.0 - metallic) * ao
        brdf_diff = kd / math.pi * Fd

        # --- Sheen lobe (fabric edge glow) ---
        FH = (1.0 - LdotH).pow(5)
        Csheen = torch.lerp(torch.ones_like(baseColor), Ctint, sheenTint)
        brdf_sheen = FH * sheen * Csheen * (1.0 - metallic)

        brdf = brdf_diff + brdf_sheen + brdf_spec

        # PDF (half diffuse, half specular) — same as anisotropic path
        pdf_spec = D * NoH / (4.0 * VoH)
        pdf_diff = NoL / math.pi
        pdf = 0.5 * (pdf_spec + pdf_diff)

        return brdf, pdf

    def forward(
        self,
        wi: torch.Tensor,
        wo: torch.Tensor,
        latent: torch.Tensor
    ):
        """
        Compute BRDF and PDF from directions and material latent.
        All inputs are in canonical/local space where normal=(0,0,1), tangent=(0,1,0).
        
        Args:
            wi: [N, 3] Incident light direction (local space)
            wo: [N, 3] Outgoing view direction (local space)
            latent: [N, C] Material properties latent
                Isotropic (C=6): [color(3), albedo(1), roughness(1), metallic(1)]
                Anisotropic (C=9): [diffuse(3), ao(1), roughness(1), metallic(1), ior(1), aniso_strength(1), aniso_rot(1)]
                Disney (C=12): [baseColor(3), ao(1), roughness(1), metallic(1), ior(1), aniso_strength(1), aniso_rot(1),
                                specularTint(1), sheen(1), sheenTint(1)]
        
        Returns:
            brdf: [N, 3] BRDF values
            pdf: [N, 1] PDF values
        """
        N = wi.shape[0]
        device = wi.device
        
        # Canonical basis vectors (expanded to batch size)
        normal = self.canonical_normal.expand(N, -1)       # [N, 3] = (0, 0, 1)
        tangent = self.canonical_tangent.expand(N, -1)     # [N, 3] = (0, 1, 0)
        
        # Check valid geometry
        NoL = (wi * normal).sum(-1, keepdim=True)
        NoV = (wo * normal).sum(-1, keepdim=True)
        valid_geometry = (NoL > 0) & (NoV > 0)
        
        if not valid_geometry.any():
            return torch.zeros_like(wi), torch.zeros(N, 1, device=device)
        
        if self.anisotropic or self.disney:
            # Parse anisotropic latent: [baseColor(3), ao(1), roughness(1), metallic(1), ior(1), aniso_strength(1), aniso_rot(1)]
            baseColor = latent[:, 0:3]
            ao = latent[:, 3:4]
            roughness = latent[:, 4:5]
            metallic = latent[:, 5:6]
            ior = latent[:, 6:7]
            aniso_strength = latent[:, 7:8]
            aniso_rot = latent[:, 8:9]
            
            # Apply constraints (shared between anisotropic and Disney)
            if self.soft_constraint:
                baseColor = torch.sigmoid(baseColor)
                ao = torch.sigmoid(ao)
                roughness = torch.sigmoid(roughness)
                metallic = torch.sigmoid(metallic)
                ior = 1.0 + torch.sigmoid(ior) * 1.5  # IOR range [1.0, 2.5]
                aniso_strength = torch.sigmoid(aniso_strength)
                aniso_rot = torch.sigmoid(aniso_rot)
            else:
                baseColor = torch.clamp(baseColor, 0.01, 0.99)
                ao = torch.clamp(ao, 0.01, 0.99)
                roughness = torch.clamp(roughness, 0.01, 0.99)
                metallic = torch.clamp(metallic, 0.01, 0.99)
                ior = torch.clamp(ior, 1.0, 2.5)
                aniso_strength = torch.clamp(aniso_strength, 0.0, 1.0)
                aniso_rot = torch.clamp(aniso_rot, 0.0, 1.0)
            
            # Compute ax and ay based on anisotropy strength
            ax = roughness * (1.0 + aniso_strength)
            ay = roughness * (1.0 - aniso_strength * 0.5)
            
            # Rotate tangent by aniso_rot in the tangent-bitangent plane
            rotation_angle = aniso_rot * 2.0 * torch.pi  # Convert [0,1] to [0, 2π]
            cos_theta = torch.cos(rotation_angle)
            sin_theta = torch.sin(rotation_angle)
            
            bitangent = self.canonical_bitangent.expand(N, -1)  # [N, 3] = (1, 0, 0)
            T_rot = cos_theta * tangent + sin_theta * bitangent
            T_rot = NF.normalize(T_rot, dim=-1)
            
            if self.disney:
                # Parse Disney-specific channels
                specularTint = latent[:, 9:10]
                sheen = latent[:, 10:11]
                sheenTint = latent[:, 11:12]
                
                if self.soft_constraint:
                    specularTint = torch.sigmoid(specularTint)
                    sheen = torch.sigmoid(sheen)
                    sheenTint = torch.sigmoid(sheenTint)
                else:
                    specularTint = torch.clamp(specularTint, 0.0, 1.0)
                    sheen = torch.clamp(sheen, 0.0, 1.0)
                    sheenTint = torch.clamp(sheenTint, 0.0, 1.0)
                
                brdf, pdf = self.compute_disney_anisotropic_svbrdf_pdf(
                    baseColor, ao, ax, ay, metallic, ior,
                    specularTint, sheen, sheenTint,
                    wi, wo, normal, T_rot
                )
            else:
                brdf, pdf = self.compute_anisotropic_svbrdf_pdf(
                    baseColor, ao, ax, ay, metallic, ior,
                    wi, wo, normal, T_rot
                )
            
        else:
            # Parse isotropic latent: [color(3), albedo(1), roughness(1), metallic(1)]
            # Matches original texture structure: color from [0:3], ARM from [3:6]
            color = latent[:, 0:3]
            albedo = latent[:, 3:4]
            roughness = latent[:, 4:5]
            metallic = latent[:, 5:6]
            
            # Apply constraints
            if self.soft_constraint:
                color = torch.sigmoid(color)
                albedo = torch.sigmoid(albedo)
                roughness = torch.sigmoid(roughness)
                metallic = torch.sigmoid(metallic)
            else:
                color = torch.clamp(color, 0.01, 0.99)
                albedo = torch.clamp(albedo, 0.01, 0.99)
                roughness = torch.clamp(roughness, 0.01, 0.99)
                metallic = torch.clamp(metallic, 0.01, 0.99)
            
            brdf, pdf = self.compute_svbrdf_pdf(
                albedo, roughness, metallic, wi, wo, normal
            )
            brdf = color * brdf  # Scale by color (same as original)
            
        return brdf, pdf


class UBOPBRLatentBRDF(LightningModule):
    """PBR BRDF model for the Bonn UBO2014 BTF dataset.

    Same per-texel latent-bank architecture as UBOLatentBRDF but replaces the
    learned MLP decoder (BRDFDecoder) with an analytical PBRDecoder.

    Latent codes encode physical material properties:
        Isotropic  (6):  [color(3), albedo(1), roughness(1), metallic(1)]
        Anisotropic(9):  [diffuse(3), ao(1), roughness(1), metallic(1),
                          ior(1), aniso_strength(1), aniso_rot(1)]
        Disney     (12): anisotropic + [specularTint(1), sheen(1), sheenTint(1)]
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.anisotropic = getattr(cfg, 'anisotropic', False)
        self.disney = getattr(cfg, 'disney', False)
        self.soft_constraint = getattr(cfg, 'soft_constraint', True)
        self.predict_frame = getattr(cfg, 'predict_frame', False)

        # PBR latent dim is fixed by model type
        if self.disney:
            self.brdf_latent_dim = 12
        elif self.anisotropic:
            self.brdf_latent_dim = 9
        else:
            self.brdf_latent_dim = 6

        # Frame dims
        if self.predict_frame:
            self.total_latent_dim = self.brdf_latent_dim + 6
        else:
            self.total_latent_dim = self.brdf_latent_dim

        self.learnable_factor = getattr(cfg, 'learnable_factor', False)
        if self.learnable_factor:
            self.factor = nn.Parameter(torch.ones(3))

        self.uv_flip_v = getattr(cfg, "uv_flip_v", True)
        self.uv_interpolation = getattr(cfg, "uv_interpolation", "bilinear")
        # API parity with UBOLatentBRDF (PBR decoder has no direction smoothness term)
        self.smooth_reg = False
        self._last_smooth_reg_loss: Optional[torch.Tensor] = None

        # Determine latent bank size from BTF file


        self._H = 400
        self._W = 400
        total_points = self._H * self._W

        self.point_latent_bank = nn.Embedding(
            num_embeddings=total_points,
            embedding_dim=self.total_latent_dim,
            sparse=False,
        )

        nn.init.normal_(self.point_latent_bank.weight, mean=0.0, std=cfg.init_std)

        # Initialize frame slots if predict_frame
        if self.predict_frame:
            with torch.no_grad():
                self.point_latent_bank.weight[:, -6:-3] = torch.tensor([0.0, 0.0, 1.0])
                self.point_latent_bank.weight[:, -3:]   = torch.tensor([0.0, 1.0, 0.0])

        # Expose latent_dim for compatibility with trainer visualization code
        self.latent_dim = self.brdf_latent_dim

        # PBRDecoder (no trainable parameters — purely analytical)
        self.decoder = PBRDecoder(
            cfg=cfg,
            soft_constraint=self.soft_constraint,
        )

        print("UBOPBRLatentBRDF initialisation complete!")

    # ------------------------------------------------------------------
    # Frame helpers (only used when predict_frame=True)
    # ------------------------------------------------------------------
    def extract_frame_from_latent(self, latent: torch.Tensor):
        predicted_normal = NF.normalize(latent[..., -6:-3], dim=-1)
        predicted_tangent = NF.normalize(latent[..., -3:], dim=-1)

        predicted_tangent = predicted_tangent - \
            torch.sum(predicted_tangent * predicted_normal, dim=-1, keepdim=True) * predicted_normal
        predicted_tangent = NF.normalize(predicted_tangent, dim=-1)

        return predicted_normal, predicted_tangent

    def world_to_local(self, v, normal, tangent):
        bitangent = torch.cross(normal, tangent, dim=-1)
        return torch.stack([
            (v * tangent).sum(dim=-1),
            (v * bitangent).sum(dim=-1),
            (v * normal).sum(dim=-1),
        ], dim=-1)

    # ------------------------------------------------------------------
    # Point bank ↔ UV (same as UBOLatentBRDF)
    # ------------------------------------------------------------------
    def sample_latent_from_uv(self, uv: torch.Tensor) -> torch.Tensor:
        """
        Sample ``[B, total_latent_dim]`` from ``point_latent_bank`` using ``uv`` ``[B, 2]``.

        Interpolation mode: ``self.uv_interpolation`` in ``('bilinear', 'nearest')``.
        """
        if uv.dim() != 2 or uv.shape[-1] != 2:
            raise ValueError(f"uv must be [B, 2], got shape {tuple(uv.shape)}")

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
    # BRDF evaluation (aligned with UBOLatentBRDF / AnisotropicLatentTexturedModel)
    # ------------------------------------------------------------------
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
        Evaluate PBR BRDF. Latent material parameters are sampled from the point bank
        using ``uv`` (bilinear / nearest), same as ``UBOLatentBRDF``.

        Returns:
            brdf: ``[B, 3]`` (cosine-weighted for consistency with UBOLatentBRDF)
            predicted_normal: ``[B, 3]``
            pdf: ``[B, 1]`` cosine-weighted ``NoL / π`` w.r.t. ``predicted_normal``
            uv_offset: ``[B, 2]`` zeros (no neural geometry)
        """
        del gt_params, pos, latent, batch_mask, footprint_vis, dp_du, dp_dv

        self._last_smooth_reg_loss = None
        uv_offset = torch.zeros_like(uv)

        latent_full = self.sample_latent_from_uv(uv)

        if self.predict_frame:
            predicted_normal, predicted_tangent = self.extract_frame_from_latent(latent_full)
        else:
            predicted_normal = NF.normalize(normal, dim=-1)
            if TBN is None:
                raise ValueError("TBN must be provided when predict_frame is False")
            predicted_tangent = NF.normalize(TBN[:, :, 0], dim=-1)

        wi_local = self.world_to_local(wi, predicted_normal, predicted_tangent)
        wo_local = self.world_to_local(wo, predicted_normal, predicted_tangent)

        brdf_lat = latent_full[:, : self.brdf_latent_dim]
        brdf, _pdf_lobe = self.decoder(wi_local, wo_local, brdf_lat)

        if self.learnable_factor:
            brdf = brdf * self.factor

        NoL = (wi * predicted_normal).sum(-1, keepdim=True).clamp(min=0.0)
        pdf = NoL / math.pi
        brdf = brdf * NoL

        return brdf, predicted_normal, pdf, uv_offset

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
        """Evaluate PBR BRDF for given directions and point IDs.

        Args:
            wi: [B, 3] light directions (local frame for flat BTF sample)
            wo: [B, 3] view directions (local frame for flat BTF sample)
            point_ids: [B] texel indices

        Returns:
            brdf: [B, 3] BRDF values
            smooth_loss: scalar (always 0 for PBR — no learned function)
        """

        latent = self.sample_latent_from_uv(uv)

        # Inference-time override: same flag as in ubo_latent.py.
        use_predicted_frame_at_eval = bool(getattr(self.cfg, "use_predicted_frame_at_eval", True))
        if self.predict_frame and use_predicted_frame_at_eval:
            predicted_normal, predicted_tangent = self.extract_frame_from_latent(latent)
            wi_local = self.world_to_local(wi, predicted_normal, predicted_tangent)
            wo_local = self.world_to_local(wo, predicted_normal, predicted_tangent)
        else:
            wi_local = wi
            wo_local = wo
            predicted_normal = torch.zeros_like(wi)
            predicted_normal[..., 2] = 1.0

        brdf_lat = latent[:, :self.brdf_latent_dim]

        # PBRDecoder takes (wi, wo, latent) directly
        brdf, pdf = self.decoder(wi_local, wo_local, brdf_lat)

        if self.learnable_factor:
            brdf = brdf * self.factor

        smooth_loss = torch.tensor(0.0, device=wi.device)

        NoL = (wi * predicted_normal).sum(-1, keepdim=True).clamp(min=0.0)
        pdf = NoL / math.pi
        G = NoL
        # brdf/=3.14
        brdf = brdf * G

        # Clamp ≥ 0 (mirror of commit e896799 fix in anisotropicLatent.py).
        # See ubo_latent.py for the rationale — predicted-normal cosines can
        # go negative at grazing on curved geometry, poisoning the back-face
        # mask and brdf*cos product into speckled silhouette artifacts.
        cos_theta_i_pred = (wi * predicted_normal).sum(-1, keepdim=True).clamp(min=0.0)
        cos_theta_o_pred = (wo * predicted_normal).sum(-1, keepdim=True).clamp(min=0.0)
        return brdf, predicted_normal, pdf, torch.zeros_like(uv), cos_theta_i_pred, cos_theta_o_pred