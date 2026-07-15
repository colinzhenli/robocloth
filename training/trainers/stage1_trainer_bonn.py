import torch
import torch.nn.functional as NF
import pytorch_lightning as pl
import pl_bolts
import cv2
import numpy as np
import os
import glob
import math
from datasets.bonn import DTYPE_POLY, DTYPE_PAN, DTYPE_LLS


def _keep_only_latest_result(output_dir, base):
    """Delete this view's pred images from previous validation steps.

    The pred filename carries the PSNR (e.g. ``<base>_psnr25.30.png``), so a
    plain overwrite never happens — every step would otherwise leave its own
    file behind. Removing the prior ``<base>_psnr*`` matches keeps only the most
    recent step. ``base`` includes the data-type ``tag`` (poly/gray), so the
    different tags for one view coexist; only across-step duplicates are pruned.
    GT images use a fixed, PSNR-free name and overwrite on their own.
    """
    for path in glob.glob(os.path.join(output_dir, base + '_psnr*')):
        try:
            os.remove(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# LLS Monte-Carlo utility
# ---------------------------------------------------------------------------

def sample_quad_uniform(corners, spp):
    """Sample points uniformly on a quadrilateral via bilinear interpolation.

    Args:
        corners: (N, 4, 3)  four corner positions
        spp:     int         number of samples per pixel

    Returns:
        sample_pos: (N, spp, 3)
    """
    N = corners.shape[0]
    device = corners.device
    u = torch.rand(N, spp, 1, device=device)
    v = torch.rand(N, spp, 1, device=device)
    c0 = corners[:, 0:1, :]
    c1 = corners[:, 1:2, :]
    c2 = corners[:, 2:3, :]
    c3 = corners[:, 3:4, :]
    return (1 - u) * (1 - v) * c0 + u * (1 - v) * c1 + u * v * c2 + (1 - u) * v * c3


# ---------------------------------------------------------------------------
# bitsandbytes compat: unwrap newer __bnb_optimizer_quant_state__ format so
# checkpoints saved with bnb >= 0.49 can be loaded by older bnb (e.g. 0.41.3).
# ---------------------------------------------------------------------------
try:
    import bitsandbytes as _bnb
    class Adam8bitCompat(_bnb.optim.Adam8bit):
        def load_state_dict(self, state_dict):
            for st in state_dict.get('state', {}).values():
                if isinstance(st, dict) and '__bnb_optimizer_quant_state__' in st:
                    st.update(st.pop('__bnb_optimizer_quant_state__'))
            return super().load_state_dict(state_dict)
except ImportError:
    Adam8bitCompat = None


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Stage1Trainer_Bonn(pl.LightningModule):
    """Trainer for Bonn SVBRDF dataset.

    Forward model (after white-frame calibration):
      - Point-lit (poly / pan):  predicted = BRDF(wi, wo)
      - LLS:  predicted = sum(BRDF(wi_k, wo) * w_k) / sum(w_k)
              where w_k = cos_theta_i_k / dist_k^2
              (matches the Bonn reference's point-light-at-center model
              when the quad is small vs. distance.)
    """

    def __init__(self, cfg, material, gt_material=None, roughness=None, metallic=None):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.automatic_optimization = False
        self.more_visualizations = True

        self.material = material
        self.freeze_decoder = cfg.model.freeze_decoder

        # Loss weights.  LLS has an empirical-only radiometric calibration
        # (pan2lls coefficients are missing from the released dataset) and a
        # larger intrinsic noise floor than pan/poly, so it is downweighted.
        self.pan_loss_weight = getattr(cfg.model.loss, 'pan_weight', 0.5)
        self.lls_loss_weight = getattr(cfg.model.loss, 'lls_weight', 0.2)
        self.lls_spp = getattr(cfg.model, 'lls_spp', 16)
        self.latent_reg_weight = getattr(cfg.model, 'latent_reg_weight', 1e-4)
        self.smooth_reg_weight = getattr(cfg.model, 'smooth_reg_weight', 1e-3)
        self.grazing_ratio = getattr(cfg.model, 'grazing_ratio', 0.0)
        # Grazing mode: 'zero_exact' (legacy), 'near_zero_brdf', 'contribution_decay'.
        self.grazing_mode = getattr(cfg.model, 'grazing_mode', 'zero_exact')
        # near_zero_brdf concat-mode: c ~ U(cos_min, cos_max).
        self.grazing_cos_min = float(getattr(cfg.model, 'grazing_cos_min', 0.005))
        self.grazing_cos_max = float(getattr(cfg.model, 'grazing_cos_max', 0.03))
        # contribution_decay regularizer: separate knobs from the concat modes.
        self.grazing_decay_cos_min = float(getattr(cfg.model, 'grazing_decay_cos_min', 0.005))
        self.grazing_decay_cos_max = float(getattr(cfg.model, 'grazing_decay_cos_max', 0.2))
        self.grazing_decay_weight = float(getattr(cfg.model, 'grazing_decay_weight', 1.0))
        _decay_eps = getattr(cfg.model, 'grazing_decay_eps', None)
        if _decay_eps is None:
            try:
                _decay_eps = float(cfg.model.loss.recon_loss.log_space.logrel_eps)
            except Exception:
                _decay_eps = 1e-4
        self.grazing_decay_eps = float(_decay_eps)
        self.reset_latent_momentum = getattr(cfg.model.optimizer, 'reset_latent_momentum_on_chunk_switch', False)
        self._opt_name = getattr(cfg.model.optimizer, 'name', 'SparseAdam')

        # Approximate RGB→gray weights for panchromatic supervision.
        # Can be made learnable via cfg.model.loss.learnable_pan_weights.
        init_pan_weights = torch.tensor([0.34, 0.36, 0.28])
        if getattr(cfg.model.loss, 'learnable_pan_weights', False):
            self.pan_weights = torch.nn.Parameter(init_pan_weights.clone())
        else:
            self.register_buffer('pan_weights', init_pan_weights)

        # Multiply BRDF output by max(predicted_normal · wi, 0) before comparing
        # to GT. Use when GT bakes in the foreshortening (e.g. fine-tuning on
        # real captured measurements) but the decoder predicts a pure BRDF.
        # Affects poly, pan, AND lls by default — the LLS forward model is
        #     predicted = Σ_k (BRDF_k · w_k) / Σ_k w_k,  w_k = cos_k / dist_k²
        # so the cos in w_k cancels with Σ w_k in the denominator and the
        # prediction does NOT vanish at grazing even though GT does. Applying
        # cos manually puts cos² in the numerator and restores the grazing
        # attenuation.
        self.apply_cosine_weight = bool(getattr(cfg.model, 'apply_cosine_weight', False))
        # Hard-coded LLS-only override (no config knob). Set to False to A/B
        # test the older "no-manual-cos" LLS behavior while leaving poly/pan
        # alone. Has no effect when ``apply_cosine_weight`` is False.
        self.apply_cosine_lls = True

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    def _attach_cosine_schedulers(self, opts):
        """Wrap one or more optimizers with an epoch-based CosineAnnealingLR.

        Gated by ``cfg.model.optimizer.use_cosine_decay`` (default False — opt-in).
        Decay goes from the per-group peak LR down to ``eta_min`` over
        ``cfg.model.trainer.max_epochs`` epochs.
        """
        use_decay = getattr(self.hparams.model.optimizer, 'use_cosine_decay', False)
        if not use_decay:
            return opts

        max_epochs = self.hparams.model.trainer.max_epochs
        eta_min = getattr(self.hparams.model.optimizer, 'eta_min', 1e-5)

        is_list = isinstance(opts, list)
        opt_list = opts if is_list else [opts]

        result = []
        for opt in opt_list:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max_epochs, eta_min=eta_min,
            )
            result.append({
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
            })

        print(f"  [+] CosineAnnealingLR attached: T_max={max_epochs} epochs, eta_min={eta_min}")
        return result if is_list else result[0]

    def configure_optimizers(self):
        lr = self.hparams.model.optimizer.lr
        decoder_lr = getattr(self.hparams.model.optimizer, 'decoder_lr', lr)
        wd = self.hparams.model.optimizer.weight_decay
        opt_name = getattr(self.hparams.model.optimizer, 'name', 'SparseAdam')

        embedding_params = list(self.material.point_latent_bank.parameters())
        factor_params = [self.material.factor] if getattr(self.material, 'learnable_factor', False) else []
        
        decoder_params = [p for p in self.parameters()
                          if p not in set(embedding_params) and p not in set(factor_params)]

        if self.freeze_decoder:
            for p in decoder_params:
                p.requires_grad = False
            print("Decoder frozen – optimising latent bank and learnable factor (if present).")

        if opt_name == 'SparseAdam':
            latent_opt = torch.optim.SparseAdam(embedding_params, lr=lr)

            dense_params = decoder_params + factor_params if not self.freeze_decoder else factor_params
            if len(dense_params) > 0:
                dense_opt = torch.optim.Adam(
                    dense_params, lr=decoder_lr, betas=(0.9, 0.999), weight_decay=wd,
                )
                print(f"Using SparseAdam (embedding lr={lr}) + Adam (dense lr={decoder_lr})")
                return self._attach_cosine_schedulers([latent_opt, dense_opt])
            else:
                print(f"Using SparseAdam (embedding only), lr={lr}")
                return self._attach_cosine_schedulers(latent_opt)

        elif opt_name == 'SparseAdam8bit':
            import bitsandbytes as bnb
            latent_opt = bnb.optim.Adam8bit(embedding_params, lr=lr, betas=(0.9, 0.999))

            dense_params = decoder_params + factor_params if not self.freeze_decoder else factor_params
            if len(dense_params) > 0:
                dense_opt = torch.optim.Adam(
                    dense_params, lr=decoder_lr, betas=(0.9, 0.999), weight_decay=wd,
                )
                print(f"Using SparseAdam8bit (embedding lr={lr}) + Adam (dense lr={decoder_lr})")
                return self._attach_cosine_schedulers([latent_opt, dense_opt])
            else:
                print(f"Using SparseAdam8bit (embedding only), lr={lr}")
                return self._attach_cosine_schedulers(latent_opt)

        elif opt_name == 'Adam':
            # NOTE: group order is [decoder, embedding] to match the
            # parameter-group layout of pre-`0be34b5` checkpoints. Do not
            # reorder without re-saving / migrating existing checkpoints.
            opt_groups = []
            dense_params = decoder_params + factor_params if not self.freeze_decoder else factor_params
            if len(dense_params) > 0:
                opt_groups.append({'params': dense_params, 'lr': decoder_lr})
            opt_groups.append({'params': embedding_params, 'lr': lr})

            opt = torch.optim.Adam(opt_groups, betas=(0.9, 0.999), weight_decay=wd)

            print(f"Using Dense Adam (embedding lr={lr}, dense lr={decoder_lr})")
            return self._attach_cosine_schedulers(opt)

        elif opt_name == 'Adam8bit':
            import bitsandbytes as bnb
            opt_groups = [{'params': embedding_params, 'lr': lr}]

            dense_params = decoder_params + factor_params if not self.freeze_decoder else factor_params
            if len(dense_params) > 0:
                opt_groups.append({'params': dense_params, 'lr': decoder_lr})

            opt = Adam8bitCompat(opt_groups, betas=(0.9, 0.999), weight_decay=wd)
            print(f"Using Adam8bit (embedding lr={lr}, dense lr={decoder_lr})")
            return self._attach_cosine_schedulers(opt)

        elif opt_name == 'SGD':
            opt_groups = [{'params': embedding_params, 'lr': lr}]

            dense_params = decoder_params + factor_params if not self.freeze_decoder else factor_params
            if len(dense_params) > 0:
                opt_groups.append({'params': dense_params, 'lr': decoder_lr})

            opt = torch.optim.SGD(opt_groups, momentum=0.0, weight_decay=wd)
            print(f"Using SGD (embedding lr={lr}, dense lr={decoder_lr})")
            return self._attach_cosine_schedulers(opt)

        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    # ------------------------------------------------------------------
    # BRDF helpers
    # ------------------------------------------------------------------
    def _eval_brdf(self, xyz, wi, wo, point_ids, material_ids, normals=None,
                   return_wi_local=False):
        """Thin wrapper around material.eval_brdf.

        When return_wi_local is True the returned tuple includes wi rotated
        into the (predicted) shading frame, so the caller can compute
        NoL = wi_local.z (= wi · predicted_normal) for cosine weighting.

        Returns (brdf, predicted_normal, smooth_loss[, wi_local]).
        """
        if normals is None:
            normals = torch.zeros_like(wi)
            normals[..., 2] = 1.0
        if return_wi_local:
            brdf, pred_normal, _pdf, smooth_loss, wi_local = self.material.eval_brdf(
                xyz, wi, wo, normals,
                point_ids=point_ids, material_ids=material_ids,
                return_wi_local=True)
            return brdf, pred_normal, smooth_loss, wi_local
        brdf, pred_normal, _pdf, smooth_loss = self.material.eval_brdf(
            xyz, wi, wo, normals,
            point_ids=point_ids, material_ids=material_ids)
        return brdf, pred_normal, smooth_loss

    def _apply_cosine(self, brdf, wi_local):
        """Multiply BRDF by max(wi_local.z, 0) = max(wi · predicted_normal, 0)."""
        if not self.apply_cosine_weight:
            return brdf
        cos_theta_i = wi_local[..., 2:3].clamp(min=0)
        return brdf * cos_theta_i

    def _lls_monte_carlo(self, xyz, wo, lls_corners, point_ids, material_ids, spp, normals=None):
        """Monte-Carlo integration over LLS quad (white-frame calibrated).

        Default forward model (``apply_cosine_weight and apply_cosine_lls``):
            predicted = Σ_k (BRDF_k · cos²_k / dist²_k) / Σ_k (cos_k / dist²_k)
        Falls back to the legacy form when either flag is off:
            predicted = Σ_k (BRDF_k · cos_k / dist²_k) / Σ_k (cos_k / dist²_k)

        The default form has cos² in the numerator and reproduces the
        small-strip "point light at center with foreshortening" radiometric
        model — the prediction vanishes at grazing, matching raw GT shape.

        Normal handling mirrors pan/poly: if the material has
        ``predict_frame=True`` the predicted normal (from the latent bank) is
        used for both the geometric weights and the BRDF shading frame. Any
        ``normals`` passed in are only used when ``predict_frame=False``.

        Args:
            xyz:          (N, 3)
            wo:           (N, 3)
            lls_corners:  (N, 4, 3)
            point_ids:    (N,)
            material_ids: (N,)
            spp:          int
            normals:      (N, 3) optional gt normals (fallback when
                          ``predict_frame=False``).

        Returns:
            pred: (N, 3)  predicted calibrated measurement (RGB)
        """
        N = xyz.shape[0]
        device = xyz.device

        # Pick the normal to use for geometric weights in the same way
        # material.eval_brdf picks its shading frame.
        if getattr(self.material, 'predict_frame', False):
            global_pids = self.material.get_global_point_id(material_ids, point_ids)
            latent = self.material.point_latent_bank(global_pids)
            use_normal, _ = self.material.extract_frame_from_latent(latent)  # (N, 3)
        elif normals is not None:
            use_normal = normals
        else:
            use_normal = torch.zeros_like(wo)
            use_normal[..., 2] = 1.0

        # Sample K points on the quad
        sample_pos = sample_quad_uniform(lls_corners, spp)           # (N, spp, 3)

        xyz_exp = xyz.unsqueeze(1).expand(-1, spp, -1)               # (N, spp, 3)
        diff = sample_pos - xyz_exp                                  # (N, spp, 3)
        dist_sq = (diff * diff).sum(-1, keepdim=True).clamp(min=1e-8)
        wi_k = diff / dist_sq.sqrt()                                 # (N, spp, 3)

        normal_exp = use_normal.unsqueeze(1).expand(-1, spp, -1)
        cos_theta_i = (wi_k * normal_exp).sum(-1, keepdim=True).clamp(min=0)
        w_k = cos_theta_i / dist_sq                                  # (N, spp, 1)

        # Flatten for batch BRDF evaluation
        wi_flat  = wi_k.reshape(N * spp, 3)
        wo_flat  = wo.unsqueeze(1).expand(-1, spp, -1).reshape(N * spp, 3)
        xyz_flat = xyz_exp.reshape(N * spp, 3)
        pid_flat = point_ids.unsqueeze(1).expand(-1, spp).reshape(N * spp)
        mid_flat = material_ids.unsqueeze(1).expand(-1, spp).reshape(N * spp)
        normals_flat = (
            normals.unsqueeze(1).expand(-1, spp, -1).reshape(N * spp, 3)
            if normals is not None else None)

        # When both flags are on, multiply BRDF by max(wi · n_pred, 0) per
        # sample so the cos in w_k stops cancelling at the prediction level.
        apply_cos_lls = self.apply_cosine_weight and self.apply_cosine_lls
        if apply_cos_lls:
            brdf_flat, _, _, wi_local_flat = self._eval_brdf(
                xyz_flat, wi_flat, wo_flat, pid_flat, mid_flat,
                normals=normals_flat, return_wi_local=True)
            brdf_flat = self._apply_cosine(brdf_flat, wi_local_flat)
        else:
            brdf_flat, _, _ = self._eval_brdf(
                xyz_flat, wi_flat, wo_flat, pid_flat, mid_flat, normals=normals_flat)
        brdf_k = brdf_flat.reshape(N, spp, 3)                       # (N, spp, 3)

        numerator   = (brdf_k * w_k).sum(dim=1)                     # (N, 3)
        denominator = w_k.sum(dim=1).clamp(min=1e-8)                # (N, 1)
        return numerator / denominator

    # ------------------------------------------------------------------
    # PSNR / MSE  (stable, comparable across batches when global_psnr=True)
    # ------------------------------------------------------------------
    def _ddp_is_active(self):
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def _all_reduce_metric(self, value, op=None):
        """Return a DDP-reduced copy of a scalar tensor, or the input in single GPU."""
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if self._ddp_is_active():
            value = value.clone()
            if op is None:
                op = torch.distributed.ReduceOp.SUM
            torch.distributed.all_reduce(value, op=op)
        return value

    def _psnr_config(self):
        psnr_cfg = getattr(self.hparams.model, 'psnr', None)
        use_global = bool(getattr(psnr_cfg, 'global_psnr', True)) if psnr_cfg is not None else True
        peak = float(getattr(psnr_cfg, 'peak', 1.0)) if psnr_cfg is not None else 1.0
        return use_global, peak

    def _compute_psnr_mse(self, pred, gt, valid=None, sync_dist=False):
        """Compute (psnr, mse) over valid (non-occluded) pixels.

        When ``cfg.model.psnr.global_psnr`` is True the PSNR uses a fixed
        ``peak`` (stable across batches/runs); otherwise it falls back to the
        legacy ``gt[valid].max()``. If ``sync_dist=True``, SSE, element count,
        and non-global peak are reduced across DDP ranks before PSNR is
        computed, so the returned value is the true global metric rather than
        a per-rank metric.
        """
        use_global, peak = self._psnr_config()

        if valid is not None:
            if valid.any():
                pred_v = pred[valid]
                gt_v = gt[valid]
            else:
                pred_v = pred.new_empty((0,) + pred.shape[1:])
                gt_v = gt.new_empty((0,) + gt.shape[1:])
        else:
            pred_v = pred
            gt_v = gt

        if pred_v.numel() == 0:
            sse = torch.zeros((), dtype=pred.dtype, device=pred.device)
            n = torch.zeros((), dtype=pred.dtype, device=pred.device)
            gt_max = torch.zeros((), dtype=gt.dtype, device=gt.device)
        else:
            diff = pred_v - gt_v
            sse = diff.pow(2).sum()
            n = torch.as_tensor(diff.numel(), dtype=pred.dtype, device=pred.device)
            gt_max = gt_v.max()

        if sync_dist:
            sse = self._all_reduce_metric(sse, op=torch.distributed.ReduceOp.SUM)
            n = self._all_reduce_metric(n, op=torch.distributed.ReduceOp.SUM)
            if not use_global:
                gt_max = self._all_reduce_metric(gt_max, op=torch.distributed.ReduceOp.MAX)

        if n.item() <= 0:
            mse = torch.tensor(1.0, dtype=pred.dtype, device=pred.device)
            max_val = torch.tensor(1.0, dtype=pred.dtype, device=pred.device)
            psnr = 10.0 * torch.log10(max_val ** 2 / mse.clamp_min(1e-10))
            return psnr, mse

        mse = sse / n.clamp_min(1.0)
        if use_global:
            max_val = torch.as_tensor(peak, dtype=pred.dtype, device=pred.device)
        else:
            max_val = gt_max.clamp_min(1e-8).to(dtype=pred.dtype, device=pred.device)
        psnr = 10.0 * torch.log10(max_val ** 2 / mse.clamp_min(1e-10))
        return psnr, mse

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _loss_per_pixel(self, pred, gt):
        """Return the unreduced per-pixel reconstruction loss.

        This keeps the original loss definition, but exposes the per-pixel
        terms so DDP can normalize each branch using the global count across
        all ranks.
        """
        loss_cfg = self.hparams.model.loss.recon_loss
        name = loss_cfg.name

        if name == 'l1':
            per_pix = (pred - gt).abs().mean(dim=-1)
        elif name == 'l2':
            per_pix = (pred - gt).pow(2).mean(dim=-1)
        else:  # logrel
            rho_ref = getattr(loss_cfg.log_space, 'logrel_ref', 0.5)
            eps     = getattr(loss_cfg.log_space, 'logrel_eps', 1e-3)
            ref = torch.as_tensor(rho_ref, dtype=pred.dtype, device=pred.device)

            def lm(x):
                return torch.log((x + eps) / (ref + eps) + 1.0)

            per_pix = (lm(pred) - lm(gt)).abs().mean(dim=-1)

        return per_pix

    def _compute_loss(self, pred, gt, confidence=None):
        """Original local mean loss, kept for validation and monitoring."""
        per_pix = self._loss_per_pixel(pred, gt)

        if confidence is not None:
            per_pix = per_pix * confidence

        return per_pix.mean()

    def _compute_loss_ddp_global(self, pred, gt, confidence=None):
        """DDP-correct global mean loss for one supervision branch.

        In DDP, the old objective was:
            mean over ranks of local branch means.

        This function instead gives the gradient of:
            global branch sum / global branch count.

        Since DDP averages gradients across ranks, each rank backprops:
            world_size * local_sum / global_count.

        On single GPU, this reduces exactly to the original local mean.
        """
        per_pix = self._loss_per_pixel(pred, gt)

        if confidence is not None:
            per_pix = per_pix * confidence

        local_sum = per_pix.sum()
        local_count = torch.as_tensor(
            per_pix.numel(), dtype=pred.dtype, device=pred.device)

        if not self._ddp_is_active():
            return local_sum / local_count.clamp_min(1.0)

        global_count = local_count.detach().clone()
        torch.distributed.all_reduce(global_count, op=torch.distributed.ReduceOp.SUM)

        world_size = torch.distributed.get_world_size()
        return float(world_size) * local_sum / global_count.clamp_min(1.0)

    def _zero_angle_aux_predictions(self, point_ids, material_ids):
        """Build a grazing-incidence augmentation in one of three modes.

        Modes (selected by ``cfg.model.grazing_mode``):
          * ``'zero_exact'`` / ``'near_zero_brdf'``  (pseudo-observations).
              wi.z = 0  (or c ~ U(cos_min, cos_max) for near_zero_brdf);
              target = 0 raw BRDF. ``pred`` is concatenated into the real
              batch so the existing reconstruction loss handles it (effective
              weight is controlled by ``grazing_ratio``).
          * ``'contribution_decay'``  (BRDF-shape regularizer — NOT an
              observation). Sample c_high ~ U(grazing_decay_cos_min,
              grazing_decay_cos_max), alpha ~ U(0,1), c_low = alpha * c_high.
              Decode q_high = brdf(wi_high, wo) * c_high and
              q_low = brdf(wi_low, wo) * c_low with the same latent / wo /
              azimuth. The loss is a one-sided log-ratio hinge that only
              penalizes q_low > alpha * q_high:
                  excess     = log(clamp(q_low,0)  + eps)
                             - log(clamp(target,0) + eps)
                  per_sample = relu(excess).mean(dim=-1)
                  decay_loss = grazing_decay_weight * per_sample.sum() / B
              with B the real batch size so the regularizer grows with
              ``grazing_ratio`` instead of being averaged away.

        Returns
        -------
        None when disabled.
        For 'zero_exact' / 'near_zero_brdf':
            ``{'kind': 'concat', 'pred': brdf [N,3], 'target': zeros [N,3]}``.
        For 'contribution_decay':
            ``{'kind': 'loss', 'loss': scalar tensor, 'diagnostics': {...}}``.

        Common to all modes: latents are drawn uniformly from the **entire
        latent bank**, wo uniform on upper hemisphere, normal=(0,0,1), no
        frame transform / LLS MC, no cosine multiplication for concat modes.
        """
        n_total = point_ids.shape[0]
        n_grazing = int(self.grazing_ratio * n_total)
        if n_grazing <= 0:
            return None

        device = point_ids.device
        total_latents = self.material.point_latent_bank.num_embeddings
        global_pids = torch.randint(0, total_latents, (n_grazing,), device=device)
        latent = self.material.point_latent_bank(global_pids)[:, :self.material.latent_dim]

        # Shared sampling: wo on upper hemisphere; azimuth phi_i for wi.
        z_o = torch.rand(n_grazing, device=device)
        phi_o = torch.rand(n_grazing, device=device) * (2 * math.pi)
        sin_t_o = torch.sqrt((1 - z_o ** 2).clamp(min=0))
        wo_local = torch.stack([sin_t_o * torch.cos(phi_o),
                                sin_t_o * torch.sin(phi_o), z_o], dim=-1)
        normal_local = torch.zeros(n_grazing, 3, device=device)
        normal_local[..., 2] = 1.0
        phi_i = torch.rand(n_grazing, device=device) * (2 * math.pi)
        cos_phi_i, sin_phi_i = torch.cos(phi_i), torch.sin(phi_i)

        mode = self.grazing_mode

        if mode in ('zero_exact', 'near_zero_brdf'):
            if mode == 'zero_exact':
                c = torch.zeros(n_grazing, device=device)
            else:
                c = (torch.rand(n_grazing, device=device)
                     * (self.grazing_cos_max - self.grazing_cos_min)
                     + self.grazing_cos_min)
            r = torch.sqrt((1 - c ** 2).clamp(min=0))
            wi_local = torch.stack([r * cos_phi_i, r * sin_phi_i, c], dim=-1)
            enc_dir = self.material.decoder.encode_directions(wi_local, wo_local, normal_local)
            brdf = self.material.decoder(enc_dir, latent)
            return {'kind': 'concat', 'pred': brdf, 'target': torch.zeros_like(brdf)}

        if mode == 'contribution_decay':
            c_high = (torch.rand(n_grazing, device=device)
                      * (self.grazing_decay_cos_max - self.grazing_decay_cos_min)
                      + self.grazing_decay_cos_min)
            alpha = torch.rand(n_grazing, device=device)
            c_low = alpha * c_high
            r_high = torch.sqrt((1 - c_high ** 2).clamp(min=0))
            r_low = torch.sqrt((1 - c_low ** 2).clamp(min=0))
            wi_high = torch.stack([r_high * cos_phi_i, r_high * sin_phi_i, c_high], dim=-1)
            wi_low = torch.stack([r_low * cos_phi_i, r_low * sin_phi_i, c_low], dim=-1)

            enc_high = self.material.decoder.encode_directions(wi_high, wo_local, normal_local)
            enc_low = self.material.decoder.encode_directions(wi_low, wo_local, normal_local)
            brdf_high = self.material.decoder(enc_high, latent)
            brdf_low = self.material.decoder(enc_low, latent)

            q_high = brdf_high * c_high.unsqueeze(-1)
            q_low = brdf_low * c_low.unsqueeze(-1)
            target = alpha.unsqueeze(-1) * q_high.detach()

            eps = self.grazing_decay_eps
            q_low_pos = q_low.clamp(min=0)
            target_pos = target.clamp(min=0)
            excess = torch.log(q_low_pos + eps) - torch.log(target_pos + eps)
            per_sample = NF.relu(excess).mean(dim=-1)        # [n_grazing]
            decay_loss = self.grazing_decay_weight * per_sample.sum() / n_total

            diagnostics = {
                'q_low_mean':  q_low.detach().mean(),
                'q_low_max':   q_low.detach().max(),
                'q_high_mean': q_high.detach().mean(),
                'q_high_max':  q_high.detach().max(),
                'excess_mean': excess.detach().mean(),
                'excess_max':  excess.detach().max(),
            }
            return {'kind': 'loss', 'loss': decay_loss, 'diagnostics': diagnostics}

        raise ValueError(f"Unknown grazing_mode: {mode!r}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        xyz          = batch['xyz'].squeeze(0)
        wi           = batch['wi'].squeeze(0)
        wo           = batch['wo'].squeeze(0)
        rgbs_gt      = batch['rgbs'].squeeze(0)
        point_ids    = batch['point_ids'].squeeze(0)
        material_ids = batch['material_ids'].squeeze(0)
        data_type    = batch['data_type'].squeeze(0)
        lls_corners  = batch['lls_corners'].squeeze(0)
        confidence   = batch['confidence'].squeeze(0)
        # Per-ray RGB→pan weights from calibration; fall back to global buffer.
        if 'pan_weights' in batch:
            ray_pan_w = batch['pan_weights'].squeeze(0)  # (N, 3)
        else:
            ray_pan_w = self.pan_weights.unsqueeze(0).expand(xyz.shape[0], -1)

        gt_normals = batch.get('gt_normals')
        if gt_normals is not None:
            gt_normals = gt_normals.squeeze(0)

        poly_mask = data_type == DTYPE_POLY
        pan_mask  = data_type == DTYPE_PAN
        lls_mask  = data_type == DTYPE_LLS

        total_loss   = torch.tensor(0.0, device=xyz.device)
        smooth_total = torch.tensor(0.0, device=xyz.device)
        poly_pred    = None
        zero         = torch.tensor(0.0, device=xyz.device)
        poly_loss_raw = zero
        pan_loss_raw  = zero
        lls_loss_raw  = zero

        # --- Grazing augmentation. Latents are drawn uniformly from the whole
        # bank so a single supervision reaches every latent — poly, pan and
        # lls all share the same latent bank + decoder. ---
        #   'concat' modes (zero_exact / near_zero_brdf): synthetic BRDFs are
        #     attached to the poly path's 3-channel loss.
        #   'loss'   mode  (contribution_decay):  the function returns its own
        #     scalar loss added directly to total_loss below.
        grazing_aug = self._zero_angle_aux_predictions(point_ids, material_ids)
        grazing_loss_log = zero
        grazing_extra_loss = zero
        grazing_diag = {}
        brdf_g = gt_g_3ch = conf_g = None
        if grazing_aug is not None:
            if grazing_aug['kind'] == 'concat':
                brdf_g = grazing_aug['pred']                     # (n_g, 3)
                gt_g_3ch = grazing_aug['target']                 # zeros
                n_g = brdf_g.shape[0]
                conf_g = torch.ones(n_g, device=xyz.device)
                # Monitoring-only metric (does not enter the gradient).
                grazing_loss_log = self._compute_loss(brdf_g, gt_g_3ch)
            else:  # 'loss'
                grazing_extra_loss = grazing_aug['loss']
                grazing_loss_log = grazing_extra_loss.detach()
                grazing_diag = grazing_aug['diagnostics']

        # --- polychromatic (RGB) loss (real poly + grazing) ---
        if poly_mask.any():
            poly_normals = gt_normals[poly_mask] if gt_normals is not None else None
            if self.apply_cosine_weight:
                brdf, _, sm, wi_local_poly = self._eval_brdf(
                    xyz[poly_mask], wi[poly_mask], wo[poly_mask],
                    point_ids[poly_mask], material_ids[poly_mask],
                    normals=poly_normals, return_wi_local=True)
                brdf = self._apply_cosine(brdf, wi_local_poly)
            else:
                brdf, _, sm = self._eval_brdf(
                    xyz[poly_mask], wi[poly_mask], wo[poly_mask],
                    point_ids[poly_mask], material_ids[poly_mask],
                    normals=poly_normals)
            smooth_total = smooth_total + sm
            poly_pred = brdf
            real_pred, real_gt, real_conf = brdf, rgbs_gt[poly_mask], confidence[poly_mask]
        else:
            real_pred = real_gt = real_conf = None

        concat_grazing = brdf_g is not None
        if concat_grazing or poly_mask.any():
            if not concat_grazing:
                pred, gt_, conf = real_pred, real_gt, real_conf
            elif real_pred is None:
                pred, gt_, conf = brdf_g, gt_g_3ch, conf_g
            else:
                pred = torch.cat([real_pred, brdf_g], dim=0)
                gt_  = torch.cat([real_gt, gt_g_3ch], dim=0)
                conf = torch.cat([real_conf, conf_g], dim=0)
            poly_loss_raw = self._compute_loss_ddp_global(pred, gt_, conf)
            total_loss = total_loss + poly_loss_raw
            poly_combined_pred, poly_combined_gt, poly_combined_conf = pred, gt_, conf
        else:
            poly_combined_pred = poly_combined_gt = poly_combined_conf = None

        # --- panchromatic (grayscale) loss (real pan only; grazing is folded
        # into the poly path above) ---
        if pan_mask.any():
            pan_normals = gt_normals[pan_mask] if gt_normals is not None else None
            if self.apply_cosine_weight:
                brdf_pan, _, sm, wi_local_pan = self._eval_brdf(
                    xyz[pan_mask], wi[pan_mask], wo[pan_mask],
                    point_ids[pan_mask], material_ids[pan_mask],
                    normals=pan_normals, return_wi_local=True)
                brdf_pan = self._apply_cosine(brdf_pan, wi_local_pan)
            else:
                brdf_pan, _, sm = self._eval_brdf(
                    xyz[pan_mask], wi[pan_mask], wo[pan_mask],
                    point_ids[pan_mask], material_ids[pan_mask],
                    normals=pan_normals)
            pred_gray = (brdf_pan * ray_pan_w[pan_mask]).sum(-1, keepdim=True)
            gt_gray   = rgbs_gt[pan_mask][:, :1]
            pan_loss_raw = self._compute_loss_ddp_global(
                pred_gray, gt_gray, confidence[pan_mask])
            total_loss = total_loss + self.pan_loss_weight * pan_loss_raw
            smooth_total = smooth_total + sm

        # --- LLS (Monte-Carlo) loss (real lls only) ---
        if lls_mask.any():
            lls_normals = gt_normals[lls_mask] if gt_normals is not None else None
            lls_pred = self._lls_monte_carlo(
                xyz[lls_mask], wo[lls_mask], lls_corners[lls_mask],
                point_ids[lls_mask], material_ids[lls_mask], self.lls_spp,
                normals=lls_normals)
            pred_gray = (lls_pred * ray_pan_w[lls_mask]).sum(-1, keepdim=True)
            gt_gray   = rgbs_gt[lls_mask][:, :1]
            lls_loss_raw = self._compute_loss_ddp_global(
                pred_gray, gt_gray, confidence[lls_mask])
            total_loss = total_loss + self.lls_loss_weight * lls_loss_raw

        # Smoothness regularisation (from poly+pan branches only to avoid
        # double-counting; grazing skips eval_brdf so contributes nothing)
        total_loss = total_loss + self.smooth_reg_weight * smooth_total
        # contribution_decay grazing loss (zero for the concat modes).
        total_loss = total_loss + grazing_extra_loss

        # PSNR — over the combined poly path (real poly + grazing). The peak
        # is unaffected by zero-target grazing samples.
        psnr = torch.tensor(0.0, device=xyz.device)
        if poly_combined_pred is not None and poly_combined_pred.numel() > 0:
            psnr, _ = self._compute_psnr_mse(
                poly_combined_pred, poly_combined_gt,
                valid=poly_combined_conf > 0,
                sync_dist=True)

        log_dict = {
            'train/total_loss': total_loss,
            'train/poly_loss':  poly_loss_raw,
            'train/pan_loss':   pan_loss_raw,
            'train/lls_loss':   lls_loss_raw,
            'train/grazing_loss': grazing_loss_log,
            'train/psnr':       psnr,
            'train/poly_pred_mean': poly_pred.mean() if poly_pred is not None else 0.0,
            'train/poly_gt_mean':   rgbs_gt[poly_mask].mean() if poly_mask.any() else 0.0,
        }
        for k, v in grazing_diag.items():
            log_dict[f'train/grazing_{k}'] = v
        self.log_dict(log_dict, prog_bar=True, batch_size=xyz.shape[0], sync_dist=True)

        opts = self.optimizers()
        if not isinstance(opts, list):
            opts = [opts]
        for opt in opts:
            opt.zero_grad()
        self.manual_backward(total_loss)

        # ----- Gradient & weight diagnostics (manual, since PL's
        #       track_grad_norm is broken with automatic_optimization=False) ----
        bs = xyz.shape[0]
        total_grad_norm_sq = torch.zeros(1, device=xyz.device)

        # Per-layer decoder gradient norms & weight norms
        for name, param in self.material.decoder.named_parameters():
            w_norm = param.detach().norm(2)
            self.log(f'weight_norm/decoder.{name}', w_norm, batch_size=bs)
            if param.grad is not None:
                g_norm = param.grad.detach().norm(2)
                self.log(f'grad_norm/decoder.{name}', g_norm, batch_size=bs)
                total_grad_norm_sq += g_norm.pow(2)

        # Latent bank: only active (non-zero grad) latents
        lat_w = self.material.point_latent_bank.weight
        lat_w_norm = lat_w.detach().norm(2)
        self.log('weight_norm/latent_bank_total', lat_w_norm, batch_size=bs)
        self.log('weight_norm/latent_bank_mean', lat_w.detach().norm(dim=1).mean(), batch_size=bs)
        self.log('weight_norm/latent_bank_std', lat_w.detach().norm(dim=1).std(), batch_size=bs)
        if lat_w.grad is not None:
            lg = lat_w.grad.detach()
            # For sparse grads, count how many latents were actually touched
            if lg.is_sparse:
                lg_dense = lg.to_dense()
            else:
                lg_dense = lg
            active_mask = lg_dense.norm(dim=1) > 0
            n_active = active_mask.sum()
            self.log('grad_norm/latent_bank_total', lg_dense.norm(2), batch_size=bs)
            self.log('grad_norm/latent_bank_n_active', n_active.float(), batch_size=bs)
            if n_active > 0:
                self.log('grad_norm/latent_bank_active_mean',
                         lg_dense[active_mask].norm(dim=1).mean(), batch_size=bs)
            total_grad_norm_sq += lg_dense.norm(2).pow(2)

        # Learnable BRDF scale factor
        if hasattr(self.material, 'learnable_factor') and self.material.learnable_factor:
            factor_val = self.material.factor.detach()
            self.log('train/learnable_factor_r', factor_val[0], batch_size=bs)
            self.log('train/learnable_factor_g', factor_val[1], batch_size=bs)
            self.log('train/learnable_factor_b', factor_val[2], batch_size=bs)
            if self.material.factor.grad is not None:
                self.log('grad_norm/learnable_factor', self.material.factor.grad.detach().norm(2), batch_size=bs)

        self.log('train/grad_norm_2', total_grad_norm_sq.sqrt(),
                 prog_bar=False, batch_size=bs)

        for opt in opts:
            opt.step()

        return total_loss

    # ------------------------------------------------------------------
    # Validation  (full-image from BonnValDataset)
    # ------------------------------------------------------------------
    def on_validation_epoch_start(self):
        # Pooled squared-error accumulators for an across-all-views PSNR.
        # PSNR is non-linear in MSE (10*log10(peak^2/MSE)), so per-batch
        # PSNRs cannot be averaged directly. Instead we accumulate
        # sum-of-squared-errors and element counts in linear space across
        # every val item (poly and gray), then convert once per epoch in
        # ``on_validation_epoch_end``. Each (pixel × channel) counts as one
        # observation: poly contributes 3 elements per pixel, gray
        # contributes 1, matching the per-element MSE convention used by
        # ``_compute_psnr_mse``.
        #
        # Keep these as tensors so DDP validation can reduce them across
        # ranks. ``_val_gt_max_total`` is used only when
        # cfg.model.psnr.global_psnr=False, matching ``_compute_psnr_mse``.
        self._val_sse_total = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._val_n_total = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._val_gt_max_total = torch.tensor(0.0, dtype=torch.float32, device=self.device)

    def validation_step(self, batch, batch_idx):
        xyz          = batch['xyz'].squeeze(0)
        wi           = batch['wi'].squeeze(0)
        wo           = batch['wo'].squeeze(0)
        rgbs_gt      = batch['rgbs'].squeeze(0)
        point_ids    = batch['point_ids'].squeeze(0)
        material_ids = batch['material_ids'].squeeze(0)
        confidence   = batch['confidence'].squeeze(0)          # (N,)
        img_hw       = batch['img_hw'].squeeze(0)              # (2,)
        lls_corners  = batch['lls_corners'].squeeze(0)
        # Scatter map: original H*W flat index for each of the N supervised
        # pixels. Identity arange when point_subsample_ratio == 1.0.
        sub_indices  = batch['sub_indices'].squeeze(0).long()  # (N,)

        # Per-ray RGB→pan weights from calibration; fall back to global buffer.
        if 'pan_weights' in batch:
            ray_pan_w = batch['pan_weights'].squeeze(0)        # (N, 3)
        else:
            ray_pan_w = self.pan_weights.unsqueeze(0).expand(xyz.shape[0], -1)

        gt_normals = batch.get('gt_normals')
        if gt_normals is not None:
            gt_normals = gt_normals.squeeze(0)

        # Each val item is one full image, so all pixels share data_type.
        data_type_val = int(batch['data_type'].squeeze(0)[0].item())

        # ---- evaluate forward model branched on data_type --------------
        # Mirror training_step exactly: poly uses regular BRDF eval; pan uses
        # regular BRDF eval then RGB→gray projection; lls uses Monte-Carlo
        # area-light integration then RGB→gray projection.
        if data_type_val == DTYPE_LLS:
            brdf = self._lls_monte_carlo(
                xyz, wo, lls_corners, point_ids, material_ids, self.lls_spp,
                normals=gt_normals)
        elif self.apply_cosine_weight:
            brdf, _, _, wi_local = self._eval_brdf(
                xyz, wi, wo, point_ids, material_ids,
                normals=gt_normals, return_wi_local=True)
            brdf = self._apply_cosine(brdf, wi_local)
        else:
            brdf, _, _ = self._eval_brdf(
                xyz, wi, wo, point_ids, material_ids, normals=gt_normals)

        # Zero out brdf at occluded pixels
        brdf = brdf * confidence.unsqueeze(-1)

        # ---- loss / PSNR / MSE per data_type ---------------------------
        # For poly: 3-channel comparison.
        # For pan and lls: project pred RGB → gray with calibrated weights,
        # then compare against rgbs_gt[:, :1] (channel 0 carries the gray
        # value). Pan and lls share the same RGB→gray projection mechanism,
        # so they are merged into a single ``gray`` bucket. PSNR/MSE for
        # the gray bucket use a 1-channel tensor so they are NOT averaged
        # together with the 3-channel poly metrics.
        if data_type_val == DTYPE_POLY:
            pred_for_metric = brdf
            gt_for_metric   = rgbs_gt
            tag = 'poly'
        else:
            pred_for_metric = (brdf * ray_pan_w).sum(-1, keepdim=True)
            gt_for_metric   = rgbs_gt[:, :1]
            tag = 'gray'

        loss = self._compute_loss(pred_for_metric, gt_for_metric, confidence)
        psnr, mse = self._compute_psnr_mse(
            pred_for_metric, gt_for_metric, valid=confidence > 0)

        self.log_dict({
            f'val/{tag}_loss': loss,
            f'val/{tag}_psnr': psnr,
            f'val/{tag}_mse':  mse,
        }, prog_bar=True, batch_size=xyz.shape[0])

        # Pool SSE / element-count for the across-all-modalities metric.
        # Same valid mask as the per-type metric so occluded pixels are
        # excluded from the aggregate too.
        valid_mask = confidence > 0
        if valid_mask.any():
            diff = pred_for_metric[valid_mask] - gt_for_metric[valid_mask]
            self._val_sse_total = self._val_sse_total + diff.pow(2).sum().detach().float()
            self._val_n_total = self._val_n_total + torch.as_tensor(
                diff.numel(), dtype=torch.float32, device=pred_for_metric.device)
            self._val_gt_max_total = torch.maximum(
                self._val_gt_max_total,
                gt_for_metric[valid_mask].max().detach().float())

        if hasattr(self.material, 'learnable_factor') and self.material.learnable_factor:
            factor_val = self.material.factor.detach()
            self.log_dict({
                'val/learnable_factor_r': factor_val[0],
                'val/learnable_factor_g': factor_val[1],
                'val/learnable_factor_b': factor_val[2],
            }, batch_size=xyz.shape[0])

        # ---- reconstruct 2-D images and save ----------------------------
        H, W = img_hw[0].item(), img_hw[1].item()

        # Scatter the N supervised flat values back onto the original H*W grid.
        # Non-supervised pixels (when point_subsample_ratio < 1.0) stay zero,
        # rendering as black in the saved PNG so the spatial layout of the
        # supervised set is visible at a glance.
        def _scatter_to_image(flat: torch.Tensor) -> torch.Tensor:
            C = flat.shape[-1]
            if sub_indices.shape[0] == H * W:
                return flat.reshape(H, W, C)
            canvas = torch.zeros(H * W, C, device=flat.device, dtype=flat.dtype)
            canvas.index_copy_(0, sub_indices.to(flat.device), flat)
            return canvas.reshape(H, W, C)

        output_dir = os.path.join(self.cfg.exp_output_root_path, 'images')
        os.makedirs(output_dir, exist_ok=True)

        mat_id = material_ids[0].item()

        # Per-view image saving is gated by ``cfg.data.valid_num`` (mirrors
        # the UBO trainer): metrics still run on every val item, but PNGs
        # are only written for the first ``valid_num`` views. valid_num <= 0
        # disables the gate (saves all views).
        valid_num = getattr(self.cfg.data, 'valid_num', -1)
        save_visuals = (valid_num <= 0) or (batch_idx < valid_num)

        if save_visuals:
            # For visualization, both gt and pred are converted to a
            # displayable 3-channel tensor:
            #   - poly views: full RGB on both sides (unchanged).
            #   - pan/lls views: GT's gray-in-channel-0 is broadcast to all
            #     3 channels; pred is the calibrated RGB→gray projection
            #     broadcast to 3 channels. This makes both sides true
            #     grayscale instead of the previous red-only artifact.
            if data_type_val == DTYPE_POLY:
                gt_for_display   = rgbs_gt
                pred_for_display = brdf
            else:
                gt_for_display   = rgbs_gt[:, :1].expand(-1, 3)
                pred_for_display = pred_for_metric.expand(-1, 3)

            gt_img   = _scatter_to_image(gt_for_display)
            pred_img = _scatter_to_image(pred_for_display)

            psnr_str = f'{psnr.item():.2f}'

            # Save 8-bit PNG (with proper tone mapping for HDR to LDR display)
            gt_png   = self._tonemap_for_display(gt_img.detach())
            pred_png = self._tonemap_for_display(pred_img.detach())

            _keep_only_latest_result(output_dir, f'pred_mat{mat_id:04d}_view{batch_idx}_{tag}')
            cv2.imwrite(
                os.path.join(output_dir, f'gt_mat{mat_id:04d}_view{batch_idx}_{tag}.png'),
                cv2.cvtColor(gt_png, cv2.COLOR_RGB2BGR))
            cv2.imwrite(
                os.path.join(output_dir,
                             f'pred_mat{mat_id:04d}_view{batch_idx}_{tag}_psnr{psnr_str}.png'),
                cv2.cvtColor(pred_png, cv2.COLOR_RGB2BGR))

        # ---- save normal and tangent maps once (view-independent) ----------
        if batch_idx == 0:
            with torch.no_grad():
                global_pids = self.material.get_global_point_id(material_ids, point_ids)
                latent      = self.material.point_latent_bank(global_pids)
                if self.material.predict_frame or gt_normals is None:
                    pred_normal, pred_tangent = self.material.extract_frame_from_latent(latent)
                else:
                    pred_normal, pred_tangent = self.material.extract_frame_from_latent(latent, gt_normals)

            normal_img  = _scatter_to_image(pred_normal)
            tangent_img = _scatter_to_image(pred_tangent)
            # map [-1, 1] → [0, 255]
            normal_png  = ((normal_img.clamp(-1.0, 1.0) * 0.5 + 0.5) * 255).byte().cpu().numpy()
            tangent_png = ((tangent_img.clamp(-1.0, 1.0) * 0.5 + 0.5) * 255).byte().cpu().numpy()
            cv2.imwrite(
                os.path.join(output_dir, f'normal_mat{mat_id:04d}.png'),
                cv2.cvtColor(normal_png, cv2.COLOR_RGB2BGR))
            cv2.imwrite(
                os.path.join(output_dir, f'tangent_mat{mat_id:04d}.png'),
                cv2.cvtColor(tangent_png, cv2.COLOR_RGB2BGR))

        if self.more_visualizations:
            if batch_idx == 0:
                self.visualize_brdf_lobe(output_dir=output_dir, num_latents=10, resolution=64)

        return loss

    def on_validation_epoch_end(self):
        # Convert pooled SSE/element-count into a single combined PSNR/MSE
        # across all data types (poly + gray). Each (pixel × channel)
        # counted as one observation in ``validation_step``, so poly's
        # 3-channel comparison and gray's 1-channel comparison are
        # implicitly weighted by the number of measurements they contribute.
        #
        # DDP-safe: reduce SSE, count, and optional non-global peak across all
        # ranks before computing PSNR.
        sse = getattr(self, '_val_sse_total', None)
        n = getattr(self, '_val_n_total', None)
        gt_max = getattr(self, '_val_gt_max_total', None)
        if sse is None or n is None or gt_max is None:
            return

        sse = self._all_reduce_metric(sse, op=torch.distributed.ReduceOp.SUM)
        n = self._all_reduce_metric(n, op=torch.distributed.ReduceOp.SUM)
        gt_max = self._all_reduce_metric(gt_max, op=torch.distributed.ReduceOp.MAX)

        if n.item() <= 0:
            return

        use_global, peak = self._psnr_config()
        combined_mse = sse / n.clamp_min(1.0)
        if use_global:
            max_val = torch.as_tensor(peak, dtype=combined_mse.dtype, device=combined_mse.device)
        else:
            max_val = gt_max.clamp_min(1e-8).to(dtype=combined_mse.dtype, device=combined_mse.device)

        combined_psnr = 10.0 * torch.log10(max_val ** 2 / combined_mse.clamp_min(1e-10))
        self.log('val/all_psnr', combined_psnr, prog_bar=True, rank_zero_only=True)
        self.log('val/all_mse', combined_mse, rank_zero_only=True)

    # -----------------------------------------------------------------
    # Tone mapping for display
    # ------------------------------------------------------------------
    @staticmethod
    def _tonemap_for_display(img_hwc):
        """Reinhard tone-map + sRGB gamma → uint8 numpy (H, W, 3)."""
        x = img_hwc.clamp(min=0.0)
        x = x / (1.0 + x)                       # Reinhard
        low  = 12.92 * x
        high = 1.055 * x.pow(1.0 / 2.4) - 0.055
        x = torch.where(x <= 0.0031308, low, high).clamp(0.0, 1.0)
        return (x * 255).byte().cpu().numpy()

    def visualize_brdf_lobe(self, output_dir, num_latents=10, resolution=64):
        if not hasattr(self.material, 'decoder') or not hasattr(self.material, 'point_latent_bank'):
            return
        if not hasattr(self.material.decoder, 'encode_directions'):
            return  # PBRDecoder uses a different API

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        device = next(self.material.parameters()).device
        latent_dim   = self.material.latent_dim
        total_points = self.material.point_latent_bank.num_embeddings

        torch.manual_seed(42)
        point_indices = torch.randint(0, total_points, (num_latents,), device=device)

        with torch.no_grad():
            all_latents  = self.material.point_latent_bank(point_indices)  # [num_latents, total_latent_dim]
        brdf_latents = all_latents[:, :latent_dim]  # [num_latents, latent_dim]

        local_normal  = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        brdf_lobe_dir = os.path.join(output_dir, 'brdf_lobes')
        os.makedirs(brdf_lobe_dir, exist_ok=True)

        # ---- Visualization 1: Fix wi, vary wo --------------------------------
        theta_i_values = [15.0, 30.0, 45.0, 60.0, 75.0]

        for latent_idx in range(num_latents):
            latent = brdf_latents[latent_idx:latent_idx + 1]
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

            for theta_i_deg in theta_i_values:
                theta_i  = np.radians(theta_i_deg)
                wi = torch.tensor([[np.sin(theta_i), 0.0, np.cos(theta_i)]],
                                  device=device, dtype=torch.float32)

                theta_o_range = np.linspace(-np.pi / 2, np.pi / 2, resolution * 2)
                wo_batch = torch.zeros(len(theta_o_range), 3, device=device)
                for j, theta_o in enumerate(theta_o_range):
                    if theta_o >= 0:
                        wo_batch[j, 0] = -np.sin(theta_o)
                        wo_batch[j, 2] =  np.cos(theta_o)
                    else:
                        wo_batch[j, 0] =  np.sin(-theta_o)
                        wo_batch[j, 2] =  np.cos(-theta_o)

                wi_batch     = wi.expand(len(theta_o_range), -1)
                normal_batch = local_normal.expand(len(theta_o_range), -1)
                latent_batch = latent.expand(len(theta_o_range), -1)

                enc_dir = self.material.decoder.encode_directions(wi_batch, wo_batch, normal_batch)
                with torch.no_grad():
                    brdf = self.material.decoder(enc_dir, latent_batch)

                ax.plot(theta_o_range, brdf.mean(dim=-1).cpu().numpy(), label=f'θ_i={theta_i_deg}°')
                ax.axvline(x=np.radians(theta_i_deg),
                           color=ax.lines[-1].get_color(), linestyle='--', alpha=0.5)

            ax.set_theta_zero_location('N')
            ax.set_theta_direction(1)
            ax.set_thetamin(-90)
            ax.set_thetamax(90)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.set_title(f'BRDF Polar Plot (vary wo) - Point {point_indices[latent_idx].item()}\n'
                         f'(Fixed wi, vary wo; 0°=normal, dashed=specular direction)')
            plt.tight_layout()
            plt.savefig(os.path.join(brdf_lobe_dir,
                                     f'polar_brdf_vary_wo_pt_{point_indices[latent_idx].item()}.png'), dpi=150)
            plt.close()

        # ---- Visualization 2: Fix wo, vary wi  (BRDF × cos_theta_i) ----------
        # 5 azimuth configs per latent. wo_phi values are restricted to {90°,
        # 180°} because the Bonn rig's 4 cameras × 5 turntable rotations only
        # cover wo_phi ∈ {90, 135, 180, 225, 270}; wo_phi=0° has no GT
        # coverage so a decoder lobe at (·, 0°) cannot be checked against
        # real measurements. wi_phi stays at cardinals — the 29 LEDs span
        # all 360° so any wi azimuth is reachable in GT.
        theta_o_values = [15.0, 30.0, 45.0, 60.0]
        phi_configs = [
            (0.0,    90.0),   # wo at +y axis, wi in x-z plane
            (90.0,   90.0),   # both azimuths 90°
            (180.0,  90.0),   # wi flipped, wo at +y
            (0.0,   180.0),   # wo at -x axis, wi in x-z plane
            (90.0,  180.0),   # wi at +y, wo at -x
        ]

        for latent_idx in range(num_latents):
            latent = brdf_latents[latent_idx:latent_idx + 1]
            for wi_phi_deg, wo_phi_deg in phi_configs:
                    wi_phi = np.radians(wi_phi_deg)
                    wo_phi = np.radians(wo_phi_deg)
                    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

                    for theta_o_deg in theta_o_values:
                        theta_o = np.radians(theta_o_deg)
                        wo = torch.tensor([[
                            np.sin(theta_o) * np.cos(wo_phi),
                            np.sin(theta_o) * np.sin(wo_phi),
                            np.cos(theta_o),
                        ]], device=device, dtype=torch.float32)

                        theta_i_range = np.linspace(-np.pi / 2 + 0.01,
                                                    np.pi / 2 - 0.01, resolution * 2)
                        ti = torch.tensor(theta_i_range, device=device, dtype=torch.float32)
                        # wi sweeps a great-circle slice through zenith with azimuth
                        # = wi_phi (theta_i<0 side) and wi_phi+pi (theta_i>0 side).
                        wi_batch = torch.stack([
                            -torch.sin(ti) * float(np.cos(wi_phi)),
                            -torch.sin(ti) * float(np.sin(wi_phi)),
                            torch.cos(ti),
                        ], dim=-1)

                        wo_batch     = wo.expand(len(theta_i_range), -1)
                        normal_batch = local_normal.expand(len(theta_i_range), -1)
                        latent_batch = latent.expand(len(theta_i_range), -1)

                        enc_dir = self.material.decoder.encode_directions(wi_batch, wo_batch, normal_batch)
                        with torch.no_grad():
                            brdf = self.material.decoder(enc_dir, latent_batch)

                        cos_theta_i = wi_batch[:, 2].clamp(min=0).cpu().numpy()
                        ax.plot(theta_i_range, brdf.mean(dim=-1).cpu().numpy() * cos_theta_i,
                                label=f'θ_o={theta_o_deg}°')
                        ax.axvline(x=np.radians(theta_o_deg),
                                   color=ax.lines[-1].get_color(), linestyle='--', alpha=0.5)

                    ax.set_theta_zero_location('N')
                    ax.set_theta_direction(1)
                    ax.set_thetamin(-90)
                    ax.set_thetamax(90)
                    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                    ax.set_title(
                        f'BRDF × cos(θ_i) - Point {point_indices[latent_idx].item()}\n'
                        f'wi_φ={wi_phi_deg:.0f}°, wo_φ={wo_phi_deg:.0f}° '
                        f'(±90° = grazing)')
                    plt.tight_layout()
                    plt.savefig(os.path.join(
                        brdf_lobe_dir,
                        f'polar_brdf_vary_wi_pt_{point_indices[latent_idx].item()}'
                        f'_wiphi{int(wi_phi_deg):03d}_wophi{int(wo_phi_deg):03d}.png'), dpi=150)
                    plt.close()

        n_vary_wi = num_latents * len(phi_configs)
        print(f"[BRDF Lobe Visualization] Saved {num_latents + n_vary_wi} figures to {brdf_lobe_dir}")

    def on_train_batch_start(self, batch, batch_idx):
        step = self.global_step
        dataset = self.trainer.train_dataloader.dataset.datasets
        dataset.set_step(step)
