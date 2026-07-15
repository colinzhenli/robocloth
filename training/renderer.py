import torch
import json
from utils.path_tracing import batched_path_tracing_tbn_preset_emitter, batched_path_tracing_tbn_real_area_emitter, points_path_tracing_real_area_emitter
from utils.scene_loader import create_rectangle_scene_params, create_hemisphere_scene_params


class ForwardRenderer:
    """Differentiable single-bounce renderer shared by both training stages.

    Builds the sample-rectangle scene (from the material's bbox.json when
    present) and dispatches to the path tracer matching the emitter type:
    per-point tracing for stage 1, batched TBN tracing for stage 2.
    """

    def __init__(self, cfg, material):
        self.cfg = cfg
        self.device = 'cuda'
        # Load center, width, and length from bbox_json file
        import os
        bbox_json_path = cfg.renderer.mesh.rectangle.bbox_json
        if os.path.exists(bbox_json_path):
            with open(bbox_json_path, 'r') as f:
                bbox_data = json.load(f)
            center = bbox_data['bbox_center']
            bbox_size = bbox_data['bbox_size']
            # Shrink the sample rectangle by 1 cm on each axis to avoid
            # boundary artifacts from cameras/lights grazing the edge.
            width = bbox_size[0] - 0.01
            length = bbox_size[1] - 0.01
            print(f"Loading rectangle scene from bbox.json: center={center}, width={width} (raw {bbox_size[0]}), length={length} (raw {bbox_size[1]})")
        else:
            center = cfg.renderer.mesh.rectangle.center
            width = cfg.renderer.mesh.rectangle.width
            length = cfg.renderer.mesh.rectangle.length
            print(f"bbox.json not found, using config: center={center}, width={width}, length={length}")
        
        if cfg.renderer.mesh.use_hemisphere:
            self.scene = create_hemisphere_scene_params(
                center=center,
                radius=0.02
            )
        else:
            self.scene = create_rectangle_scene_params(
                center=center,
                width=width,
                length=length
            )
        self.material = material.to(self.device)

        if cfg.renderer.emitter.type == 'presetpoint':
            self.ray_tracer = batched_path_tracing_tbn_preset_emitter
        elif cfg.renderer.emitter.type in ('realarea', 'multiarea', 'rotatearea'):
            if cfg.model.stage == 1:
                self.ray_tracer = points_path_tracing_real_area_emitter
            else:
                self.ray_tracer = batched_path_tracing_tbn_real_area_emitter
        else:
            raise ValueError(f"Unsupported emitter type: {cfg.renderer.emitter.type}")

        self.SPP_chunk = cfg.renderer.SPP_chunk
    

    def stage1_render(self, emitter, rays, xyz, light_idx, material_idx, point_ids, spp, gt_params=None, latent=None, validation=False):
        rays_x, rays_d, dxdu, dydv = rays[..., :3], rays[..., 3:6], rays[..., 6:9], rays[..., 9:12]
        L = torch.zeros_like(rays_x)
        ray_params = torch.zeros_like(rays)
        is_graypatch = self.material.__class__.__name__ == 'GreyPatchBRDF'
        smooth_loss_accumulated = torch.tensor(0.0, device=rays_x.device)
        
        if is_graypatch:
            pixel_all_ok_accumulated = None
            cosine_emitter_angle_accumulated = torch.zeros_like(rays_x[..., :1]).squeeze(0).squeeze(-1)
        else:
            uv_offset_accumulated = torch.zeros_like(rays_x[..., :2])
        if validation:
            self.SPP_chunk = 2
        if spp < self.SPP_chunk:
            self.SPP_chunk = spp

        if emitter is None:
            for _ in range(spp // self.SPP_chunk):
                L0, vis, ray_params, extra_output, smooth_loss = self.ray_tracer(
                    self.scene, self.emitter, self.material,
                    rays_x, rays_d, xyz, dxdu, dydv, 
                    light_idx, material_idx, point_ids, self.SPP_chunk, brdf_sampling=self.cfg.renderer.brdf_sampling, emitter_sampling=self.cfg.renderer.emitter_sampling, gt_params=gt_params, latent=latent
                )
                L += L0
                smooth_loss_accumulated = smooth_loss_accumulated + smooth_loss
                if is_graypatch:
                    pixel_all_ok, cosine_emitter_angle = extra_output
                    cosine_emitter_angle_accumulated += cosine_emitter_angle
                    if pixel_all_ok_accumulated is None:
                        pixel_all_ok_accumulated = pixel_all_ok
                    else:
                        pixel_all_ok_accumulated = pixel_all_ok_accumulated & pixel_all_ok
                else:
                    uv_offset = extra_output
                    uv_offset_accumulated += uv_offset
        else:
            for _ in range(spp // self.SPP_chunk):
                L0, vis, ray_params, extra_output, smooth_loss = self.ray_tracer(
                    self.scene, emitter, self.material,
                    rays_x, rays_d, xyz, dxdu, dydv, 
                    light_idx, material_idx, point_ids, self.SPP_chunk, brdf_sampling=self.cfg.renderer.brdf_sampling, emitter_sampling=self.cfg.renderer.emitter_sampling, gt_params=gt_params, latent=latent
                )
                L += L0
                smooth_loss_accumulated = smooth_loss_accumulated + smooth_loss
                if is_graypatch:
                    pixel_all_ok, cosine_emitter_angle = extra_output
                    cosine_emitter_angle_accumulated += cosine_emitter_angle
                    if pixel_all_ok_accumulated is None:
                        pixel_all_ok_accumulated = pixel_all_ok
                    else:
                        pixel_all_ok_accumulated = pixel_all_ok_accumulated & pixel_all_ok
                else:
                    uv_offset = extra_output
                    uv_offset_accumulated += uv_offset
        rgbs = L / (spp // self.SPP_chunk)
        rgbs = rgbs.squeeze(0) # squeeze the batch dimension
        smooth_loss_accumulated = smooth_loss_accumulated / (spp // self.SPP_chunk)
        
        if is_graypatch:
            return rgbs, vis, ray_params, (pixel_all_ok_accumulated, cosine_emitter_angle_accumulated/(spp // self.SPP_chunk)), smooth_loss_accumulated
        else:
            uv_offset_accumulated = uv_offset_accumulated / (spp // self.SPP_chunk)
            uv_offset_accumulated = uv_offset_accumulated.squeeze(0)
            return rgbs, vis, ray_params, uv_offset_accumulated, smooth_loss_accumulated
        
    def stage2_render(self, emitter, rays, light_idx, spp, gt_params=None, latent=None, validation=False):
        rays_x, rays_d, dxdu, dydv = rays[..., :3], rays[..., 3:6], rays[..., 6:9], rays[..., 9:12]
        L = torch.zeros_like(rays_x)
        ray_params = torch.zeros_like(rays)
        is_graypatch = self.material.__class__.__name__ == 'GreyPatchBRDF'
        
        if is_graypatch:
            pixel_all_ok_accumulated = None
            cosine_emitter_angle_accumulated = torch.zeros_like(rays_x[..., :1]).squeeze(0).squeeze(-1)
        else:
            uv_offset_accumulated = torch.zeros_like(rays_x[..., :2])
        if validation:
            self.SPP_chunk = 2
        if spp < self.SPP_chunk:
            self.SPP_chunk = spp

        if emitter is None:
            for _ in range(spp // self.SPP_chunk):
                L0, vis, ray_params, extra_output = self.ray_tracer(
                    self.scene, self.emitter, self.material,
                    rays_x, rays_d, dxdu, dydv, 
                    light_idx, self.SPP_chunk, brdf_sampling=self.cfg.renderer.brdf_sampling, emitter_sampling=self.cfg.renderer.emitter_sampling, gt_params=gt_params, latent=latent
                )
                L += L0
                if is_graypatch:
                    pixel_all_ok, cosine_emitter_angle = extra_output
                    cosine_emitter_angle_accumulated += cosine_emitter_angle
                    if pixel_all_ok_accumulated is None:
                        pixel_all_ok_accumulated = pixel_all_ok
                    else:
                        pixel_all_ok_accumulated = pixel_all_ok_accumulated & pixel_all_ok
                else:
                    uv_offset = extra_output
                    uv_offset_accumulated += uv_offset
        else:
            for _ in range(spp // self.SPP_chunk):
                L0, vis, ray_params, extra_output = self.ray_tracer(
                    self.scene, emitter, self.material,
                    rays_x, rays_d, dxdu, dydv, 
                    light_idx, self.SPP_chunk, brdf_sampling=self.cfg.renderer.brdf_sampling, emitter_sampling=self.cfg.renderer.emitter_sampling, gt_params=gt_params, latent=latent
                )
                L += L0
                if is_graypatch:
                    pixel_all_ok, cosine_emitter_angle = extra_output
                    cosine_emitter_angle_accumulated += cosine_emitter_angle
                    if pixel_all_ok_accumulated is None:
                        pixel_all_ok_accumulated = pixel_all_ok
                    else:
                        pixel_all_ok_accumulated = pixel_all_ok_accumulated & pixel_all_ok
                else:
                    uv_offset = extra_output
                    uv_offset_accumulated += uv_offset
        rgbs = L / (spp // self.SPP_chunk)
        rgbs = rgbs.squeeze(0) # squeeze the batch dimension
        
        if is_graypatch:
            return rgbs, vis, ray_params, (pixel_all_ok_accumulated, cosine_emitter_angle_accumulated/(spp // self.SPP_chunk))
        else:
            uv_offset_accumulated = uv_offset_accumulated / (spp // self.SPP_chunk)
            uv_offset_accumulated = uv_offset_accumulated.squeeze(0)
            return rgbs, vis, ray_params, uv_offset_accumulated