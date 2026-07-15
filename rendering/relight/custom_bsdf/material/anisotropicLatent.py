import torch
import torch.nn as nn
import torch.nn.functional as NF
import math
from pytorch_lightning import LightningModule
import sys
from custom_bsdf.utils.ops import *
from custom_bsdf.utils import coords,fastmerl
from custom_bsdf.utils.ops import _std_coords_to_half_diff_coords
from custom_bsdf.utils.cuda_manage import print_cuda_memory_info
from pathlib import Path
import struct
import numpy as np
import math



def load_pbr_texture(pbr_folder):
    import cv2
    from PIL import Image
    import torchvision.transforms as T
    import os

    def read_img(fname):
        path = os.path.join(pbr_folder, fname)
        if path.endswith(".exr"):
            exr = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # H × W × C, float32
            # Convert BGR to RGB for OpenCV
            exr = exr[..., ::-1]

            tensor = torch.from_numpy(exr.copy())         # torch.float32
            # Check if it's only two channels and unsqueeze if needed
            if len(tensor.shape) == 2:
                tensor = tensor.unsqueeze(2)  # Add channel dimension if missing
            return tensor.permute(2, 0, 1)                # [C,H,W]
        else:
            img = Image.open(path).convert('RGB')
            tensor = T.ToTensor()(img).float()
            return tensor if tensor.max() <= 1.0 else tensor / 255.0

    # Detect cloth name from path
    cloth_name = "fabric_pattern_07"  # default
    if "denim" in pbr_folder.lower():
        cloth_name = "denim_fabric_03"
    
    # Collect PBR texture data
    try:
        # Read all texture maps based on the detected cloth name
        if cloth_name == "denim_fabric_03":
            diffuse = read_img(f"{cloth_name}_diff_4k.jpg")                    # [3,H,W] - Diffuse map
            ao = read_img(f"{cloth_name}_ao_4k.jpg")                         # [3,H,W] - Ambient occlusion
            arm = read_img(f"{cloth_name}_arm_4k.jpg")                       # [3,H,W] - (A, Roughness, Metalness)
            rough = read_img(f"{cloth_name}_rough_4k.exr")                   # [1,H,W] - Roughness map (EXR)
            metal = read_img(f"{cloth_name}_metal_4k.exr")                   # [1,H,W] - Metallic map (EXR)
            spec_ior = read_img(f"{cloth_name}_spec_ior_4k.exr")             # [1,H,W] - Specular IOR (EXR)
            aniso_rot = read_img(f"{cloth_name}_anisotropy_rotation_4k.jpg") # [3,H,W] - Anisotropy rotation
            aniso_str = read_img(f"{cloth_name}_anisotropy_strength_4k.jpg") # [3,H,W] - Anisotropy strength
            nor_dx = read_img(f"{cloth_name}_nor_dx_4k.exr")                 # [3,H,W] - Normal map X
            nor_gl = read_img(f"{cloth_name}_nor_gl_4k.exr")                 # [3,H,W] - Normal map GL
            
            # Combine all available channels for denim fabric
            # [Color (3) + AO (3) + ARM (3) + Roughness (1) + Metal (1) + Spec IOR (1) + Aniso Rot (3) + Aniso Str (3) + Normal DX (1) + Normal GL (1)] = 20 channels
            tex = torch.cat([
                diffuse,                # RGB color (3 channels) 0:3
                ao,                   # Ambient occlusion (3 channels) 3:6
                arm,                  # ARM texture (3 channels) 6:9
                rough,                # Roughness map (1 channel) 9:10
                metal,                # Metallic map (1 channel) 10:11
                spec_ior,             # Specular IOR (1 channel) 11:12
                aniso_rot,            # Anisotropy rotation (3 channels) 12:15
                aniso_str,            # Anisotropy strength (3 channels) 15:18
                nor_dx,               # Normal map X (3 channels) 18:21
                nor_gl                # Normal map GL (3 channels) 21:24
            ], dim=0)  # [24,H,W]
        else:
            # Default fabric pattern maps
            col_1 = read_img(f"{cloth_name}_col_1_4k.jpg")      # [3,H,W] - Color map
            ao = read_img(f"{cloth_name}_ao_4k.jpg")            # [3,H,W] - Ambient occlusion
            arm = read_img(f"{cloth_name}_arm_4k.jpg")          # [3,H,W] - (A, Roughness, Metalness)
            rough = read_img(f"{cloth_name}_rough_4k.exr")      # [1,H,W] - Roughness map (EXR)
            nor_dx = read_img(f"{cloth_name}_nor_dx_4k.exr")    # [3,H,W] - Normal map X
            nor_gl = read_img(f"{cloth_name}_nor_gl_4k.exr")    # [3,H,W] - Normal map GL
            
            # Combine all available channels for default fabric
            # [Color (3) + AO (3) + ARM (3) + Roughness (1) + Normal DX (1) + Normal GL (1)] = 12 channels
            tex = torch.cat([
                col_1,                # RGB color (3 channels) 0:3
                ao,                   # Ambient occlusion (3 channels) 3:6
                arm,                  # ARM texture (3 channels) 6:9
                rough,                # Roughness map (1 channel) 9:10
                nor_dx,               # Normal map X (3 channels) 10:13
                nor_gl                # Normal map GL (3 channels) 13:16
            ], dim=0)  # [16,H,W]
        
        return tex.permute(1, 2, 0).contiguous()  # [H,W,C] => [U,V,C]
    except Exception as e:
        print(f"Error loading PBR textures: {e}")
        # Return a default texture if loading fails
        H, W = 1024, 1024
        return torch.ones(H, W, 12)  # Default to 12 channels for compatibility

class LatentTexture(nn.Module):
    """
    2D texture grid storing latent codes with interpolation and Gaussian blur.
    Each texel contains a latent code vector.
    """
    def __init__(
        self, 
        resolution: int,
        latent_dim: int,
        predict_frame: bool = False,
        init_std: float = 0.1,
        blur_config: dict = None
    ):
        """
        Args:
            resolution: Texture resolution (HxW)
            latent_dim: Dimension of each latent code
            predict_frame: Whether to include normal+tangent (6D) in latent
            init_std: Standard deviation for initialization
            blur_config: Dict with blur_sigma0 and blur_half_life
        """
        super().__init__()
        self.resolution = resolution
        self.latent_dim = latent_dim
        self.predict_frame = predict_frame
        
        # Initialize latent grid [1, latent_dim, H, W]
        latent_init = torch.randn(1, latent_dim, resolution, resolution) * init_std
        
        # Initialize frame components if needed
        if predict_frame:
            # Last 6 dimensions: normal (0,0,1) and tangent (0,1,0)
            latent_init[:, -6:-3, :, :] = torch.tensor([0.0, 0.0, 1.0]).view(1, 3, 1, 1)
            latent_init[:, -3:, :, :] = torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1)
        
        self.params = nn.Parameter(latent_init)
        
        # Gaussian blur parameters
        if blur_config is not None:
            self.blur_sigma0 = blur_config.get('blur_sigma0', 8.0)
            self.blur_half_life = blur_config.get('blur_half_life', 3333)
        else:
            self.blur_sigma0 = 8.0
            self.blur_half_life = 3333
    
    def _gaussian_kernel(self, sigma: float, channels: int):
        """Generate Gaussian kernel for depth-wise convolution."""
        if sigma < 0.5:
            return None
        
        radius = int(math.ceil(3 * sigma))
        ksize = 2 * radius + 1
        grid = torch.arange(-radius, radius + 1,
                           dtype=self.params.dtype,
                           device=self.params.device)
        g1d = torch.exp(-0.5 * (grid / sigma) ** 2)
        g1d = g1d / g1d.sum()
        g2d = (g1d[:, None] * g1d[None, :]).expand(channels, 1, ksize, ksize)
        return g2d
    
    def apply_gaussian_blur(self, step: int):
        """
        Apply progressive Gaussian blur based on training step.
        σ(t) = σ₀ · 2^{-t/half_life}
        
        Args:
            step: Current training step
        
        Returns:
            Blurred latent texture [1, total_dim, H, W]
        """
        sigma = self.blur_sigma0 * (0.5 ** (step / self.blur_half_life))
        kernel = self._gaussian_kernel(sigma, self.params.shape[1])
        
        if kernel is None:
            return self.params
        
        pad = kernel.shape[-1] // 2
        return NF.conv2d(self.params, kernel, padding=pad, groups=self.params.shape[1])
    
    def query(self, uv: torch.Tensor, blur_step: int = None):
        """
        Query latent codes at UV coordinates with bilinear interpolation.
        
        Args:
            uv: [B, 2] UV coordinates in [0, 1]
            blur_step: If not None, apply Gaussian blur for this training step
        
        Returns:
            latent: [B, total_dim] sampled latent codes
        """
        # Apply blur if training step is provided
        if blur_step is not None:
            texture = self.apply_gaussian_blur(blur_step)
        else:
            texture = self.params
        
        # Convert UV to grid coordinates for F.grid_sample
        # grid_sample expects coordinates in [-1, 1]
        grid_coords = uv * 2.0 - 1.0  # [B, 2] -> [-1, 1]
        grid_coords = grid_coords.unsqueeze(1).unsqueeze(0)  # [1, 1, B, 2]
        
        # Bilinear sampling
        latent = NF.grid_sample(
            texture,  # [1, D, H, W]
            grid_coords,  # [1, 1, B, 2]
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )  # [1, D, 1, B]
        
        # Reshape to [B, D]
        latent = latent.squeeze(0).squeeze(-1).transpose(0, 1)
        
        return latent
    
    def get_full_texture(self):
        """Get the full latent texture grid"""
        return self.params
    
    def save(self, path: str):
        """Save latent texture to file"""
        torch.save(self.params.data, path)
    
    def load(self, path: str):
        """Load latent texture from file"""
        loaded = torch.load(path)
        assert loaded.shape == self.params.shape, \
            f"Shape mismatch: {loaded.shape} vs {self.params.shape}"
        self.params.data = loaded

class BRDFDecoder(nn.Module):
    """
    MLP decoder that maps (encoded_directions + latent) -> BRDF value.
    With different_decoder=False: one MLP; output is cfg.output_channels (or [B,3]
    when use_color_decomp).
    With different_decoder=True: three independent MLPs (no shared trunk). Latent
    must be [B, 3 * latent_dim] — slices [:D], [D:2D], [2D:3D] feed R, G, B
    decoders respectively (each MLP sees encoded directions + its D-dim slice).

    Supports two additional conditioning modes (configurable via cfg):
      - FiLM conditioning (use_film): latent modulates hidden features via
        learned per-layer scale (gamma) and shift (beta) instead of being
        concatenated at the input.
      - Explicit color decomposition (use_color_decomp): latent is split into
        a color part (projected to base_color) and a shape part (used for
        angular response).  Output = base_color * angular_response.
    Both can be combined.  When both are False the decoder is identical to the
    original concatenation-based design.
    """
    def __init__(
        self,
        cfg,
        latent_dim: int,
        use_pos_enc: bool = True,
        different_decoder: bool = False
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_pos_enc = use_pos_enc
        self.different_decoder = different_decoder

        # Skip connection config
        self.use_skip_connection = getattr(cfg, 'use_skip_connection', False)
        self.skip_layer = getattr(cfg, 'skip_layer', None)

        # FiLM and color decomposition config
        self.use_film = getattr(cfg, 'use_film', False)
        self.use_color_decomp = getattr(cfg, 'use_color_decomp', False)
        self.color_latent_dim = getattr(cfg, 'color_latent_dim', 3)

        if self.use_color_decomp:
            assert self.color_latent_dim < latent_dim, \
                f"color_latent_dim ({self.color_latent_dim}) must be < latent_dim ({latent_dim})"
            self.shape_latent_dim = latent_dim - self.color_latent_dim
        else:
            self.shape_latent_dim = latent_dim

        # Setup positional encoding
        if use_pos_enc:
            self.degree = getattr(cfg, 'degree', 3)
            use_nerfstudio_sh = getattr(cfg, 'use_nerfstudio_sh', False)

            if use_nerfstudio_sh:
                self.sh_encoder = encoding.SHEncoding(levels=self.degree + 1)
                sh_dim = (self.degree + 1) ** 2
            else:
                self.sh_encoder = lambda x: components_from_spherical_harmonics(self.degree, x)
                sh_dim = num_sh_bases(self.degree)

            encoded_input_dim = sh_dim * 3  # wi, wo, normal
        else:
            encoded_input_dim = cfg.input_channels  # 9 (wi + wo + normal)

        self.encoded_input_dim = encoded_input_dim

        # MLP input dimension depends on conditioning mode
        if self.use_film:
            input_dim = encoded_input_dim
        else:
            input_dim = encoded_input_dim + self.shape_latent_dim

        self.input_dim = input_dim

        # Determine skip layer index (default to middle)
        num_hidden = len(cfg.hidden_layers)
        if self.skip_layer is None:
            self.skip_layer = num_hidden // 2

        # Output activation
        act = cfg.activation.lower()
        if act == "leakyrelu":
            output_activation = nn.LeakyReLU()
        elif act == "softplus":
            output_activation = nn.Softplus()
        else:
            output_activation = nn.ReLU()

        # Intermediate (hidden-layer) activation — defaults to "relu" for
        # backward compatibility with the original merl-branch weights.
        inter_act = getattr(cfg, 'intermediate_activation', 'relu').lower()
        if inter_act == "leakyrelu":
            intermediate_activation = nn.LeakyReLU()
        elif inter_act == "softplus":
            intermediate_activation = nn.Softplus()
        else:
            intermediate_activation = nn.ReLU()

        # MLP output channels: 1 when color_decomp provides the color
        mlp_output_channels = 1 if self.use_color_decomp else cfg.output_channels

        # ----- Color decomposition: latent_color -> base_color [B, 3] -----
        if self.use_color_decomp:
            self.color_proj = nn.Sequential(
                nn.Linear(self.color_latent_dim, 3),
                nn.Softplus(),
            )

        # ----- FiLM mappers: latent_shape -> (gamma, beta) per hidden layer -----
        if self.use_film:
            self.film_mappers = nn.ModuleList()
            for hidden_dim in cfg.hidden_layers:
                mapper = nn.Linear(self.shape_latent_dim, 2 * hidden_dim)
                nn.init.zeros_(mapper.weight)
                nn.init.zeros_(mapper.bias)
                mapper.bias.data[:hidden_dim] = 1.0  # gamma=1, beta=0
                self.film_mappers.append(mapper)

        # ----- Build MLP(s) -----
        def build_mlp():
            if self.use_film or self.use_skip_connection:
                layers = nn.ModuleList()
                prev_dim = input_dim
                for i, hidden_dim in enumerate(cfg.hidden_layers):
                    if self.use_skip_connection and i == self.skip_layer:
                        prev_dim = prev_dim + input_dim
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    prev_dim = hidden_dim
                layers.append(nn.Linear(prev_dim, mlp_output_channels))
                return layers
            else:
                layers = []
                prev_dim = input_dim
                for hidden_dim in cfg.hidden_layers:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(intermediate_activation)
                    prev_dim = hidden_dim
                layers.append(nn.Linear(prev_dim, mlp_output_channels))
                layers.append(output_activation)
                return nn.Sequential(*layers)

        # Store activation for manual forward passes (FiLM / skip connection)
        if self.use_film or self.use_skip_connection:
            self.activation = intermediate_activation
            self.output_activation = output_activation

        if different_decoder:
            assert not self.use_film and not self.use_color_decomp, \
                "FiLM / color_decomp not supported with different_decoder=True"
            assert not self.use_skip_connection, \
                "skip connection not supported with different_decoder=True"

            def build_channel_mlp():
                layers = []
                prev_dim = input_dim
                for hidden_dim in cfg.hidden_layers:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(intermediate_activation)
                    prev_dim = hidden_dim
                layers.append(nn.Linear(prev_dim, 1))
                layers.append(output_activation)
                return nn.Sequential(*layers)

            self.mlp_r = build_channel_mlp()
            self.mlp_g = build_channel_mlp()
            self.mlp_b = build_channel_mlp()
        else:
            self.mlp = build_mlp()
    
    def encode_directions(
        self,
        wi_local: torch.Tensor,
        wo_local: torch.Tensor,
        normal_local: torch.Tensor
    ):
        """
        Encode local-space directions with spherical harmonics.
        
        Args:
            wi_local: [B, 3] incoming light direction (local space)
            wo_local: [B, 3] outgoing view direction (local space)
            normal_local: [B, 3] normal (local space, typically [0,0,1])
        
        Returns:
            encoded: [B, encoded_dim] encoded directions
        """
        if self.use_pos_enc:
            wi_enc = self.sh_encoder(wi_local)
            wo_enc = self.sh_encoder(wo_local)
            normal_enc = self.sh_encoder(normal_local)
            return torch.cat([wi_enc, wo_enc, normal_enc], dim=-1)
        else:
            return torch.cat([wi_local, wo_local, normal_local], dim=-1)
    
    def _forward_with_skip(self, mlp_input: torch.Tensor, layers: nn.ModuleList):
        """Forward pass with skip connection for ModuleList-based MLP."""
        x = mlp_input
        num_layers = len(layers)
        
        for i, layer in enumerate(layers):
            # Inject skip connection at specified layer
            if i == self.skip_layer:
                x = torch.cat([x, mlp_input], dim=-1)
            
            x = layer(x)
            
            # Apply activation: ReLU for middle layers, output_activation for last layer
            if i < num_layers - 1:
                x = self.activation(x)
            else:
                x = self.output_activation(x)
        
        return x

    def _forward_with_film(self, mlp_input: torch.Tensor, layers: nn.ModuleList,
                           film_latent: torch.Tensor):
        """Forward pass with FiLM conditioning (and optional skip connection)."""
        x = mlp_input
        num_layers = len(layers)
        num_hidden = num_layers - 1

        for i in range(num_hidden):
            if self.use_skip_connection and i == self.skip_layer:
                x = torch.cat([x, mlp_input], dim=-1)
            x = layers[i](x)
            film_out = self.film_mappers[i](film_latent)
            gamma, beta = film_out.chunk(2, dim=-1)
            x = gamma * x + beta
            x = self.activation(x)

        x = layers[-1](x)
        x = self.output_activation(x)
        return x

    def forward(
        self,
        enc_dir: torch.Tensor,
        latent: torch.Tensor,
        channel: str = None
    ):
        """
        Decode BRDF from encoded directions and latent.

        Returns:
            brdf: [B, 3] when different_decoder=True or use_color_decomp=True,
                  else [B, output_channels]
            If different_decoder and channel is 'r'|'g'|'b', returns [B, 1] for
            that channel only (latent must still contain the full 3*latent_dim
            slice passed by the caller).
        """
        if self.use_color_decomp:
            latent_color = latent[:, :self.color_latent_dim]
            latent_shape = latent[:, self.color_latent_dim:]
            base_color = self.color_proj(latent_color)  # [B, 3]
        else:
            latent_shape = latent

        if self.use_film:
            mlp_input = enc_dir
        else:
            mlp_input = torch.cat([enc_dir, latent_shape], dim=-1)

        if self.different_decoder:
            d = self.latent_dim
            if latent.shape[-1] < 3 * d:
                raise ValueError(
                    f"different_decoder expects latent dim >= 3*latent_dim ({3*d}), "
                    f"got {latent.shape[-1]}"
                )
            zr = latent[:, :d]
            zg = latent[:, d : 2 * d]
            zb = latent[:, 2 * d : 3 * d]
            ir = torch.cat([enc_dir, zr], dim=-1)
            ig = torch.cat([enc_dir, zg], dim=-1)
            ib = torch.cat([enc_dir, zb], dim=-1)
            if channel is None:
                brdf_r = self.mlp_r(ir)
                brdf_g = self.mlp_g(ig)
                brdf_b = self.mlp_b(ib)
                return torch.cat([brdf_r, brdf_g, brdf_b], dim=-1)
            ch = channel.lower()
            if ch == 'r':
                return self.mlp_r(ir)
            if ch == 'g':
                return self.mlp_g(ig)
            if ch == 'b':
                return self.mlp_b(ib)
            raise ValueError(f"channel must be None, 'r', 'g', or 'b', got {channel!r}")

        if self.use_film:
            angular = self._forward_with_film(mlp_input, self.mlp, latent_shape)
        elif self.use_skip_connection:
            angular = self._forward_with_skip(mlp_input, self.mlp)
        else:
            angular = self.mlp(mlp_input)

        if self.use_color_decomp:
            return base_color * angular  # [B, 3] * [B, 1] -> [B, 3]

        return angular

class NeuralGeometry(nn.Module):
    """
    Neural network for predicting UV offsets based on viewing/lighting directions.
    This enables view-dependent displacement/parallax effects.
    """
    def __init__(
        self,
        cfg,
        geometry_latent_dim: int,
        use_local_wi_wo: bool = True,
        use_pos_enc: bool = True
    ):
        """
        Args:
            cfg: Configuration with hidden_layers, activation, output_channels
            geometry_latent_dim: Dimension of geometry-specific latent
            use_local_wi_wo: Use local space directions (vs world space)
            use_pos_enc: Use spherical harmonics encoding
        """
        super().__init__()
        self.geometry_latent_dim = geometry_latent_dim
        self.use_local_wi_wo = use_local_wi_wo
        self.use_pos_enc = use_pos_enc
        
        # Setup positional encoding if enabled
        if use_pos_enc:
            self.degree = 3
            self.sh_encoder = lambda x: components_from_spherical_harmonics(self.degree, x)
            sh_dim = num_sh_bases(self.degree)
            encoded_dim = sh_dim * 2  # wi and wo
        else:
            encoded_dim = 6  # wi (3) + wo (3)
        
        # Input: encoded directions (or geometry latent + directions)
        input_dim = encoded_dim + geometry_latent_dim if not use_pos_enc else encoded_dim
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        for hidden_dim in cfg.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if cfg.activation.lower() == "relu":
                layers.append(nn.ReLU())
            else:
                layers.append(nn.LeakyReLU(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, cfg.output_channels))  # 2D UV offset
        layers.append(nn.Tanh())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        wi: torch.Tensor,
        wo: torch.Tensor,
        geometry_latent: torch.Tensor = None
    ):
        """
        Predict UV offset based on directions.
        
        Args:
            wi: [B, 3] incoming light direction (local or world space)
            wo: [B, 3] outgoing view direction (local or world space)
            geometry_latent: [B, geometry_latent_dim] geometry latent code
        
        Returns:
            uv_offset: [B, 2] UV offset
        """
        if self.use_pos_enc:
            wi_enc = self.sh_encoder(wi)
            wo_enc = self.sh_encoder(wo)
            mlp_input = torch.cat([wi_enc, wo_enc], dim=-1)
        else:
            if geometry_latent is not None:
                mlp_input = torch.cat([geometry_latent, wi, wo], dim=-1)
            else:
                mlp_input = torch.cat([wi, wo], dim=-1)
        
        return self.mlp(mlp_input)

class AnisotropicLatentTexturedModel(LightningModule):
    """
    Complete material model combining all components.
    Provides the same API as the original class for compatibility.
    """
    def __init__(self, cfg):
        super().__init__()
        
        # Store configuration
        self.cfg = cfg
        self.latent_dim = cfg.latent_dim
        self.colorful_texture = cfg.colorful_texture
        self.larger_latent_dim = cfg.larger_latent_dim
        self.different_decoder = cfg.different_decoder
        self.predict_frame = cfg.predict_frame
        # Inference-only toggle. When False, eval_brdf skips
        # extract_frame_from_latent and uses Mitsuba's geometric (normal, TBN)
        # instead. Does NOT change construction — the latent layout still
        # allocates the 6 frame channels so the ckpt loads cleanly.
        self.use_predicted_frame_at_eval = getattr(cfg, 'use_predicted_frame_at_eval', True)
        self.gt_frame = cfg.gt_frame
        self.anisotropic = True
        self.Gaussian_blur = cfg.Gaussian_blur
        self.learnable_factor = cfg.learnable_factor
        if self.learnable_factor:
            self.factor = nn.Parameter(torch.ones(3))

        self.mono_brdf = cfg.mono_brdf

        # Neural geometry settings
        self.neural_geometry_enabled = cfg.neural_geometry.enable
        self.geometry_latent_dim = cfg.neural_geometry.latent_dim if self.neural_geometry_enabled else 0
        self.local_wi_wo = cfg.neural_geometry.local_wi_wo if self.neural_geometry_enabled else False
        self.neural_geometry_pos_enc = cfg.neural_geometry.positional_encoding if self.neural_geometry_enabled else False
        self.recompute_frame = cfg.neural_geometry.recompute_frame if self.neural_geometry_enabled else False
        self.neural_geometry_factor = cfg.neural_geometry.factor if self.neural_geometry_enabled else 0.4
        # Inference-only toggle. When False, eval_brdf skips the
        # neural-geometry UV offset + texture re-sample step. Does NOT change
        # construction — the NeuralGeometry MLP still exists and its weights
        # still load from the ckpt; only its eval-time call is bypassed.
        self.use_neural_geometry_at_eval = (
            getattr(cfg.neural_geometry, 'use_at_eval', True)
            if self.neural_geometry_enabled else False
        )
        # Calculate total latent dimension
        if self.colorful_texture and self.larger_latent_dim:
            brdf_latent_dim = self.latent_dim * 3
        elif self.different_decoder:
            brdf_latent_dim = self.latent_dim * 3
        else:
            brdf_latent_dim = self.latent_dim
        self.brdf_latent_dim = brdf_latent_dim

        # Add frame dimensions if predicting TBN
        total_latent_dim = brdf_latent_dim
        if self.predict_frame:
            total_latent_dim += 6  # normal (3) + tangent (3)
        
        # Add geometry latent dimensions
        if self.neural_geometry_enabled:
            total_latent_dim += self.geometry_latent_dim
        
        if self.mono_brdf:
            total_latent_dim += 3 # add three color channels
        # 1. Create LatentTexture
        self.texture_resolution = getattr(cfg, 'texture_resolution', 256)
        blur_config = {
            'blur_sigma0': 2.0,
            'blur_half_life': 3333
        } if self.Gaussian_blur else None
        
        self.latent_texture = LatentTexture(
            resolution=self.texture_resolution,
            latent_dim=total_latent_dim,
            predict_frame=self.predict_frame,
            init_std=0.1,
            blur_config=blur_config
        )
        
        # 2. Create BRDFDecoder
        self.decoder = BRDFDecoder(
            cfg=cfg.decoder,
            latent_dim=self.latent_dim,
            use_pos_enc=True,
            different_decoder=self.different_decoder
        )
        
        # 3. Create NeuralGeometry if enabled
        if self.neural_geometry_enabled:
            self.neural_geometry = NeuralGeometry(
                cfg=cfg.neural_geometry,
                geometry_latent_dim=self.geometry_latent_dim,
                use_local_wi_wo=self.local_wi_wo,
                use_pos_enc=self.neural_geometry_pos_enc
            )
        else:
            self.neural_geometry = None
        
        # 4. Proxy BRDF for importance sampling
        #self.proxy_brdf = ProxyPBRBRDF()
        
        # 5. Latent bank mode (alternative to texture-based sampling)
        self.use_latent_bank = getattr(cfg, 'use_latent_bank', False)
        
        if self.use_latent_bank:
            observations_folder = cfg.observations_folder
            
            # Compute global point offset for this material
            self.global_point_offset = self._compute_global_point_offset(observations_folder)
            
            # Load point positions from observations
            self.point_positions = self._load_point_positions(observations_folder)
            num_points = self.point_positions.shape[0]
            print(f"[LatentBank] Loaded {num_points:,} point positions from {observations_folder}")
            
            # Build KD-tree for efficient nearest neighbor lookup
            from scipy.spatial import cKDTree
            self.kdtree = cKDTree(self.point_positions.numpy())
            print(f"[LatentBank] Built KD-tree for {num_points:,} points")
            
            # Latent bank will be loaded from checkpoint - don't initialize here
            self.point_latent_bank = None
            self.bank_latent_dim = brdf_latent_dim + (6 if self.predict_frame else 0)
            print(f"[LatentBank] Latent bank will be loaded from checkpoint (latent_dim={self.bank_latent_dim})")
    
    # ------------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------------
    def compute_uv(self, pos: torch.Tensor, width: float = 0.4, length: float = 0.4):
        """
        Compute UV coordinates from 3D positions.
        
        Args:
            pos: [B, 3] 3D positions
            width: Width of the surface
            length: Length of the surface
        
        Returns:
            uv: [B, 2] UV coordinates in [0, 1]
        """
        half_w = width * 0.5
        half_l = length * 0.5
        x, z = pos[:, 0], pos[:, 2]
        
        u = (x + half_w) / width
        v = (z + half_l) / length
        
        return torch.stack([u, v], dim=-1)
    
    def world_to_local(
        self,
        v: torch.Tensor,
        normal: torch.Tensor,
        tangent: torch.Tensor
    ):
        """
        Transform vector from world space to local tangent space.
        
        Args:
            v: [B, 3] vector in world space
            normal: [B, 3] normal in world space
            tangent: [B, 3] tangent in world space
        
        Returns:
            v_local: [B, 3] vector in local space
        """
        if tangent is None:
            # Generate arbitrary tangent perpendicular to normal
            tangent = torch.cross(
                normal,
                torch.tensor([0.0, 0.0, 1.0], device=normal.device).expand_as(normal)
            )
        
        tangent_len = tangent.norm(dim=-1, keepdim=True)
        tangent = tangent / (tangent_len + 1e-8)
        bitangent = torch.cross(normal, tangent)
        
        v_local = torch.stack([
            (v * tangent).sum(dim=-1),
            (v * bitangent).sum(dim=-1),
            (v * normal).sum(dim=-1)
        ], dim=-1)
        
        return v_local
    
    def extract_frame_from_latent(self, latent: torch.Tensor):
        """
        Extract and orthonormalize normal and tangent from latent code.
        
        Args:
            latent: [B, total_dim] latent code with last 6 dims as normal+tangent
        
        Returns:
            normal: [B, 3] normalized normal vector
            tangent: [B, 3] normalized tangent vector (orthogonal to normal)
        """
        predicted_normal = latent[..., -6:-3]
        predicted_tangent = latent[..., -3:]
        
        # Normalize
        predicted_normal = NF.normalize(predicted_normal, dim=-1)
        predicted_tangent = NF.normalize(predicted_tangent, dim=-1)
        
        # Gram-Schmidt orthogonalization
        predicted_tangent = predicted_tangent - \
            torch.sum(predicted_tangent * predicted_normal, dim=-1, keepdim=True) * predicted_normal
        predicted_tangent = NF.normalize(predicted_tangent, dim=-1)
        
        return predicted_normal, predicted_tangent
    
    def _sample_from_texture(self, uv: torch.Tensor, texture: torch.Tensor):
        """
        Sample latent from a given texture at UV coordinates.
        
        Args:
            uv: [B, 2] UV coordinates in [0, 1]
            texture: [1, D, H, W] texture to sample from
        
        Returns:
            latent: [B, D] sampled latent codes
        """
        # Convert UV to grid coordinates for F.grid_sample
        grid_coords = uv * 2.0 - 1.0  # [B, 2] -> [-1, 1]
        grid_coords = grid_coords.unsqueeze(1).unsqueeze(0)  # [1, 1, B, 2]
        
        # Bilinear sampling
        latent = NF.grid_sample(
            texture,  # [1, D, H, W]
            grid_coords,  # [1, 1, B, 2]
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )  # [1, D, 1, B]
        
        # Reshape to [B, D]
        latent = latent.squeeze(0).squeeze(-1).transpose(0, 1)
        
        return latent
    
    # ------------------------------------------------------------------------
    # Latent Bank Methods (for use_latent_bank mode)
    # ------------------------------------------------------------------------
    def _compute_global_point_offset(self, observations_folder: str) -> int:
        """
        Compute the global point offset for this material based on folder path.
        
        The observations folder path contains the material ID (e.g., 'data/0/observations/').
        The offset is the sum of num_points from all materials with ID < current material ID.
        
        Args:
            observations_folder: Path to observations folder (e.g., 'data/0/observations/')
        
        Returns:
            offset: int, global point offset for indexing into the latent bank
        """
        import json
        from pathlib import Path
        
        obs_folder = Path(observations_folder)
        material_folder = obs_folder.parent
        data_folder = material_folder.parent
        
        # Extract current material ID from folder name
        current_material_id = int(material_folder.name)
        
        # Find all material folders with ID < current material ID
        offset = 0
        for mat_id in range(current_material_id):
            mat_folder = data_folder / str(mat_id)
            metadata_path = mat_folder / "point_metadata.json"
            
            if not metadata_path.exists():
                print(f"[LatentBank] Warning: {metadata_path} not found, skipping material {mat_id}")
                continue
            
            try:
                with open(metadata_path, 'r') as f:
                    point_meta = json.load(f)
                num_points = point_meta['num_points']
                offset += num_points
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[LatentBank] Warning: Failed to read {metadata_path}: {e}, skipping material {mat_id}")
                continue
        
        print(f"[LatentBank] Material {current_material_id}: global point offset = {offset:,}")
        return offset
    
    def _load_point_positions(self, observations_folder: str) -> torch.Tensor:
        """
        Load point positions from point_positions.npz file.
        
        Args:
            observations_folder: Path to folder containing observation chunks
        
        Returns:
            point_positions: [num_unique_points, 3] tensor of unique point positions
        """
        import numpy as np
        from pathlib import Path
        
        obs_folder = Path(observations_folder)
        material_folder = obs_folder.parent
        positions_path = material_folder / "point_positions.npz"
        
        data = np.load(positions_path)
        point_positions = torch.from_numpy(data['positions']).float()
        print(f"[LatentBank] Loaded point positions: {point_positions.shape}")
        
        return point_positions
    
    def _find_nearest_point_ids(self, pos: torch.Tensor, k: int = 1) -> tuple:
        """
        Find K nearest point IDs for query positions using KD-tree.
        
        Args:
            pos: [B, 3] query positions
            k: number of nearest neighbors
        
        Returns:
            distances: [B, k] distances to nearest points
            point_ids: [B, k] tensor of nearest point IDs
        """
        # Query KD-tree (CPU operation)
        distances, point_ids = self.kdtree.query(pos.detach().cpu().numpy(), k=k)
        distances = torch.from_numpy(distances).to(pos.device).float()
        point_ids = torch.from_numpy(point_ids).to(pos.device)
        return distances, point_ids
    
    def _sample_from_latent_bank(self, pos: torch.Tensor, k: int = 8) -> torch.Tensor:
        """
        Sample latent codes from point bank using K-NN with inverse distance weighting.
        
        Args:
            pos: [B, 3] query positions
            k: number of nearest neighbors for interpolation
        
        Returns:
            latent: [B, latent_dim] interpolated latent codes
        """
        # Find K nearest point IDs and distances (local indices for this material)
        distances, local_point_ids = self._find_nearest_point_ids(pos, k=k)  # [B, k], [B, k]
        
        # Convert local point IDs to global point IDs for latent bank lookup
        global_point_ids = local_point_ids + self.global_point_offset
        
        # Lookup latents for all K neighbors using global IDs
        latents = self.point_latent_bank(global_point_ids)  # [B, k, latent_dim]
        
        # Inverse distance weighting
        eps = 1e-8
        weights = 1.0 / (distances + eps)  # [B, k]
        weights = weights / weights.sum(dim=-1, keepdim=True)  # normalize to sum to 1
        
        # Weighted sum of latents
        latent = (weights.unsqueeze(-1) * latents).sum(dim=1)  # [B, latent_dim]
        
        return latent
    
    # ------------------------------------------------------------------------
    # Main BRDF evaluation API
    # ------------------------------------------------------------------------
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
        dp_dv=None
    ):
        """
        Evaluate BRDF at given geometry and directions.
        
        Args:
            gt_params: Ground truth parameters (optional)
            pos: [B, 3] 3D positions
            wi: [B, 3] incoming light directions (world space)
            wo: [B, 3] outgoing view directions (world space)
            normal: [B, 3] normals (world space)
            uv: [B, 2] UV coordinates
            TBN: [B, 3, 3] tangent-bitangent-normal frame
            latent: Ignored (latent is sampled internally)
            batch_mask: Batch mask for batched operations
            footprint_vis: Footprint for mipmap level selection
            dp_du, dp_dv: Ray differentials
        
        Returns:
            brdf: [B, 3] BRDF values
            pdf: [B, 1] probability density
            uv_offset: [B, 2] UV offset (if neural geometry enabled, else zeros)
        """
        # NB: this NoL/NoV only feeds the returned `pdf`; mlp.py discards that
        # value, so the choice of normal here doesn't affect the rendered image.
        NoL = (wi * normal).sum(-1, keepdim=True)
        NoV = (wo * normal).sum(-1, keepdim=True)
        
        # Initialize uv_offset (used in return, always needed)
        uv_offset = torch.zeros_like(uv)
        
        # =====================================================================
        # LATENT BANK MODE: Sample latent from point bank using position
        # =====================================================================
        if self.use_latent_bank:
            # Sample latent from latent bank using nearest neighbor lookup
            latent = self._sample_from_latent_bank(pos)
            
            # Extract frame from latent if predicting frame
            if self.predict_frame:
                predicted_normal, predicted_tangent = self.extract_frame_from_latent(latent)
            else:
                # Use geometry normal and tangent
                predicted_normal = normal
                predicted_tangent = TBN[:, :, 0] if TBN is not None else None
            
            # No neural geometry in latent bank mode
        
        # =====================================================================
        # TEXTURE MODE: Sample latent from texture using UV coordinates
        # =====================================================================
        else:
            # 1. Get the (optionally blurred) texture ONCE
            if self.training and self.Gaussian_blur:
                tex = self.latent_texture.apply_gaussian_blur(self.global_step)
            else:
                tex = self.latent_texture.params
            
            # 2. Sample latent from the texture
            latent = self._sample_from_texture(uv, tex)
            print("latent",latent.shape)

            # 3. Extract frame from latent if predicting frame.
            #    use_predicted_frame_at_eval=False bypasses the trained frame
            #    at inference (latent layout is unchanged — still 6 frame dims).
            if self.predict_frame and self.use_predicted_frame_at_eval:
                predicted_normal, predicted_tangent = self.extract_frame_from_latent(latent)
            else:
                # Use geometry normal and tangent
                predicted_normal = normal
                predicted_tangent = TBN[:, :, 0] if TBN is not None else None

            # 4. Predict UV offset if neural geometry is enabled.
            #    use_neural_geometry_at_eval=False bypasses the UV-offset step
            #    at inference without altering construction or ckpt loading.
            if self.neural_geometry_enabled and self.use_neural_geometry_at_eval:
                # Extract geometry latent
                geometry_latent = latent[..., -6-self.geometry_latent_dim:-6] if self.predict_frame else \
                                 latent[..., -self.geometry_latent_dim:]
                
                # Get directions for geometry network
                if self.local_wi_wo:
                    wi_for_geo = self.world_to_local(wi, predicted_normal, predicted_tangent)
                    wo_for_geo = self.world_to_local(wo, predicted_normal, predicted_tangent)
                else:
                    wi_for_geo = wi
                    wo_for_geo = wo
                
                # Predict UV offset
                uv_offset = self.neural_geometry(wi_for_geo, wo_for_geo, geometry_latent) * self.neural_geometry_factor
                uv = uv + uv_offset
                uv = ((uv%1)+1)%1
                # Sample from the SAME blurred texture
                latent = self._sample_from_texture(uv, tex)
                
                # Recompute frame if needed
                if self.recompute_frame and self.predict_frame and self.use_predicted_frame_at_eval:
                    predicted_normal, predicted_tangent = self.extract_frame_from_latent(latent)
        
        # =====================================================================
        # Common code path for both modes
        # =====================================================================
        # 5. Transform directions to local space
        wi_local = self.world_to_local(wi, predicted_normal, predicted_tangent)
        wo_local = self.world_to_local(wo, predicted_normal, predicted_tangent)
        local_normal = torch.zeros_like(wi_local)
        local_normal[..., 2] = 1.0  # (0, 0, 1) in local space
        
        # 5. Encode directions
        enc_dir = self.decoder.encode_directions(wi_local, wo_local, local_normal)     
        
        # 6. Extract BRDF latent and decode
        if self.different_decoder:
            brdf = self.decoder(enc_dir, latent[..., :self.brdf_latent_dim])
        elif self.colorful_texture:
            brdf = self.decoder(enc_dir, latent[..., :self.latent_dim])
        else:
            brdf = self.decoder(enc_dir, latent[..., :self.latent_dim])
            brdf = brdf.repeat(1, 3)  # Replicate to RGB
        
        # 7. Calculate PDF (cosine-weighted)
        pdf = NoL / math.pi
        if self.learnable_factor:
            brdf = brdf * self.factor
        
        cos_theta_i_pred = (wi * predicted_normal).sum(-1, keepdim=True).clamp(min=0.0)
        cos_theta_o_pred = (wo * predicted_normal).sum(-1, keepdim=True)
        brdf = brdf * cos_theta_i_pred
        return brdf, predicted_normal, pdf, uv_offset, cos_theta_i_pred, cos_theta_o_pred
    
    
    # ------------------------------------------------------------------------
    # Save/Load functionality
    # ------------------------------------------------------------------------
    def save_latent(self, path: str):
        """Save latent texture to file"""
        self.latent_texture.save(path)
    
    def load_latent(self, path: str):
        """Load latent texture from file"""
        self.latent_texture.load(path)

            
class MERLBRDF(LightningModule):
    """
    Multi-material BRDF model using auto-decoder architecture.
    - Per-material latent codes (M materials)
    - Per-point latent codes (sum of points across all materials)
    - Shared MLP decoder across all materials
    
    Expected folder structure:
        data_folder/
            0/                      # material_id = 0
                point_metadata.json # contains {"num_points": N, ...}
            1/                      # material_id = 1
                point_metadata.json
            ...
    """
    def __init__(self, cfg):
        super().__init__()
        
        # Store configuration
        self.cfg = cfg
        self.num_materials = cfg.num_materials
        data_folder = getattr(cfg, 'data_folder', None)
        self.start_material_id = cfg.start_material_id
        # Latent dimensions
        self.latent_dim = cfg.latent_dim
        self.predict_frame = cfg.predict_frame
        self.total_latent_dim = self.latent_dim + (6 if self.predict_frame else 0)
        
        self.random_latent = 0.1 *torch.randn((self.total_latent_dim)).cuda()
        # BRDF decoder settings
        self.use_pos_enc = cfg.use_pos_enc
        self.different_decoder = cfg.different_decoder

        total_points=1920*1920
        self.point_latent_bank = nn.Embedding(
            num_embeddings=total_points,
            embedding_dim=self.total_latent_dim
        )
        nn.init.normal_(self.point_latent_bank.weight, mean=0.0, std=cfg.init_std)
        
        if self.predict_frame:
            # Last 6 dimensions: normal (0,0,1) and tangent (0,1,0)
            with torch.no_grad():
                self.point_latent_bank.weight[:, -6:-3] = torch.tensor([0.0, 0.0, 1.0])  # normal
                self.point_latent_bank.weight[:, -3:] = torch.tensor([0.0, 1.0, 0.0])    # tangent
        
        # Shared BRDF decoder
        self.decoder = BRDFDecoder(
            cfg=cfg.decoder,
            latent_dim=self.latent_dim,
            use_pos_enc=self.use_pos_enc,
            different_decoder=self.different_decoder
        )
        
        print("Initialization complete!")

    
    def directions_to_rusinkiewicz(self,wi, wo, eps=1e-6):
        """
        Convert incoming (wi) and outgoing (wo) directions to
        Rusinkiewicz parameterization.

        Args:
            wi: [B, 3] incoming directions (not necessarily normalized)
            wo: [B, 3] outgoing directions (not necessarily normalized)

        Returns:
            theta_h: [B] half-angle in [0, pi/2]
            theta_d: [B] difference angle in [0, pi/2]
            phi_d:   [B] azimuthal difference in [0, 2*pi)
        """

        # ------------------------------------------------------------
        # 1. Normalize directions
        # ------------------------------------------------------------
        wi = NF.normalize(wi, dim=-1)
        wo = NF.normalize(wo, dim=-1)

        # ------------------------------------------------------------
        # 2. Half vector
        # ------------------------------------------------------------
        h = wi + wo
        h = NF.normalize(h, dim=-1)

        # ------------------------------------------------------------
        # 3. theta_h = angle between h and surface normal (0,0,1)
        # ------------------------------------------------------------
        theta_h = torch.acos(torch.clamp(h[..., 2], -1.0, 1.0))

        # ------------------------------------------------------------
        # 4. theta_d = half-angle between wi and wo
        # ------------------------------------------------------------
        cos_wo_wi = torch.sum(wi * wo, dim=-1)
        theta_d = 0.5 * torch.acos(torch.clamp(cos_wo_wi, -1.0, 1.0))

        # ------------------------------------------------------------
        # 5. Build local frame with h as z-axis
        # ------------------------------------------------------------
        # Choose a helper vector not parallel to h
        up = torch.zeros_like(h)
        up[..., 2] = 1.0

        # If h is too close to (0,0,1), switch helper vector
        mask = torch.abs(h[..., 2]) > 0.999
        up[mask] = torch.tensor([1.0, 0.0, 0.0], device=h.device)

        x = torch.cross(up, h, dim=-1)
        x = NF.normalize(x, dim=-1)

        y = torch.cross(h, x, dim=-1)

        # ------------------------------------------------------------
        # 6. Transform wi and wo to h-local coordinates
        # ------------------------------------------------------------
        def to_local(v):
            return torch.stack([
                torch.sum(v * x, dim=-1),
                torch.sum(v * y, dim=-1),
                torch.sum(v * h, dim=-1),
            ], dim=-1)

        wi_l = to_local(wi)
        wo_l = to_local(wo)

        # ------------------------------------------------------------
        # 7. Azimuthal difference phi_d
        # ------------------------------------------------------------
        phi_i = torch.atan2(wi_l[..., 1], wi_l[..., 0])
        phi_o = torch.atan2(wo_l[..., 1], wo_l[..., 0])

        phi_d = phi_i - phi_o
        phi_d = torch.remainder(phi_d, 2.0 * math.pi)

        return theta_h, theta_d, phi_d
    
    def rotate_to_canonical_frame(self,wi, wo):
        """
        Rotate wi and wo around z-axis by the same angle to make wi.x = 0.

        This exploits isotropy by rotating the coordinate frame so that the incoming 
        direction lies in the yz-plane (x component = 0). The outgoing direction is 
        rotated by the same amount.

        Args:
            wi: [B, 3] incoming direction vectors
            wo: [B, 3] outgoing direction vectors

        Returns:
            wi_rotated: [B, 3] rotated incoming vectors with x component = 0
            wo_rotated: [B, 3] rotated outgoing vectors
        """
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

    def eval_brdf(
        self,
        wi,
        wo,
        material_id,
        weights
    ):
        """
        Evaluate BRDF at given geometry and directions.
        
        Args:
            pos: [B, 3] 3D positions
            wi: [B, 3] incoming light directions (world space)
            wo: [B, 3] outgoing view directions (world space)
            normal: [B, 3] normals (world space)
            latent: Ignored (latents retrieved from banks)
            point_ids: [B] LOCAL point indices (per-material, from dataloader)
            material_ids: [B] material indices (required for global point ID computation)
        
        Returns:
            brdf: [B, 3] BRDF values
            normal: [B, 3] normals (local space)
            pdf: [B, 1] probability density
        """
        wi,wo=self.rotate_to_canonical_frame(wi,wo)
        print("using neural merl model")
        # Retrieve latents from banks
        #print("material_id",material_id)
        NoL=wi[:,2:3].repeat(1, 3)
        NoV=wo[:,2:3].repeat(1, 3)
        
        print("material_id",material_id)
        latent = self.point_latent_bank(material_id)        # [B, latent_dim]
        print("latent",latent.shape)
        latent=(latent*weights.unsqueeze(-1)).sum(dim=1)
        print("latent_processed",latent.shape)
        # [B, latent_dim]
        #latent[:]=self.random_latent

        normal_local = torch.zeros_like(wi)
        normal_local[..., 2] = 1.0  # Normal is always (0,0,1) in local space
        
        # Encode directions
        enc_dir = self.decoder.encode_directions(wi, wo, normal_local)
        
        # Decode BRDF
        if self.different_decoder:
            # Decode each channel separately
            brdf_r = self.decoder(enc_dir, latent[:,:self.latent_dim], channel='r')
            brdf_g = self.decoder(enc_dir, latent[:,:self.latent_dim], channel='g')
            brdf_b = self.decoder(enc_dir, latent[:,:self.latent_dim], channel='b')
            brdf = torch.cat([brdf_r, brdf_g, brdf_b], dim=-1)  # [B, 3]
        else:
            brdf = self.decoder(enc_dir, latent[:, :self.latent_dim])  # [B, 1] or [B, 3]
            if brdf.shape[-1] == 1:
                brdf = brdf.expand(-1, 3)  # Expand to RGB
        
        
        return brdf*NoL
    
    
    '''
    def eval_brdf_wiwo(
        self,
        wi,
        wo,
        material_id,
    ):
        wi = NF.normalize(wi, dim=-1)
        wo = NF.normalize(wo, dim=-1)
        
        # Convert wi to spherical coordinates
        theta_in = torch.acos(torch.clamp(wi[..., 2], -1.0, 1.0))
        phi_in = torch.atan2(wi[..., 1], wi[..., 0])
        
        # Convert wo to spherical coordinates
        theta_out = torch.acos(torch.clamp(wo[..., 2], -1.0, 1.0))
        phi_out = torch.atan2(wo[..., 1], wo[..., 0])
        
        theta_h, phi_h, theta_d, phi_d = _std_coords_to_half_diff_coords(
            theta_in, phi_in, theta_out, phi_out)
        
        vectors = self.rangles_to_rvectors(theta_h, theta_d, phi_d)
        
        # Split into h_vec and d_vec
        h_vec = vectors[:, :3]  # [batch_size, 3]
        d_vec = vectors[:, 3:]  # [batch_size, 3]
        return self.eval_brdf(h_vec,d_vec,material_id)
    ''' 
    
    def eval_brdf_wiwo(
        self,
        wi,
        wo,
        material_id,
    ):
        wi = NF.normalize(wi, dim=-1)
        wo = NF.normalize(wo, dim=-1)
        
        rgb,phi_d,theta_d,theta_h = self.merl_interface.lookup_wiwo(wi, wo, material_id)
        
        vectors = self.rangles_to_rvectors(theta_h, theta_d, phi_d)
        
        # Split into h_vec and d_vec
        h_vec = vectors[:, :3]  # [batch_size, 3]
        d_vec = vectors[:, 3:]  # [batch_size, 3]
        return self.eval_brdf(h_vec,d_vec,material_id)
    
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
        if not isinstance(theta_h, torch.Tensor):
            theta_h = torch.tensor(theta_h, device=self.device, dtype=torch.float32)
        if not isinstance(theta_d, torch.Tensor):
            theta_d = torch.tensor(theta_d, device=self.device, dtype=torch.float32)
        if not isinstance(phi_d, torch.Tensor):
            phi_d = torch.tensor(phi_d, device=self.device, dtype=torch.float32)
        
        hx = torch.sin(theta_h) * torch.cos(torch.zeros_like(theta_h))
        hy = torch.sin(theta_h) * torch.sin(torch.zeros_like(theta_h))
        hz = torch.cos(theta_h)
        dx = torch.sin(theta_d) * torch.cos(phi_d)
        dy = torch.sin(theta_d) * torch.sin(phi_d)
        dz = torch.cos(theta_d)
        
        return torch.stack([hx, hy, hz, dx, dy, dz], dim=-1)
    
    
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
    
    def __init__(self, brdf_dir, device='cuda'):
        """
        Initialize MERL BRDF interface and load all materials from directory.
        
        Args:
            brdf_dir: Path to directory containing .binary BRDF files
            device: Device to store tensors ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        self.brdf_dir = Path(brdf_dir)
        
        if not self.brdf_dir.exists():
            raise FileNotFoundError(f"BRDF directory not found: {brdf_dir}")
        
        if not self.brdf_dir.is_dir():
            raise ValueError(f"Path is not a directory: {brdf_dir}")
        
        # Load all BRDF materials
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
        in_vec = NF.normalize(in_vec, dim=-1)
        
        # Compute out vector (lines 93-99)
        out_vec_z = torch.cos(theta_out)
        proj_out_vec = torch.sin(theta_out)
        out_vec_x = proj_out_vec * torch.cos(phi_out)
        out_vec_y = proj_out_vec * torch.sin(phi_out)
        out_vec = torch.stack([out_vec_x, out_vec_y, out_vec_z], dim=-1)
        out_vec = NF.normalize(out_vec, dim=-1)
        
        # Compute halfway vector (lines 102-107)
        half_x = (in_vec[..., 0] + out_vec[..., 0]) / 2.0
        half_y = (in_vec[..., 1] + out_vec[..., 1]) / 2.0
        half_z = (in_vec[..., 2] + out_vec[..., 2]) / 2.0
        half = torch.stack([half_x, half_y, half_z], dim=-1)
        half = NF.normalize(half, dim=-1)
        
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
        theta_h, phi_h, theta_d, phi_d = _std_coords_to_half_diff_coords(
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
        
        return rgb.float()
    
    def compute_svbrdf_pdf(self, albedo, roughness, metallic, wi, wo, normal):
        roughness=roughness.unsqueeze(1)
        metallic=metallic.unsqueeze(1)
        h = NF.normalize(wi+wo,dim=-1)
        
        NoL_rough=(wi*normal).sum(-1,keepdim=True)
        NoV_rough=(wo*normal).sum(-1,keepdim=True)
        print("NoL_rough",(NoL_rough < 0).sum().item())
        print("NoV_rough",(NoV_rough < 0).sum().item())
        NoL = (wi*normal).sum(-1,keepdim=True).relu()
        NoV = (wo*normal).sum(-1,keepdim=True).relu()
        VoH = (wo*h).sum(-1,keepdim=True).relu()
        NoH = (normal*h).sum(-1,keepdim=True).relu()

        D = D_GGX(NoH,roughness)
        D = D_GGX(NoH,roughness)
        pdf_spec = D.data/(4*VoH.clamp_min(1e-4))*NoH
        pdf_diff = NoL/math.pi
        pdf = 0.5*pdf_spec + 0.5*pdf_diff

        kd = albedo*(1-metallic)
        ks = 0.04*(1-metallic) + albedo*metallic

        G = G_Smith(NoV,NoL,roughness)
        F = fresnelSchlick(VoH,ks)
        brdf_diff = kd/math.pi*NoL
        brdf_spec = D*G*F/4.0*NoL

        brdf = brdf_diff + brdf_spec
        
        if torch.isnan(brdf).any() or torch.isinf(brdf).any():
            import pdb; pdb.set_trace()

        return brdf, pdf
    
    def eval_brdf(self, wi, wo, material_id):
        """
        Look up BRDF values for given incoming/outgoing direction vectors.
        
        This is a wrapper around lookup() that converts direction vectors to angles.
        
        Args:
            wi: [B, 3] incoming light directions (normalized, pointing toward surface)
            wo: [B, 3] outgoing view directions (normalized, pointing away from surface)
        
        Returns:
            rgb: [B, 3] BRDF RGB values
        """
        print("using_reference")
        NoL=wi[:,2:3].repeat(1, 3)
        NoV=wo[:,2:3].repeat(1, 3)
        normal = torch.tensor([0.0, 0.0, 1.0], device=wi.device, dtype=wi.dtype)
        normal = normal.unsqueeze(0).expand(wi.shape[0], -1)
        normal=NF.normalize(normal, dim=1)
        roughness=torch.tensor([0.4], device=wi.device, dtype=wi.dtype)
        metallic=torch.tensor([0.8], device=wi.device, dtype=wi.dtype)
        albedo=torch.tensor([1.0, 1.0, 1.0], device=wi.device, dtype=wi.dtype)
        F0=torch.tensor([0.5], device=wi.device, dtype=wi.dtype)
        brdf,_=self.compute_svbrdf_pdf(albedo, roughness, metallic, wi, wo, normal)
        #return brdf
        # Normalize directions
        wi = NF.normalize(wi, dim=-1)
        wo = NF.normalize(wo, dim=-1)
        
        # Convert wi to spherical coordinates
        theta_in = torch.acos(torch.clamp(wi[..., 2], -1.0, 1.0))
        phi_in = torch.atan2(wi[..., 1], wi[..., 0])
        
        # Convert wo to spherical coordinates
        theta_out = torch.acos(torch.clamp(wo[..., 2], -1.0, 1.0))
        phi_out = torch.atan2(wo[..., 1], wo[..., 0])
        
        # Count the number of phi values within [0, pi/2] for both phi_in and phi_out
        count_theta_in = torch.sum((theta_in >= 0) & (theta_in <= torch.pi / 2)).item()
        count_theta_out = torch.sum((theta_out >= 0) & (theta_out <= torch.pi / 2)).item()

        # Print the counts
        print(f"Number of phi_in values within [0, pi/2]: {count_theta_in}")
        print(f"Number of phi_out values within [0, pi/2]: {count_theta_out}")
        
        # Call the main lookup function
        return self.lookup(theta_in, phi_in, theta_out, phi_out, material_id)*NoL
    
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

    
class MerlTorch:
    """
    PyTorch version of Merl class with multi-material support.
    
    Supports multiple materials and batched operations on GPU/CPU.
    All operations use PyTorch tensors for efficient computation.
    
    Args:
        merl_files: Path to directory containing .binary BRDF files
        device: Device to store tensors ('cuda' or 'cpu')
    """
    sampling_theta_h = 90
    sampling_theta_d = 90
    sampling_phi_d = 180
    
    scale = torch.tensor([1. / 1500, 1.15 / 1500, 1.66 / 1500])
    
    def __init__(self, merl_files, device='cuda'):
        """
        Initialize and load MERL BRDF file(s) from directory.
        
        Args:
            merl_files: Path to directory containing .binary BRDF files
            device: Device to store tensors on ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        self.scale = self.scale.to(self.device)
        
        # If it's a directory, load all .binary files from it
        merl_path = Path(merl_files)
        if merl_path.is_dir():
            merl_files = sorted(list(merl_path.glob('*.binary')))
            if not merl_files:
                raise ValueError(f"No .binary files found in directory: {merl_path}")
        elif isinstance(merl_files, (str, Path)):
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

    def eval_brdf(self, wi, wo, material_id):
        """
        Look up BRDF values for given incoming/outgoing direction vectors.
        
        This is a wrapper around lookup() that converts direction vectors to angles.
        
        Args:
            wi: [B, 3] incoming light directions (normalized, pointing toward surface)
            wo: [B, 3] outgoing view directions (normalized, pointing away from surface)
        
        Returns:
            rgb: [B, 3] BRDF RGB values
        """
        print("using_reference")
        #return brdf
        # Normalize directions
        wi = NF.normalize(wi, dim=-1)
        wo = NF.normalize(wo, dim=-1)
        normal_local = torch.zeros_like(wi)
        normal_local[..., 2] = 1.0  # Normal is always (0,0,1) in local space
        NoL=(wi*normal_local).sum(-1,keepdim=True)
        
        # Convert wi to spherical coordinates
        theta_in = torch.acos(torch.clamp(wi[..., 2], -1.0, 1.0))
        phi_in = torch.atan2(wi[..., 1], wi[..., 0])
        
        # Convert wo to spherical coordinates
        theta_out = torch.acos(torch.clamp(wo[..., 2], -1.0, 1.0))
        phi_out = torch.atan2(wo[..., 1], wo[..., 0])
        
        # Count the number of phi values within [0, pi/2] for both phi_in and phi_out
        count_theta_in = torch.sum((theta_in >= 0) & (theta_in <= torch.pi / 2)).item()
        count_theta_out = torch.sum((theta_out >= 0) & (theta_out <= torch.pi / 2)).item()

        # Print the counts
        print(f"Number of phi_in values within [0, pi/2]: {count_theta_in}")
        print(f"Number of phi_out values within [0, pi/2]: {count_theta_out}")
        theta_h, phi_h, theta_d, phi_d = _std_coords_to_half_diff_coords(
            theta_in, phi_in, theta_out, phi_out)

        # Call the main lookup function
        return self.eval_interp(theta_h, theta_d, phi_d,material_id)*NoL
    
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

    def wiwi2rangles(self, wi,wo):
        theta_in = torch.acos(torch.clamp(wi[..., 2], -1.0, 1.0))
        phi_in = torch.atan2(wi[..., 1], wi[..., 0])
        
        # Convert wo to spherical coordinates
        theta_out = torch.acos(torch.clamp(wo[..., 2], -1.0, 1.0))
        phi_out = torch.atan2(wo[..., 1], wo[..., 0])
        theta_h, phi_h, theta_d, phi_d = _std_coords_to_half_diff_coords(
            theta_in, phi_in, theta_out, phi_out)
        return phi_d,theta_d,theta_h