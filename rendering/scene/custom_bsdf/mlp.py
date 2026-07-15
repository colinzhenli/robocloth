import drjit as dr
import mitsuba as mi
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import os
import re
import yaml
from custom_bsdf.utils.ops import double_sided
import torch.nn.functional as NF

from custom_bsdf.material.anisotropicLatent import AnisotropicLatentTexturedModel,MERLBRDF,MERLInterface,MerlTorch
from custom_bsdf.material.isotropicLatent import LatentTexturedModel
from custom_bsdf.material.svpbr import SvPBRBRDF,AXFBRDF
from custom_bsdf.material.compressed import CompressedModel
from custom_bsdf.utils.cuda_manage import print_cuda_memory_info
from custom_bsdf.material.moeLatent import MoeLatentModel
#from mitsuba.render import Integrator, SurfaceInteraction3f, BSDFContext, Ray3f

# from pytorch_model.bvpnet import SingleBVPNet

def create_anisotropic_model(model_path=None, material_type="AnisotropicLatentTexturedModel"):
    """
    Create or load AnisotropicLatentTexturedModel
    
    Args:
        model_path: Model weight file path, if None create new model
        config_path: Configuration file path, default to AnisotropicLatentTexturedModel.yaml
    
    Returns:
        model: AnisotropicLatentTexturedModel instance
    """
    config_path=f"config/material/{material_type}.yaml"
    print(f"Loading configuration from config file: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Create configuration class with recursive dict conversion
    class Config:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                if key != 'type':  # Skip type field
                    # Recursively convert nested dicts to Config objects
                    if isinstance(value, dict):
                        setattr(self, key, Config(value))
                    else:
                        setattr(self, key, value)
    
    config = Config(config_dict)
    #print(f"Model configuration: latent_dim={config.latent_dim}, texture_resolution={config.texture_resolution}")
    #print(f"hidden_layers={config.hidden_layers}, activation={config.activation}")
    
    if material_type == "AnisotropicLatentTexturedModel":
        model = AnisotropicLatentTexturedModel(config).cuda()
    elif material_type == "LatentTexturedModel":
        model = LatentTexturedModel(config)
    elif material_type == "SvPBRBRDF":
        model = SvPBRBRDF(config)
    elif material_type == "CompressedModel":
        model = CompressedModel(config)
    elif material_type == "MoeLatentModel":
        model = MoeLatentModel(config).cuda()
    elif material_type == "MERLBRDF":
        model = MERLBRDF(config).cuda()
    elif material_type == "AXFBRDF":
        axf_path="/home/haoran/axf/mat0001.axf"
        model = AXFBRDF(config,axf_path).cuda()
    else:
        raise ValueError(f"Unsupported material type: {material_type}")
    
    if model_path=="":
        print("Using PBR model")
        return model.cuda()
    
    # If model path is provided, load pretrained weights
    if model_path and os.path.exists(model_path):
        print(f"Loading pretrained model: {model_path}")
        try:
            # First try safe loading (weights only)
            checkpoint = torch.load(model_path, map_location='cuda', weights_only=True)
            print("Using safe mode to load checkpoint")
        except Exception as e:
            print(f"Safe loading failed (may contain config objects): trying unsafe mode...")
            try:
                # For PyTorch Lightning checkpoints, usually need weights_only=False
                checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
                print("Successfully loaded checkpoint (containing config objects)")
            except Exception as e2:
                print(f"Loading failed: {e2}")
                print("Using randomly initialized model")
                return model
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                # PyTorch Lightning (.ckpt) or standard format
                state_dict = checkpoint['state_dict']
                print("Detected PyTorch Lightning checkpoint format")
            elif 'model_state_dict' in checkpoint:
                # Standard PyTorch format
                state_dict = checkpoint['model_state_dict']
                print("Detected standard PyTorch checkpoint format")
            elif any(key.startswith(('mlp', 'latent_texture')) for key in checkpoint.keys()):
                # Direct state_dict (possibly from weights_only=True)
                state_dict = checkpoint
                print("Detected direct state_dict format")
            else:
                # Try to load directly
                state_dict = checkpoint
                print("Using default loading method")
        else:
            # Non-dictionary format, possibly other types of checkpoints
            print(f"Warning: Unknown checkpoint format: {type(checkpoint)}")
            print("Using randomly initialized model")
            return model
        
        # Handle PyTorch Lightning key names (remove prefix)
        cleaned_state_dict = {}
        print(f"Checkpoint contains {len(state_dict)} keys")
        material_keys = [k for k in state_dict.keys() if k.startswith('material.')]
        if material_keys:
            print(f"Found {len(material_keys)} keys with 'material.' prefix")
        
        for key, value in state_dict.items():
            if key.startswith('material.'):
                # Remove 'material.' prefix
                new_key = key[9:]  # 'material.' is 9 characters
                cleaned_state_dict[new_key] = value
            elif key.startswith('model.'):
                # Remove 'model.' prefix
                new_key = key[6:]  # 'model.' is 6 characters
                cleaned_state_dict[new_key] = value
            else:
                cleaned_state_dict[key] = value
        
        print(f"Cleaned state_dict contains {len(cleaned_state_dict)} keys")
        # Display first few keys for diagnostic
        key_samples = list(cleaned_state_dict.keys())[:5]
        print(f"Example keys: {key_samples}")
        
        try:
            model.load_state_dict(cleaned_state_dict)
            print("Model weights loaded successfully")
        except Exception as e:
            print(f"Weight loading failed: {e}")
            print("Attempting to load with strict=False mode...")
            try:
                model.load_state_dict(cleaned_state_dict, strict=False)
                print("Model weights loaded successfully (non-strict mode)")
            except Exception as e2:
                print(f"Non-strict mode also failed: {e2}")
                print("Using randomly initialized model")
                return model
    else:
        if model_path:
            print(f"Warning: Model file {model_path} does not exist, using randomly initialized model")
        else:
            print("Using randomly initialized AnisotropicLatentTexturedModel")
    
    model.eval()

    return model.cuda()

def ggx_D(cos_theta_h, alpha):
        a2 = alpha * alpha
        denom = cos_theta_h * cos_theta_h * (a2 - 1.0) + 1.0
        return a2 / (dr.pi * denom * denom)


def smith_G1(cos_theta, alpha):
    tan2 = (1.0 - cos_theta * cos_theta) / (cos_theta * cos_theta)
    return 2.0 / (1.0 + dr.sqrt(1.0 + alpha * alpha * tan2))


def smith_G(cos_theta_i, cos_theta_o, alpha):
    return smith_G1(cos_theta_i, alpha) * smith_G1(cos_theta_o, alpha)


def fresnel_schlick(cos_theta, F0):
    ct = dr.clamp(cos_theta, 0.0, 1.0)
    x = 1.0 - ct
    x2 = x * x
    x5 = x2 * x2 * x
    return F0 + (1.0 - F0) * x5

def square_to_ggx(sample, alpha):
    u1 = sample.x
    u2 = sample.y

    a2 = alpha * alpha

    # tan^2(theta)
    tan2_theta = a2 * u1 / dr.maximum(1.0 - u1, 1e-6)
    cos_theta = 1.0 / dr.sqrt(1.0 + tan2_theta)
    sin_theta = dr.sqrt(dr.maximum(0.0, 1.0 - cos_theta * cos_theta))

    phi = 2.0 * dr.pi * u2
    sin_phi, cos_phi = dr.sin(phi), dr.cos(phi)

    h = mi.Vector3f(
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta
    )
    return h
def reflect(wi, h):
    return 2.0 * dr.dot(wi, h) * h - wi

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=3):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        #self._init_weights()
        
    def _init_weights(self):
    # Initialize all weights and biases to fixed values (e.g., 0.1 and 0.0)
        init.constant_(self.fc1.weight, 10)
        init.constant_(self.fc1.bias, 0.0)

        init.constant_(self.fc2.weight, 10)
        init.constant_(self.fc2.bias, 0.0)

        init.constant_(self.fc3.weight, 10)
        init.constant_(self.fc3.bias, 0.0)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MLPBRDF(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)

        self.eta = props.get('eta', 1.33)

        # [BRDFMAX] per-material max-BRDF tracking (diagnostic)
        self.mat_tag = "unknown"
        self._brdf_max = 0.0

        # Per-shape UV tiling multiplier. Default 3.0 reproduces the legacy
        # hardcoded uv*3.0; larger -> pattern repeats more often (smaller motif).
        self.tiling = float(props.get('tiling', '3.0') or 3.0)

        # Debug: mask brdf to zero where the incoming-light direction (BRDF's
        # wi = Mitsuba's wo) makes an angle > grazing_mask_deg with the local
        # +z (geometric normal). Default 0 = disabled.
        grazing_deg = float(props.get('grazing_mask_deg', '0') or 0)
        self.grazing_mask_cos = math.cos(math.radians(grazing_deg)) if grazing_deg > 0 else 0.0

        # Check if using AnisotropicLatentTexturedModel
        if props.has_property('model_path'):
            # Load AnisotropicLatentTexturedModel from path
            model_path = props.get('model_path', '')  # Provide default value
            self.model_path = model_path
            _m = re.search(r'Material-(\d+)', model_path or '')
            self.mat_tag = _m.group(1) if _m else os.path.basename(os.path.dirname(model_path or 'unknown'))
            print(f"MLPBRDF: Loading AnisotropicLatentTexturedModel from {model_path}")
            # Get config path from props, use default if not provided
            material_type = props.get('material_type', '')
            self.anisotropic_model = create_anisotropic_model(model_path, material_type)
            #self.anisotropic_model=MerlTorch(merl_files="/home/featurize/data/demo")
            #self.anisotropic_model=HyperBRDF("/home/featurize/data/results/merl/MERL/pt_results/gray-plastic.pt","/home/featurize/work/HyperBRDF/data/merl_median.binary")
            torch.cuda.empty_cache()
            self.use_anisotropic = True
            print("MLPBRDF: AnisotropicLatentTexturedModel loaded")
        elif props.has_property('use_anisotropic') and bool(props.get('use_anisotropic', False)):
            # Use randomly initialized AnisotropicLatentTexturedModel
            print("MLPBRDF: Creating randomly initialized AnisotropicLatentTexturedModel")
            # Get config path from props, use default if not provided
            material_type = props.get('material_type', '')
            self.anisotropic_model = create_anisotropic_model(None, material_type)
            self.use_anisotropic = True
            print("MLPBRDF: AnisotropicLatentTexturedModel created")
        else:
            # Use simple MLP and traditional weights
            self.use_anisotropic = False
            if props.has_property('w1'):
                # Use passed weights
                self.w1 = props['w1']
                self.b1 = props['b1']
                self.w2 = props['w2']
                self.b2 = props['b2']
                self.w3 = props['w3']
                self.b3 = props['b3']
            
            self.model = SimpleMLP().cuda()
            print("MLPBRDF: Using SimpleMLP")
        
        self.m_flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide

        #self.set_point_lights(props)

    def __del__(self):
        """
        Destructor: Explicitly release CUDA memory
        """
        try:
            if hasattr(self, 'use_anisotropic') and self.use_anisotropic:
                if hasattr(self, 'anisotropic_model'):
                    print("MLPBRDF: Releasing AnisotropicLatentTexturedModel CUDA memory")
                    # Delete model reference
                    del self.anisotropic_model
            
            if hasattr(self, 'model'):
                # Release SimpleMLP model
                del self.model
            
            # Force clean up PyTorch CUDA cache
            torch.cuda.empty_cache()
            
            # Clean up DrJit cache
            dr.flush_malloc_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            print("MLPBRDF: CUDA memory release completed")
            
        except Exception as e:
            # Exceptions in destructor should not be thrown
            print(f"MLPBRDF destructor warning: {e}")

    def set_point_lights(self,props):
        self.point_lights = []
        if props.has_property('emitter1'):
            print("Found emitter1")
            emitter1 = props['emitter1']
            # Mitsuba has already converted the config to an Emitter object, we need to use mi.traverse to get parameters
            try:
                emitter1_params = mi.traverse(emitter1)
                position = emitter1_params.get('position', [1.0, 2.0, 3.0])
                # For point lights, intensity is usually stored in 'intensity.value'
                intensity_value = emitter1_params.get('intensity.value', 20.0)
                if hasattr(intensity_value, '__iter__') and len(intensity_value) == 3:
                    # If it's an RGB value, take the average or the first component
                    intensity_value = float(intensity_value[0])
                self.point_lights.append((position, intensity_value))
                print(f"emitter1: position={position}, intensity={intensity_value}")
            except Exception as e:
                print(f"Failed to parse emitter1: {e}")
                # Use default value
                self.point_lights.append(([1.0, 2.0, 3.0], 5.0))
        
        if props.has_property('emitter2'):
            print("Found emitter2")
            emitter2 = props['emitter2']
            try:
                emitter2_params = mi.traverse(emitter2)
                position = emitter2_params.get('position', [-2.0, 1.0, 0.0])
                intensity_value = emitter2_params.get('intensity.value', 15.0)
                if hasattr(intensity_value, '__iter__') and len(intensity_value) == 3:
                    intensity_value = float(intensity_value[0])
                self.point_lights.append((position, intensity_value))
                print(f"emitter2: position={position}, intensity={intensity_value}")
            except Exception as e:
                print(f"Failed to parse emitter2: {e}")
                # Use default value
                self.point_lights.append(([-2.0, 1.0, 0.0], 5.0))
        
        # If no lights are found, use default values
        if not self.point_lights:
            print("No light configuration found, using default values")
            self.point_lights = [
                ([1.0, 2.0, 3.0], 20.0),
                ([-2.0, 1.0, 0.0], 15.0)
            ]
        
        print(f"Set {len(self.point_lights)} point lights")

    def flip(self,si,mask):
        si.wi[mask] = -si.wi[mask]
        si.sh_frame.n[mask] = -si.sh_frame.n[mask]
        si.sh_frame.s[mask] = -si.sh_frame.s[mask]  # flip tangent
        si.sh_frame.t[mask] = -si.sh_frame.t[mask]  # flip bitangent
    '''
    def sample(self, ctx, si, sample1, sample2, active):
        print("sample2",sample2)
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        active &= cos_theta_i > 0
        #active&=si.wi.z>0
        self.flip(si,~active)

        bs = mi.BSDFSample3f()
        bs.wo  = mi.warp.square_to_cosine_hemisphere(sample2)
        #bs.wo=si.wi
        bs.pdf = mi.warp.square_to_cosine_hemisphere_pdf(bs.wo)
        #bs.pdf=0.1
        bs.eta = 1.0
        bs.sampled_type = +mi.BSDFFlags.GlossyReflection
        bs.sampled_component = 0

        brdf_res=self.eval(ctx, si, bs.wo, active)
        #print("brdf_res:", brdf_res)
        value = mi.Vector3f(0.001) / bs.pdf
        
        # Only clear cache after large batch sampling
        sample_count = dr.width(si.wi.x) if hasattr(dr, 'width') else 1
        if sample_count > 1500:  # Medium threshold, as sample might have medium frequency
            torch.cuda.empty_cache()
        return ( bs, dr.select(active & (bs.pdf > 0.0), value, mi.Vector3f(0)) )
        #return ( bs, value )   
    '''
    
    '''
    def sample(self, ctx, si, sample1, sample2, active):
        
        roughness=0.2
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        active &= cos_theta_i > 0.0

        # Roughness -> alpha
        alpha = dr.maximum(roughness, 1e-4)

        # 1. 采样 GGX 半程向量 h
        h = square_to_ggx(sample2, alpha)
        cos_theta_h = mi.Frame3f.cos_theta(h)

        # 2. 反射方向
        wo = reflect(si.wi, h)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        active &= cos_theta_o > 0.0

        # 3. GGX 分布
        D = ggx_D(cos_theta_h, alpha)

        # 4. Smith 几何项
        G = smith_G(cos_theta_i, cos_theta_o, alpha)

        # 5. Fresnel（假设 dielectric，F0 = 0.04）
        F0 = mi.Vector3f(0.04)
        F = fresnel_schlick(dr.dot(si.wi, h), F0)

        # 6. Cook–Torrance BRDF
        brdf = self.eval(ctx, si, wo, active)

        # 7. PDF（从 h 转换到 wo）
        pdf_h = D * cos_theta_h
        pdf = pdf_h / (4.0 * dr.dot(si.wi, h))

        # 8. BSDFSample
        bs = mi.BSDFSample3f()
        bs.wo = wo
        bs.pdf = pdf
        bs.eta = 1.0
        bs.sampled_type = +mi.BSDFFlags.GlossyReflection
        bs.sampled_component = 0

        value = dr.select(active & (pdf > 0.0),
                          brdf * cos_theta_o / pdf,
                          mi.Vector3f(0.0))

        return bs, value
    def pdf(self, ctx, si, wo, active=True):
        roughness=0.2
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        active &= (cos_theta_i > 0.0) & (cos_theta_o > 0.0)

        # 半程向量
        h = dr.normalize(si.wi + wo)

        # 防止背面 / 数值问题
        dot_wi_h = dr.dot(si.wi, h)
        active &= dot_wi_h > 0.0

        # GGX 参数（和 sample 中一致！）
        alpha = dr.maximum(roughness * roughness, 1e-4)

        # D(h)
        cos_theta_h = mi.Frame3f.cos_theta(h)
        a2 = alpha * alpha
        denom = cos_theta_h * cos_theta_h * (a2 - 1.0) + 1.0
        D = a2 / (dr.pi * denom * denom)

        # p(wo) = D(h) * cos(theta_h) / (4 * dot(wi, h))
        pdf = D * cos_theta_h / (4.0 * dot_wi_h)

        return dr.select(active, pdf, 0.0)
    '''
    def sample(self, ctx, si, sample1, sample2, active):
        # 1. Determine the number of rays/samples in the current batch
        batch_size = dr.width(sample2)

        # 2. Generate samples in PyTorch (e.g., Uniform 0 to 1)
        # Ensure the device matches your Mitsuba variant (usually 'cuda')
        torch_samples = torch.rand((batch_size, 2), device='cuda', dtype=torch.float32)

        # 3. Transform PyTorch tensor back to Mitsuba type (mi.Point2f)
        # We split the tensor columns and wrap them into Dr.Jit arrays

        sample2_torch = mi.Vector2f(
            dr.cuda.Float(torch_samples[:, 0]),
            dr.cuda.Float(torch_samples[:, 1])
        )

        #sample2_torch=mi.Vector2f(mi.Float(torch_samples[:, 0].cpu().contiguous()), mi.Float(torch_samples[:, 1].cpu().contiguous()))
        # 4. Proceed with standard Mitsuba warping
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        active &= cos_theta_i > 0
        #self.flip(si, ~active)

        bs = mi.BSDFSample3f()
        # Now using the PyTorch-generated samples
        bs.wo = mi.warp.square_to_cosine_hemisphere(sample2_torch)
        #bs.wo=si.wi
        bs.pdf = mi.warp.square_to_cosine_hemisphere_pdf(bs.wo)
        #bs.pdf=0.1
        bs.eta = 1.0
        bs.sampled_type = +mi.BSDFFlags.GlossyReflection
        bs.sampled_component = 0

        brdf_res=self.eval(ctx, si, bs.wo, active)
        #print("brdf_res:", brdf_res)
        value = brdf_res / bs.pdf
        
        # Only clear cache after large batch sampling
        sample_count = dr.width(si.wi.x) if hasattr(dr, 'width') else 1
        if sample_count > 1500:  # Medium threshold, as sample might have medium frequency
            torch.cuda.empty_cache()
        return ( bs, dr.select(active & (bs.pdf > 0.0), value, mi.Vector3f(0)) )
        #return ( bs, value )   
        
    def pdf(self, ctx, si, wo, active=True):
        
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        pdf = dr.clamp(mi.Frame3f.cos_theta(wo),1e-5,1)*1/dr.pi
        # pdf = mi.warp.square_to_beckmann_pdf(wo,0.08)

        return dr.select((cos_theta_i > 0.0) & (cos_theta_o > 0.0), pdf, 0.0)
    
    
    def eval_pdf(self, ctx, si, wo, active=True):
        return self.eval(ctx, si, wo, active), self.pdf(ctx, si, wo, active)
    '''
    def sample_point(self, ctx, si, sample1, sample2, active):
        # Filter invalid directions
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        active &= cos_theta_i > 0

        self.flip(si,~active)

        bs = mi.BSDFSample3f()

        # Number of point lights
        num_lights = len(self.point_lights)

        # Sample one light uniformly
        light_index = 0
        light_pos, light_intensity = self.point_lights[light_index]

        

        # Set BSDFSample3f fields
        bs.eta = 1.0
        bs.sampled_type = +mi.BSDFFlags.DeltaReflection  # treat as delta reflection
        bs.sampled_component = 0

        # Assume uniform light selection => PDF = 1 / num_lights
        # Compute direction to light
        direction_to_light = light_pos - si.p
        direction_to_light=-direction_to_light
        dist_squared = dr.squared_norm(direction_to_light)
        bs.wo = dr.normalize(direction_to_light)
        bs.pdf = 10000

        # Evaluate BRDF
        brdf_val = self.eval(ctx, si, bs.wo, active)

        # Calculate contribution (note: cosine not included because it's a point light)
        value = brdf_val / bs.pdf

        value = mi.Color3f(1)
        return ( bs, value )   
        #return bs, dr.select(active & (bs.pdf > 0), value, mi.Color3f(0.0))
    '''
    
    def eval(self, ctx, si, wo, active):
        print("eval called")  # 🎯 Add debug information
        if self.use_anisotropic:
            # Get sample count to decide whether to use batching
            sample_count = dr.width(si.wi.x)
            
            # For large batches, automatically use batch version to save memory
            if False:
            #if sample_count > 250000:  # Use batching for more than 16K samples
                print("batch")
                value = self.eval_anisotropic_mlp_batched(si, wo, max_batch_size=min(sample_count, 500000))
            else:
                print("no batch")
                value = self.eval_anisotropic_mlp(si, wo)
        else:
            value = self.mlp(si, wo)
        
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        # Only clear cache after large batch evaluation
        sample_count = dr.width(si.wi.x) if hasattr(dr, 'width') else 1
        if sample_count > 2000:  # Higher threshold, as eval might be called frequently
            torch.cuda.empty_cache()
        #return value
        #return mi.Vector3f(0.1)
        # [BRDFMAX] track the max raw BRDF value emitted by THIS material so we
        # can compare materials (e.g. is 190 >> others?). value is the brdf result.
        if os.environ.get('BRDFMAX_LOG'):
            try:
                _vmax = float(value.torch().max().item())
                if _vmax > self._brdf_max:
                    self._brdf_max = _vmax
                print(f"[BRDFMAX] mat={self.mat_tag} call_max={_vmax:.4f} running_max={self._brdf_max:.4f}")
            except Exception as _e:
                print(f"[BRDFMAX] mat={self.mat_tag} err={_e}")
        # Subsume the back-face mask: grazing_mask_cos >= 0 ; when set above 0
        # it also masks the trained-decoder's grazing-angle predictions.
        thresh = max(self.grazing_mask_cos, 0.0)
        return dr.select((cos_theta_i > 0) & (cos_theta_o > thresh), value, mi.Vector3f(0.0))
        #return dr.select(active & (cos_theta_i > 0) & (cos_theta_o > 0), value, mi.Vector3f(0))
    
    def uv_to_grid_id(self,uv, resolution=512):
        # 防止越界
        uv = torch.clamp(uv, 0.0, 1.0 - 1e-8)

        # 映射到像素坐标
        m = (uv[:, 0] * resolution).long()  # u -> 列
        n = (uv[:, 1] * resolution).long()  # v -> 行

        # 计算 id
        grid_id = n * resolution + m

        return grid_id
    
    def uv_to_bilinear_ids(self, uv, resolution=512):
        """
        输入:
            uv: [B,2] in [0,1]
        输出:
            ids: [B,4]   四个格子 id
            weights: [B,4] 双线性权重
        """

        # 安全处理
        uv = torch.nan_to_num(uv, nan=0.0)
        uv = torch.clamp(uv, 0.0, 1.0 - 1e-6)

        # 映射到连续像素坐标
        x = uv[:, 0] * (resolution - 1)
        y = uv[:, 1] * (resolution - 1)

        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        x1 = torch.clamp(x0 + 1, max=resolution - 1)
        y1 = torch.clamp(y0 + 1, max=resolution - 1)

        # 计算权重
        wx = x - x0.float()
        wy = y - y0.float()

        w00 = (1 - wx) * (1 - wy)
        w10 = wx * (1 - wy)
        w01 = (1 - wx) * wy
        w11 = wx * wy

        # 计算4个id
        id00 = y0 * resolution + x0
        id10 = y0 * resolution + x1
        id01 = y1 * resolution + x0
        id11 = y1 * resolution + x1

        ids = torch.stack([id00, id10, id01, id11], dim=-1)
        weights = torch.stack([w00, w10, w01, w11], dim=-1)

        return ids, weights

    def eval_anisotropic_mlp(self, si, wo):
        print_cuda_memory_info("eval_anisotropic_mlp")
        """Evaluate BRDF using AnisotropicLatentTexturedModel"""
        # Get sample count
        sample_count = dr.width(si.wi.x)
        print("sample_count", sample_count)
        
        # Immediately extract all required data and convert to independent PyTorch tensors
        # This allows releasing references to DrJit arrays

        # Mitsuba 3 returns Vector*.torch() in channels-first layout (K, N);
        # the model expects batch-first (N, K), so we transpose at the boundary.
        def _bf(t):
            return t.T.contiguous() if t.dim() == 2 and t.shape[0] in (2, 3) and t.shape[0] != t.shape[1] else t

        # 1. World coordinate position - create independent copy
        pos_torch = _bf(si.p.torch().detach())

        # 2. Geometric normal (world coordinate) - create independent copy
        normal_torch = NF.normalize(_bf(si.sh_frame.n.torch().detach()), dim=-1)
        tangent_torch = NF.normalize(_bf(si.sh_frame.s.torch().detach()), dim=-1)
        bitangent_torch = NF.normalize(_bf(si.sh_frame.t.torch().detach()), dim=-1)
        TBN_torch = torch.stack([tangent_torch, bitangent_torch, normal_torch], dim=-1)  # [B, 3, 3]

        # 3. UV texture coordinates - create independent copy
        uv_torch = _bf(si.uv.torch().detach())
        uv_torch=(uv_torch*self.tiling)%1.0
        uv_torch=(uv_torch*0.8+0.1)%1.0
        #material_ids=torch.full((wo_torch.shape[0],), 0, dtype=torch.int, device="cuda")
        material_ids,weights=self.uv_to_bilinear_ids(uv_torch)
        #uv_torch=(uv_torch*4.0)%1.0
        # 4. Incident direction (world coordinate) - create independent copy
        wi_torch = _bf(si.wi.torch().detach())

        # 5. Exit direction (world coordinate) - create independent copy
        wo_torch = _bf(wo.torch().detach())
        
        # Now we have independent tensor copies, we can release some memory in eval_brdf
        # Force Python garbage collection, clean up any temporary DrJit array references
        #normal_torch = double_sided(wo_torch,normal_torch)
        del si
        
        # Evaluate BRDF using AnisotropicLatentTexturedModel
        #flip_mask=wi_torch[:,2]<0
        #print("flip_mask",torch.sum(flip_mask))
        #wi_torch[flip_mask]=-wi_torch[flip_mask]
        with torch.no_grad():
            brdf, _,_,_ = self.anisotropic_model.eval_brdf({}, pos_torch.cuda(), wo_torch.cuda(), wi_torch.cuda(), normal_torch.cuda(),uv_torch.cuda(), TBN_torch.cuda())
            #brdf,_ = self.anisotropic_model.eval_brdf(wo_torch.cuda(), wi_torch.cuda(),uv_torch.cuda())
        
        #brdf=brdf.detach().cpu()

        #print("brdf",brdf)
        '''
        max_batch_size=1000000
        with torch.no_grad():
            brdf_list=[]
            for start_idx in range(0, sample_count, max_batch_size):
                end_idx = min(start_idx + max_batch_size, sample_count)
                batch_size = end_idx - start_idx
                brdf_batch, _ = self.anisotropic_model.eval_brdf({}, pos_torch[start_idx:end_idx], wi_torch[start_idx:end_idx], wo_torch[start_idx:end_idx], normal_torch[start_idx:end_idx], uv_torch[start_idx:end_idx])
                brdf_list.append(brdf_batch)
            # brdf shape: (N, 3)
            brdf=torch.cat(brdf_list, dim=0)
        # Immediately delete input tensors to release memory
        '''
        del pos_torch, normal_torch, uv_torch, wi_torch, wo_torch, TBN_torch
        
        # Convert directly from PyTorch to DrJit (stay on CUDA)
        '''
        result=mi.Vector3f(
            mi.Float(brdf[:, 0].contiguous()),
            mi.Float(brdf[:, 1].contiguous()),
            mi.Float(brdf[:, 2].contiguous())
        )
        '''
        result = mi.Vector3f(
            dr.cuda.Float(brdf[:, 0].contiguous()),
            dr.cuda.Float(brdf[:, 1].contiguous()), 
            dr.cuda.Float(brdf[:, 2].contiguous())
        )
        
        # Delete BRDF tensor reference
        
        del brdf
        torch.cuda.empty_cache()                
        # Clean up DrJit cache
        dr.flush_malloc_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        return result

    def eval_anisotropic_mlp_batched(self, si, wo, max_batch_size=8192):
        """Evaluate BRDF using batching to save memory"""
        sample_count = dr.width(si.wi.x)
        
        # If sample count is less than batch size, use regular method
        if sample_count <= max_batch_size:
            return self.eval_anisotropic_mlp(si, wo)
        
        # Large batches use batch processing
        print(f"Using batch mode: {sample_count} samples, batch size {max_batch_size}")
        
        # Pre-allocate result array
        results = []
        
        for start_idx in range(0, sample_count, max_batch_size):
            end_idx = min(start_idx + max_batch_size, sample_count)
            batch_size = end_idx - start_idx
            
            # Extract data for the current batch
            pos_batch = si.p.torch().detach()[start_idx:end_idx]
        
            # 2. Geometric normal (world coordinate) - create independent copy
            normal_batch = si.sh_frame.n.torch().detach()[start_idx:end_idx]
            
            # 3. UV texture coordinates - create independent copy
            uv_batch = si.uv.torch().detach()[start_idx:end_idx]
            
            # 4. Incident direction (world coordinate) - create independent copy
            wi_batch = si.wi.torch().detach()[start_idx:end_idx]
            
            # 5. Exit direction (world coordinate) - create independent copy
            wo_batch = wo.torch().detach()[start_idx:end_idx]
            
            # Evaluate current batch
            with torch.no_grad():
                brdf_batch, _ = self.anisotropic_model.eval_brdf({}, pos_batch, wi_batch, wo_batch, normal_batch, uv_batch)
            
            # Immediately delete input tensors
            del pos_batch, normal_batch, uv_batch, wi_batch, wo_batch
            
            # Convert to DrJit and save results
            batch_result = mi.Vector3f(
                dr.cuda.Float(brdf_batch[:, 0]),
                dr.cuda.Float(brdf_batch[:, 1]),
                dr.cuda.Float(brdf_batch[:, 2])
            )
            results.append(batch_result)
            
            # Delete BRDF tensor and force cleanup
            del brdf_batch
            torch.cuda.empty_cache()
            
            print(f"Batch {start_idx//max_batch_size + 1}/{(sample_count-1)//max_batch_size + 1} completed")
        
        # Merge all batch results - use PyTorch for merging then convert back to DrJit
        all_results_r = []
        all_results_g = []
        all_results_b = []
        
        for batch_result in results:
            # Convert DrJit arrays back to PyTorch for merging
            all_results_r.append(batch_result.x.torch())
            all_results_g.append(batch_result.y.torch())
            all_results_b.append(batch_result.z.torch())
        
        # Merge all batches in PyTorch
        combined_r = torch.cat(all_results_r, dim=0)
        combined_g = torch.cat(all_results_g, dim=0)
        combined_b = torch.cat(all_results_b, dim=0)
        
        # Clean up temporary results
        del results, all_results_r, all_results_g, all_results_b
        
        # Convert back to DrJit
        final_result = mi.Vector3f(
            dr.cuda.Float(combined_r),
            dr.cuda.Float(combined_g),
            dr.cuda.Float(combined_b)
        )
        
        # Clean up merged tensors
        del combined_r, combined_g, combined_b
        torch.cuda.empty_cache()
                
        # Clean up DrJit cache
        dr.flush_malloc_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        return final_result

    def relu(self, x):
        return torch.max(0, x)
    
    def mlp_layer_con(self, x, w, b):
        # Implement one layer of MLP using basic operations
        # For each output neuron, compute weighted sum of inputs
        result = dr.zeros(mi.TensorXf, shape=len(b))
        for i in range(len(b)):
            # Compute dot product for this neuron
            neuron_result = b[i]
            # Get weights for this neuron (row i of weight matrix)
            w_slice = w[i]
            # Compute dot product of weights and input
            
            neuron_result = neuron_result + dr.dot(w_slice, x)
            result[i] = neuron_result
        return result
        
    def mlp_layer(self, x, w, b):
        # x: input vector of shape (in_dim,)
        # w: weight matrix of shape (out_dim, in_dim)
        # b: bias vector of shape (out_dim,)
        #print("w.shape:", dr.shape(w))
        #print("x.shape:", dr.shape(x))
        return w@x + b

    def mlp(self, si, wo):
        # Prepare input tensor: manually construct tensor from wi and wo components
        x = dr.zeros(mi.TensorXf, 6)
        x[0] = si.wi.x
        x[1] = si.wi.y 
        x[2] = si.wi.z
        x[3] = wo.x
        x[4] = wo.y
        x[5] = wo.z
        x=x.torch().cuda()
        out=self.model(x)
        return mi.Vector3f(float(out[0]), float(out[1]), float(out[2]))

        # Layer 1: MLP layer + ReLU
        h1 = self.mlp_layer(x, self.w1.torch(), self.b1.torch())
        h1 = self.relu(h1)
        
        # Layer 2: MLP layer + ReLU
        h2 = self.mlp_layer(h1, self.w2, self.b2) 
        h2 = self.relu(h2)
        
        # Layer 3: MLP layer + Sigmoid
        out = self.mlp_layer(h2, self.w3, self.b3)
        out = dr.sigmoid(out)
        
        # Convert output tensor to Vector3f
        return mi.Vector3f(out[0], out[1], out[2])

