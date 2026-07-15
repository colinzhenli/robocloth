import drjit as dr
import mitsuba as mi
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import os
import yaml
from custom_bsdf.utils.ops import double_sided

from custom_bsdf.material.moeLatent import MoeLatentModel
from custom_bsdf.material.anisotropicLatent import AnisotropicLatentTexturedModel
from custom_bsdf.material.isotropicLatent import LatentTexturedModel
from custom_bsdf.material.svpbr import SvPBRBRDF, AXFBRDF
from custom_bsdf.material.compressed import CompressedModel
from custom_bsdf.utils.cuda_manage import print_cuda_memory_info

def create_model(model_path=None, material_type="AnisotropicLatentTexturedModel"):
    """
    Create or load AnisotropicLatentTexturedModel
    
    Args:
        model_path: Model weight file path, if None create new model
        config_path: Configuration file path, default to AnisotropicLatentTexturedModel.yaml
    
    Returns:
        model: AnisotropicLatentTexturedModel instance
    """
    print("model_path",model_path)
    config_path=f"../config/material/{material_type}.yaml"
    print(f"Loading configuration from config file: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Create configuration class
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
    elif material_type == "AXFBRDF":
        model = AXFBRDF(config)
    elif material_type == "CompressedModel":
        model = CompressedModel(config)
    elif material_type == "MoeLatentModel":
        model = MoeLatentModel(config).cuda()
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