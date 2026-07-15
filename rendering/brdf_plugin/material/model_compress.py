import torch
import torch.nn as nn
import torch.nn.functional as NF
import math
from pytorch_lightning import LightningModule
import sys
from brdf_plugin.utils.ops import *
from brdf_plugin.utils.cuda_manage import print_cuda_memory_info
import torch.nn.functional as F


class Model_compress(nn.Module):
    def __init__(self,input_dim):
        super(Model_compress, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 3)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x=F.relu(x)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        x=self.activation(x)
        return x

    def eval_brdf(self, pos, wi, wo, normal,tangent,uv,batch_mask=None):
        print("using anisotropic latent model")
        
        # Ensure normal is normalized
        NoL=wi[:,2:3].repeat(1, 3)
        NoV=wo[:,2:3].repeat(1, 3)
        #NoL = (wi*normal).sum(-1,keepdim=True)
        #NoV = (wo*normal).sum(-1,keepdim=True)

        
        if self.training and self.Gaussian_blur:
            tex = self._blur_latent(self.global_step)
        else:
            tex = self.latent_texture       
        latent = self.sample_latent_from_texture(pos, tex,uv)
        
            
        if self.predict_frame:
            # Extract predicted normal and tangent from latent
            predicted_normal = latent[..., -6:-3]  # Last 6-3 dimensions for normal
            predicted_tangent = latent[..., -3:]   # Last 3 dimensions for tangent
            
            # Normalize predicted vectors
            predicted_normal = torch.nn.functional.normalize(predicted_normal, dim=-1)
            predicted_tangent = torch.nn.functional.normalize(predicted_tangent, dim=-1)
            #print("predicted_tangent", predicted_tangent)
            #print("predicted_normal", predicted_normal)
            # Use Gram-Schmidt orthogonalization to make tangent perpendicular to normal
            # Keep normal unchanged and orthogonalize tangent
            predicted_tangent = predicted_tangent - torch.sum(predicted_tangent * predicted_normal, dim=-1, keepdim=True) * predicted_normal
            predicted_tangent = torch.nn.functional.normalize(predicted_tangent, dim=-1)

            #predicted_normal = vector_transform(predicted_normal)
            #predicted_tangent = vector_transform(predicted_tangent)#modify2:The original normal space is wrong

            #predicted_normal = local2world(predicted_normal,tangent,normal)
            #predicted_normal=-predicted_normal
            #predicted_tangent = local2world(predicted_tangent,tangent,normal)

        wi = wi[:, [0, 2, 1]]
        wi[:,2]=-wi[:,2]
        #wi[:,1]=-wi[:,1]

        wo = wo[:, [0, 2, 1]]
        wo[:,2]=-wo[:,2]

        wi_local = self.world_to_local(wi, predicted_normal, predicted_tangent)
        wo_local = self.world_to_local(wo, predicted_normal, predicted_tangent)
        local_normal = torch.zeros_like(wi_local)
        local_normal[..., 2] = 1.0  # Normal is always (0,0,1) in local space

        # Split latent into three parts for RGB channels
        if self.colorful_texture:
            if self.larger_latent_dim:
                latent_r = latent[..., :self.latent_dim]
                latent_g = latent[..., self.latent_dim:2*self.latent_dim]
                latent_b = latent[..., 2*self.latent_dim:3*self.latent_dim]
            
                # Get BRDF value for each channel
                brdf_r = self.forward(pos, wi_local, wo_local, local_normal, latent_r, batch_mask, 'r')
                brdf_g = self.forward(pos, wi_local, wo_local, local_normal, latent_g, batch_mask, 'g')
                brdf_b = self.forward(pos, wi_local, wo_local, local_normal, latent_b, batch_mask, 'b')
                # Combine channels
                brdf = torch.cat([brdf_r, brdf_g, brdf_b], dim=-1)
                
            else:
                brdf = self.forward(pos, wi_local, wo_local, local_normal, latent, batch_mask)
        else:
            brdf = self.forward(pos, wi_local, wo_local, local_normal, latent, batch_mask)
            brdf = brdf.repeat(1,3)
        # # brdf = brdf * color
        pdf = NoL / math.pi

        return brdf, pdf