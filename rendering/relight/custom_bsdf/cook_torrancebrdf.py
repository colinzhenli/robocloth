import drjit as dr
import mitsuba as mi
import torch
import torch.nn as nn
# from pytorch_model.bvpnet import SingleBVPNet

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.eta = nn.Parameter(torch.randn(1))
        self.mlp = nn.Linear(1, 1)

    def forward(self, x):
        ret = self.mlp(x.unsqueeze(0))
        print(ret.requires_grad) 
        return ret

@dr.wrap_ad(source='drjit', target='torch')
def pass_mlp(eta):
    model = Model().cuda()
    ret = model(eta)
    return ret


class CookTorranceBRDF(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)

        self.roughness = props.get('roughness', 0.5)
        self.roughness = mi.Float(self.roughness)

        # Fresnel IOR (eta), SH coefficients, and tint from the original code
        self.eta = props.get('eta', 1.33)
        self.eta = mi.Float(self.eta)
        
        self.m_flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide

    def sample(self, ctx, si, sample1, sample2, active):
        # mapped_eta = pass_mlp(dr.cuda.ad.TensorXf(self.eta))
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        active &= cos_theta_i > 0

        bs = mi.BSDFSample3f()
        bs.wo  = mi.warp.square_to_cosine_hemisphere(sample2)
        bs.pdf = mi.warp.square_to_cosine_hemisphere_pdf(bs.wo)
        bs.eta = self.eta
        bs.sampled_type = +mi.BSDFFlags.GlossyReflection
        bs.sampled_component = 0

        value = self.eval(ctx, si, bs.wo, active) / bs.pdf
        return ( bs, dr.select(active & (bs.pdf > 0.0), value, mi.Vector3f(0)) )


    def eval(self, ctx, si, wo, active):
        # Retrieve incident and outgoing directions

        # value = self.phong(wi, wo, N) * mi.Frame3f.cos_theta(wo)
        value = self.cook_torrance(si, wo) * mi.Frame3f.cos_theta(wo)

        # Only return value if cos_theta_i > 0, cos_theta_o > 0, and active
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        return dr.select(active & (cos_theta_i > 0) & (cos_theta_o > 0), value, mi.Vector3f(0))

    
    def cook_torrance(self, si, wo):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        F = self.fresnel(cos_theta_i)

        alpha = self.roughness ** 2
        D = self.ggx_distribution(si, wo, alpha)
        G = self.geometric_attenuation(si, wo)
        
        denom = 4.0 * cos_theta_i * cos_theta_o
        value = F*G*D / denom
        value = dr.clamp(value, 0, 1)
        return value

    def phong(self, si, wo):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        # Constants for Phong shading
        k_d = 0.5  
        k_s = 0.8  
        shininess = 200.0 

        # Diffuse Term (Lambert's Law)
        cos_theta_i = dr.maximum(0.0, cos_theta_i)
        diffuse = k_d * cos_theta_i

        # Specular Term
        R = mi.reflect(si.wi)
        cos_alpha = dr.maximum(0.0, dr.dot(R, wo))  # Cosine of angle between reflected vector and outgoing direction
        specular = k_s * (cos_alpha ** shininess)

        # Combine the diffuse and specular components
        value = diffuse + specular

        return value

    def ggx_distribution(self, si, wo, alpha):
        h = si.wi + wo
        h = h / (dr.sqrt(dr.dot(h, h)))
        cos_theta_h = mi.Frame3f.cos_theta(h)
        denom = cos_theta_h ** 2 * (alpha ** 2 - 1.0) + 1.0
        D = dr.select(cos_theta_h > 0, alpha ** 2 / (dr.pi * denom ** 2), 0)
        return D

    def smith_geometry(self, alpha, cos_theta_i):
        lambda_i = (-1 + (1 + alpha ** 2 * (1 - cos_theta_i ** 2) / (cos_theta_i ** 2)) ** 0.5) / 2
        G = 1 / (1 + lambda_i)
        return G
    
    def geometric_attenuation(self, si, wo):
        """ h is the microfacet normal, v is the view direction, l is the incident direction, n is the surface normal """
        h = si.wi + wo
        h = h / (dr.sqrt(dr.dot(h, h))) # local frame
        hn = mi.Frame3f.cos_theta(h)
        nv = mi.Frame3f.cos_theta(wo)
        nl = mi.Frame3f.cos_theta(si.wi)
        vh = dr.dot(wo, h)
        G = dr.minimum(2 * hn * nv / vh, 2 * hn * nl / vh)
        G = dr.minimum(1, G)
        return G

    def fresnel(self, cos_theta_i):
        """Compute Fresnel term using Schlick's approximation and normal incidence reflectance."""
        F0 = ((self.eta - 1) / (self.eta + 1)) ** 2
        F0 = self.eta

        return F0 + (1.0 - F0) * (1.0 - cos_theta_i) ** 5
    
    def pdf(self, ctx, si, wo, active):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo)

        return dr.select(active & (cos_theta_i > 0.0) & (cos_theta_o > 0.0), pdf, 0.0)


    def traverse(self, callback):
        # Register learnable parameters for optimization
        callback.put_parameter('roughness', self.roughness, mi.ParamFlags.Differentiable)
        callback.put_parameter('eta', self.eta, mi.ParamFlags.Differentiable)
