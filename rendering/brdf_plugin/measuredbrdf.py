import mitsuba as mi
import drjit as dr
import numpy as np


class MeasuredBRDF(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        self.filename = props['filename']
        # TODO: implement BTF loading and evaluation
        
        self.m_flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide

    def sample(self, ctx, si, sample1, sample2, active):
        pass

    def eval(self, ctx, si, wo, active):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        # Sample cosine hemisphere
        bs = mi.BSDFSample3f()
        bs.wo = mi.warp.square_to_cosine_hemisphere(sample2)
        bs.pdf = mi.warp.square_to_cosine_hemisphere_pdf(bs.wo)
        bs.eta = 1.0
        bs.sampled_type = +mi.BSDFFlags.GlossyReflection
        bs.sampled_component = 0

        # Read loaded BTF
        value = get_btf_raw_value()

        return value

    def get_btf_raw_value(self):
        """
        Get raw BTF values
        """
        # Get BTF pixel value
        value = dr.sqrt(wo[0] + wo[1] + wo[2])
        return mi.Color3f(value)

    def pdf(self, ctx, si, wo, active):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo)

        return dr.select(active & (cos_theta_i > 0.0) & (cos_theta_o > 0.0), pdf, 0.0)

