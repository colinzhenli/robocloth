import mitsuba as mi
import drjit as dr
import numpy as np
from .utils.brdf_interpolator import BrdfInterpolator
from .utils.coord_system_transfer import orthogonal2spherical, mirror_uv

class MeasuredBTF(mi.BSDF):
    def __init__(self, props):
        super().__init__(props)
        
        # BTF zip file path
        self.m_filename = props["filename"]

        # Power parameter
        if props.has_property("power_parameter"):
            self.m_power_parameter = mi.Float(props["power_parameter"])
        else:
            self.m_power_parameter = mi.Float(4.0)
        # Wrap mode
        if props.has_property("wrap_mode"):
            self.m_wrap_mode = str(props["wrap_mode"])
        else:
            self.m_wrap_mode = "repeat"
        
        # Apply inverse gamma correction or not
        if props.has_property("inverse_gamma_correction"):
            self.m_inverse_gamma_correction = bool(props["inverse_gamma_correction"])
        else:
            self.m_inverse_gamma_correction = False
        
        # Reflectance
        if props.has_property("reflectance"):
            self.m_reflectance = mi.Float(props["reflectance"])
        else:
            self.m_reflectance = mi.Float(1.0)
        
        # Load BTF
        self.btf = BrdfInterpolator(self.m_filename, p=self.m_power_parameter, wrap_mode=self.m_wrap_mode)
        
        self.m_flags = mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide
        self.m_components  = [self.m_flags]
        # self.m_flags = reflection_flags
        # self.m_flags = BSDFFlags.DiffuseReflection | BSDFFlags.FrontSide
        # self.m_components = [self.m_flags]
    
    def get_btf(self, wi, wo, uv):
        """
        Get raw BTF values

        wi : enoki.scalar.Vector3f
            Incident direction
        
        wo : enoki.scalar.Vector3f
            Outgoing direction

        uv : enoki.scalar.Vector2f
            UV mapping transformation

        """
        # Camera side direction
        view_elevation, view_azimuth = orthogonal2spherical(wo[0], wo[1], wo[2])[1:]
        
        # Light source side direction
        light_elevation, light_azimuth = orthogonal2spherical(wi[0], wi[1], wi[2])[1:]
        
        # Get coordinate position in the image
        u = uv[0]
        v = uv[1]
        # If warp_mode is mirror, transform coordinates
        if self.m_wrap_mode == "mirror":
            u = mirror_uv(u)
            v = mirror_uv(v)
        
        # Get BTF pixel value
        rgb = self.btf.eval(light_elevation, light_azimuth, view_elevation, view_azimuth, u, v)
        
        # Apply inverse gamma correction
        if self.m_inverse_gamma_correction:
            rgb = rgb ** 2.2
        
        return mi.Vector3f(rgb) * self.m_reflectance
