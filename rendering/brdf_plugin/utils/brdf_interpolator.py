"""
Library for interpolating and returning BTF images at arbitrary angles based on BTFDBB.
Obtain interpolated images in ndarray format from arbitrary angles (tl, pl, tv, pv).

For loading BTFDBB, refer to btf-extractor (https://github.com/2-propanol/BTF_extractor).
"""
import os
import datetime
import numpy as np
from scipy.spatial import cKDTree
from btf_extractor import Ubo2003, Ubo2014
from .coord_system_transfer import io_to_hd, xyz2sph, xyz2sph
from utils import coords, common, fastmerl

# import tqdm if available
import importlib
tqdm_spec = importlib.util.find_spec("tqdm")
use_tqdm = tqdm_spec is not None
if use_tqdm:
    from tqdm import tqdm

class BrdfInterpolator:
    """
    Interpolate and return BRDF values at arbitrary angles based on MERL BRDF data.
    Uses the default interpolation method from the Merl class.
    """
    def __init__(self, filepath, p=4.0):
        """
        Load MERL BRDF data and create an interpolator.
        
        Parameters
        ----------
        filepath : str
            Path to the MERL BRDF file
        p : float, optional
            Power parameter for the BRDF interpolation (default is 4.0)
        """
        self.merl_brdf = fastmerl.Merl(filepath)

    def eval_interp(self, theta_h, theta_d, phi_d):
        """
        Lookup the BRDF value for given half diff coordinates and perform an
        interpolation over theta_h, theta_d and phi_d
        
        Parameters
        ----------
        theta_h : array-like
            Half vector elevation angle in radians
        theta_d : array-like
            Diff vector elevation angle in radians
        phi_d : array-like
            Diff vector azimuthal angle in radians
        
        Returns
        -------
        brdf : array of shape (3, n)
            Interpolated BRDF values (RGB) in linear RGB
        """
        return self.merl_brdf.eval_interp(theta_h, theta_d, phi_d)

    def eval(self, wi, wo):
        """
        Evaluate BRDF for given spherical coordinates
        
        Parameters
        ----------
        theta_i, phi_i : array-like
            Incident light direction in spherical coordinates
        theta_o, phi_o : array-like
            Outgoing light direction in spherical coordinates
        
        Returns
        -------
        brdf : array of shape (3, n)
            Interpolated BRDF values (RGB)
        """
        wi = wi.numpy()[0]
        wo = wo.numpy()[0]
        half, diff = io_to_hd(wi, wo)
        r_h, theta_h, phi_h = xyz2sph(*half)
        r_d, theta_d, phi_d = xyz2sph(*diff)
        
        return self.eval_interp(theta_h, theta_d, phi_d)
