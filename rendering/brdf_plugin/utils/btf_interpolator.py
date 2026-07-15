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
from .coord_system_transfer import spherical2orthogonal
from utils import coords, common, fastmerl


# import tqdm if available
import importlib
tqdm_spec = importlib.util.find_spec("tqdm")
use_tqdm = tqdm_spec is not None
if use_tqdm:
    from tqdm import tqdm

class BtfInterpolator:
    """
    Interpolate and return BTF images at arbitrary angles based on BTFDBB.
    Interpolation uses Inverse Distance Weighting (IDW) from k-nearest neighbors.
    """
    def __init__(self, filepath, k=4, p=4.0):
        """
        Load BTFDBB and create an interpolator.
        (Loading takes a little time)
        
        Parameters
        ----------
        filepath : str
            Path to the BTFDBB file
            If the extension is ".zip", it's "Ubo2003"
            If the extension is ".btf", it's "Ubo2014"
        k : int
            Number of neighboring points used for interpolation
        p : float
            Determines the strength of influence of neighboring points in IDW interpolation
            Smaller p means greater influence of neighboring point values

        Methods
        -------
        angles_xy_to_pixel(tl, pl, tv, pv, x, y)
            Interpolate and return image values at coordinates x, y for angle conditions tl, pl, tv, pv
        angles_uv_to_pixel(tl, pl, tv, pv, u, v)
            Interpolate and return image values at coordinates u, v for angle conditions tl, pl, tv, pv
        angles_to_image(tl, pl, tv, pv)
            Interpolate and return image values for angle conditions tl, pl, tv, pv

        Notes
        -----
        Light source angles are tl, pl
        Observation angles are tv, pv
        """
        # Load BTFDBB
        print("Loading BTFDBB: ", filepath)
        print("Start time: ", datetime.datetime.now())
        
        self.k = k
        self.p = p
        
        is_ubo2003 = filepath.split(".")[-1] == "zip"
        if is_ubo2003:
            self.btf = Ubo2003(filepath)
        else:
            self.btf = Ubo2014(filepath)
        
        # Get image size
        print("Reading one image to get size...")
        
        light_elevations = self.btf.light_elevations
        light_azimuths = self.btf.light_azimuths
        view_elevations = self.btf.view_elevations
        view_azimuths = self.btf.view_azimuths
        
        # Generate load list for reading all file entities and angle information
        btf_items = []
        for i, tl in enumerate(light_elevations):
            for j, pl in enumerate(light_azimuths):
                for k, tv in enumerate(view_elevations):
                    for l, pv in enumerate(view_azimuths):
                        btf_items.append((i, j, k, l, tl, pl, tv, pv))
        
        if use_tqdm:
            btf_items = tqdm(btf_items)
        
        # Read images and angles
        self.pixels = []
        self.angles = []
        # Convert angles from spherical to Cartesian coordinates
        for i, j, k, l, tl, pl, tv, pv in btf_items:
            angle_cartesian = spherical2orthogonal(1, tl, pl) + spherical2orthogonal(1, tv, pv)
            btf_image = self.btf.get_image(i, j, k, l)  # BGR

            # Save images and angles to list
            self.pixels.append(btf_image)
            self.angles.append(angle_cartesian)
        
        # Build KDTree from angle information
        self.angles = np.array(self.angles)
        self.tree = cKDTree(self.angles)
        
        print("Loading completed: ", datetime.datetime.now())
        
    def __uv_to_xy(self, u, v):
        """Convert uv coordinates (float) to xy coordinates (int) corresponding to BTF image
        """
        xf = np.mod(u * (self.__width-1), self.__width)
        yf = np.mod(v * (self.__height-1), self.__height)
        x = np.array(xf, dtype=np.uint16)
        y = np.array(yf, dtype=np.uint16)
        return x, y
        
    def angles_xy_to_pixel(self, tl, pl, tv, pv, x, y):
        """
        Get BTF pixel values at specified angles and coordinates.
        
        Parameters
        ----------
        tl : float
            Light elevation angle [degree]
        pl : float
            Light azimuth angle [degree]  
        tv : float
            View elevation angle [degree]
        pv : float
            View azimuth angle [degree]
        x : int
            x coordinate in image
        y : int
            y coordinate in image
        
        Returns
        -------
        interpolated_pixel : numpy.ndarray, shape=(3,)
            BGR pixel value
        """
        # Convert angles from spherical to Cartesian coordinates
        angle_cartesian = spherical2orthogonal(1, tl, pl) + spherical2orthogonal(1, tv, pv)
        angle_cartesian = np.array(angle_cartesian)
        
        # Execute k-nearest neighbor search
        # Distance is L2 norm
        distances, indices = self.tree.query(angle_cartesian, k=self.k)
        
        # Get BTF image values for corresponding angles and xy coordinates
        neighbor_pixels = []
        for idx in indices:
            pixel = self.pixels[idx][y, x, :]
            neighbor_pixels.append(pixel)
        neighbor_pixels = np.array(neighbor_pixels)
        
        if self.k == 1:
            # No interpolation
            interpolated_pixel = neighbor_pixels[0]
        else:
            # Interpolation using inverse distance weighting
            weights = 1 / (distances ** self.p + 1e-10)  # Avoid division by zero
            weights /= np.sum(weights)  # Normalize
            interpolated_pixel = np.sum(weights[:, np.newaxis] * neighbor_pixels, axis=0)
        
        return interpolated_pixel
    
    def angles_uv_to_pixel(self, tl, pl, tv, pv, u, v):
        """
        Interpolate and return image values at coordinates u, v for angle conditions tl, pl, tv, pv

        Parameters
        ----------
        tl, pl : array of floats
            Light source direction (tl: theta, pl: phi)
        tv, pv : array of floats
            Camera direction (tv: theta, pv: phi)
        u, v : array of floats
            BTF uv image coordinates

        Returns
        -------
        pixel : array of floats
            BTF pixel values
        """ 
        x, y = self.__uv_to_xy(u, v)
        return self.angles_xy_to_pixel(tl, pl, tv, pv, x, y)

    def angles_to_image(self, tl, pl, tv, pv):
        """
        Interpolate and return image for angle conditions tl, pl, tv, pv

        Parameters
        ----------
        tl, pl : array of floats
            Light source direction (tl: theta, pl: phi)
        tv, pv : array of floats
            Camera direction (tv: theta, pv: phi)

        Returns
        -------
        image : array of floats
            BTF image
        """
        x = np.arange(self.__width)
        y = np.arange(self.__height)
        return self.angles_xy_to_pixel(tl, pl, tv, pv, x, y)
