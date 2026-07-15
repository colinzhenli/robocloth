import numpy as np
import drjit as dr
import mitsuba as mi

def sph_to_cart(theta, phi):
    """
    Convert from spherical coordinates to Cartesian coordinates.
    theta and phi units are in degrees.
    """
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    x = np.sin(theta_rad) * np.cos(phi_rad)
    y = np.sin(theta_rad) * np.sin(phi_rad)
    z = np.cos(theta_rad)
    return x, y, z

def cart_to_sph(x, y, z):
    """
    Convert from Cartesian coordinates to spherical coordinates.
    theta and phi units are in degrees.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.degrees(np.arccos(z / r))
    phi = np.degrees(np.arctan2(y, x))
    return theta, phi

def uv_flip(u, v):
    """
    Transform UV coordinates so that the boundaries are inverted
    """
    # Original UV coordinates
    u_old = u
    v_old = v
    
    # Transformed UV coordinates
    u_new = 1 - u_old  # Flip U coordinate
    v_new = 1 - v_old  # Flip V coordinate
    
    # Before transformation -> After transformation
    return u_new, v_new


def rotate_vector(v, axis, angle):
    return v * np.cos(angle) + axis * dot(axis, v) * (1 - np.cos(angle)) + cross(axis, v) * np.sin(angle)

def io_to_hd(i, o):
    """Convert input and output vectors to half and difference vectors."""
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    # Compute the halfway vector
    h = normalize(*(i + o))

    # Convert halfway vector to spherical coordinates
    r_h, theta_h, phi_h = xyz2sph(*h)

    # Rotate the input vector i around z-axis by -phi_h
    tmp = rotate_vector(i, z_axis, -phi_h)

    # Rotate the result around y-axis by -theta_h to get the difference vector
    d = rotate_vector(tmp, y_axis, -theta_h)

    return h, d

def hd_to_io(half, diff):
    r_h, theta_h, phi_h = xyz2sph(*half)

    y_axis = np.tile([0.0, 1.0, 0.0], (half.shape[1], 1)).T
    z_axis = np.tile([0.0, 0.0, 1.0], (half.shape[1], 1)).T

    tmp = rotate_vector(diff, y_axis, theta_h)
    wi = normalize(*rotate_vector(tmp, z_axis, phi_h))
    wo = normalize(*(2 * dot(wi, half) * half - wi))
    return wi, wo


def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def cross(v1, v2):
    return np.cross(v1.T, v2.T).T


def xyz2sph(x, y, z):
    r2_xy = x ** 2 + y ** 2
    r = np.sqrt(r2_xy + z ** 2)
    theta = np.arctan2(np.sqrt(r2_xy), z)
    phi = np.arctan2(y, x)
    return np.array([r, theta, phi])


def normalize(x, y, z):
    norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    norm = np.where(norm == 0, np.inf, norm)
    return np.array([x, y, z]) / norm


def sph2xyz(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


# assumes phi_h=0 and both norms=1
def rangles_to_rvectors(theta_h, theta_d, phi_d):
    hx = np.sin(theta_h) * np.cos(0.0)
    hy = np.sin(theta_h) * np.sin(0.0)
    hz = np.cos(theta_h)
    dx = np.sin(theta_d) * np.cos(phi_d)
    dy = np.sin(theta_d) * np.sin(phi_d)
    dz = np.cos(theta_d)
    return np.array([hx, hy, hz, dx, dy, dz])


def rsph_to_rvectors(half_sph, diff_sph):
    hx, hy, hz = sph2xyz(*half_sph)
    dx, dy, dz = sph2xyz(*diff_sph)
    return np.array([hx, hy, hz, dx, dy, dz])


def rvectors_to_rsph(hx, hy, hz, dx, dy, dz):
    half_sph = xyz2sph(hx, hy, hz)
    diff_sph = xyz2sph(dx, dy, dz)
    return half_sph, diff_sph


def rvectors_to_rangles(hx, hy, hz, dx, dy, dz):
    theta_h = np.arctan2(np.sqrt(hx ** 2 + hy ** 2), hz)
    theta_d = np.arctan2(np.sqrt(dx ** 2 + dy ** 2), dz)
    phi_d = np.arctan2(dy, dx)
    return np.array([theta_h, theta_d, phi_d])

def dr_io_to_hd(i, o):
    """Convert input and output vectors to half and difference vectors."""
    # Define the y and z axes
    y_axis = mi.Vector3f(0, 1, 0)
    z_axis = mi.Vector3f(0, 0, 1)

    # Compute the halfway vector
    h = dr.normalize(i + o)

    # Convert halfway vector to spherical coordinates
    r_h, theta_h, phi_h = xyz2sph(*h)

    # Rotate the input vector `i` around the z-axis by -phi_h
    tmp = rotate_vector(i, z_axis, -phi_h)

    # Rotate the result around the y-axis by -theta_h to get the difference vector
    d = rotate_vector(tmp, y_axis, -theta_h)

    return h, d



def dr_xyz2sph(x, y, z):
    """Convert Cartesian coordinates to spherical coordinates using Dr.JIT."""
    r2_xy = x ** 2 + y ** 2
    r = dr.sqrt(r2_xy + z ** 2)              # Radius
    theta = dr.atan2(dr.sqrt(r2_xy), z)      # Inclination angle (0 <= theta <= pi)
    phi = dr.atan2(y, x)                     # Azimuthal angle (-pi <= phi <= pi)
    
    return mi.Vector3f([r, theta, phi])


def dr_rotate_vector(v, axis, angle):
    return v * dr.cos(angle) + axis * dot(axis, v) * (1 - dr.cos(angle)) + dr.cross(axis, v) * dr.sin(angle)