import numpy as np

TURNTABLE_CENTER = np.array([0.21056884, -0.1938618, 0.0])
R_CAMERA2GRIPPER = np.array([[-0.00369406,  0.99992083,  0.01202885],
                      [-0.00272167,  0.01201883, -0.99992407],
                      [-0.99998947, -0.00372652,  0.00267706]])
# t_CAMERA2GRIPPER = np.array([0.03668630125, -0.02461733549, 0.02764501449])
t_CAMERA2GRIPPER = np.array([0.02634460753, -0.01919117879, 0.03014509088])

CLOCKWISE_ROTATION = True

def build_4x4(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T

def build_ccw_rotz_from_deg(deg):
    """
    Build a 3x3 rotation matrix that rotates by deg degrees about the z-axis.
    The rotation is counter-clockwise.
    """
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    Rz = np.array([[ c, -s, 0.0],
                  [ s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=float)
    return Rz

def build_cw_rotz_from_deg(deg):
    """
    Build a 3x3 rotation matrix that rotates by deg degrees about the z-axis.
    The rotation is clockwise.
    """
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    Rz = np.array([[ c,  s, 0.0],
                  [-s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=float)
    return Rz

def rodrigues_axis_angle(n: np.ndarray, degrees: float | np.ndarray) -> np.ndarray:
    """
    Right-handed CCW rotation about axis n by angle_rad (rad).
    Supports scalar or vector of angles → returns (..,3,3).
    """
    th = np.deg2rad(degrees)
    n = np.asarray(n, float)
    n = n / (np.linalg.norm(n) + 1e-15)
    nx, ny, nz = n
    K = np.array([[0.0, -nz,  ny],
                  [nz,  0.0, -nx],
                  [-ny,  nx,  0.0]], dtype=float)
    I = np.eye(3)
    angle = np.asarray(th, float)[..., None, None]
    Sa = np.sin(angle); Ca = np.cos(angle)
    return I + Sa * K + (1.0 - Ca) * (K @ K)

def build_rot_about_point(R, p=TURNTABLE_CENTER):
    """
    Return a 4x4 transformation matrix that rotates by rotation matrix R about point p.
    
    Args:
        R: 3x3 rotation matrix
        p: 3D point to rotate about (default: TURNTABLE_CENTER)
        
    Returns:
        4x4 transformation matrix that applies rotation R about point p
    """
    T1 = np.eye(4, dtype=float); T1[:3, 3] =  np.asarray(p, dtype=float)
    T2 = np.eye(4, dtype=float); T2[:3, :3] = np.asarray(R, dtype=float)
    T3 = np.eye(4, dtype=float); T3[:3, 3] = -np.asarray(p, dtype=float)
    Tr = T1 @ T2 @ T3
    return Tr