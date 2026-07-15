#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hand–eye calibration with metric upgrade.

*   Matches robot‐arm poses (scan_log.json) to camera poses (transforms.json)
*   Estimates global **similarity** (scale + R + t) that maps COLMAP’s world to
    the robot base (Umeyama 1991)
*   Runs OpenCV Tsai hand‐eye to obtain camera-to-gripper transform
"""

import os
import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np

def build_4x4(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T

def quat_to_rot_matrix(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x**2 + z**2),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
    ])
    
def quat_to_rot_matrix_scipy(q):
    r = R.from_quat(q)
    return r.as_matrix()

def _cv_to_gl(cv):
    # convert to GL convention used in iNGP
    gl = cv * np.array([1, -1, -1, 1])
    return gl

def _gl_to_cv(gl):
    # convert from GL convention used in iNGP
    cv = gl * np.array([1, -1, -1, 1])
    return cv

def average_rotations(R_list):
    """Average rotations via quaternions (equal weights)."""
    qs = R.from_matrix(R_list).as_quat()          # (N,4)  [x y z w]
    # resolve antipodal symmetry so that dot(q_i,q_0) >= 0
    qs *= np.sign((qs * qs[0]).sum(-1, keepdims=True))
    q_mean = qs.mean(0)
    q_mean /= np.linalg.norm(q_mean)
    return R.from_quat(q_mean).as_matrix()

def official_umeyama(X, Y):
    """
    Estimates the Sim(3) transformation between `X` and `Y` point sets.

    Estimates c, R and t such as c * R @ X + t ~ Y.

    Parameters
    ----------
    X : numpy.array
        (m, n) shaped numpy array. m is the dimension of the points,
        n is the number of points in the point set.
    Y : numpy.array
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
        That is, Y[:, i] must be the point corresponding to X[:, i].
    
    Returns
    -------
    c : float
        Scale factor.
    R : numpy.array
        (3, 3) shaped rotation matrix.
    t : numpy.array
        (3, 1) shaped translation vector.
    """
    mu_x = X.mean(axis=1).reshape(-1, 1)
    mu_y = Y.mean(axis=1).reshape(-1, 1)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    return c, R, t

def sim3_umeyama(P, Q, with_scale=True):
    """
    P (N,3): COLMAP camera centres
    Q (N,3): robot gripper positions
    Returns R (3,3), t (3,), s (scalar)
    """
    P, Q = np.asarray(P), np.asarray(Q)
    mu_P, mu_Q = P.mean(0), Q.mean(0)
    P0, Q0 = P - mu_P, Q - mu_Q

    # equation 12 in Umeyama
    Sigma = Q0.T @ P0 / len(P)
    U, D, VT = np.linalg.svd(Sigma)
    S = np.diag([1, 1, np.sign(np.linalg.det(U) * np.linalg.det(VT))])
    R = U @ S @ VT

    if with_scale:
        var_P = (P0**2).sum() / len(P)
        s = np.trace(np.diag(D) @ S) / var_P
    else:
        s = 1.0

    t = mu_Q - s * R @ mu_P
    return s, R, t


def camera_centres_from_c2w(c2w_list):
    """Return (N,3) array of camera centres from camera-to-world matrices."""
    return np.array([m[:3, 3] for m in c2w_list])

# -----------------------------------------------------------------------------#
#  Matching images ↔ robot log
# -----------------------------------------------------------------------------#
def extract_phi_theta_from_filename(filename):
    """Extract φ and θ from `scan-phi0.583_theta4.608.png`."""
    # Handle format: 'scan-0-phi0.079_theta3.142.png'
    base = (
        filename.replace(".png", "")
        .replace(".jpg", "")
        .replace("scan-", "")
    )   
    if "-phi" in base and "_theta" in base:
        # Split on the phi part
        phi_part = base.split("-phi")[1]
        phi_theta_parts = phi_part.split("_theta")
        phi = float(phi_theta_parts[0])
        theta = float(phi_theta_parts[1])
        return phi, theta

    parts = base.split("_")
    phi = float([p for p in parts if p.startswith("phi")][0][3:]) # 3 if previous names
    theta = float([p for p in parts if p.startswith("theta")][0][5:]) # 5 if previous names
    return phi, theta


def find_matching_robot_pose(trg_phi, trg_theta, scan_log, tol=1e-3):
    """Return index in scan_log that best matches (φ,θ)."""
    best_i, best_d = None, float("inf")
    for i, e in enumerate(scan_log):
        d = max(abs(trg_phi - e["phi"]), abs(trg_theta - e["theta"]))
        if d < best_d:
            best_d, best_i = d, i
    return (best_i, best_d) if best_d <= tol else (None, best_d)


def load_matched_robot_poses(scan_log_path, transforms_json_path, tol=1e-3):
    """Match camera frames to robot poses via φ/θ."""
    with open(scan_log_path, "r") as f:
        scan_log = json.load(f)
    with open(transforms_json_path, "r") as f:
        tjson = json.load(f)

    robot_poses, cam_c2w, info = [], [], []

    for i, frame in enumerate(tjson["frames"]):
        fname = frame["file_path"].removeprefix("images/")
        
        # Skip files that don't contain "theta" in the filename
        if "theta" not in fname:
            continue
            
        phi, theta = extract_phi_theta_from_filename(fname)

        idx, dist = find_matching_robot_pose(phi, theta, scan_log, tol)
        if idx is None:
            continue

        # camera pose
        cam_c2w.append(_cv_to_gl(np.array(frame["transform_matrix"])))

        # robot pose (base → holder)
        entry = scan_log[idx]
        T = np.eye(4)
        T[:3, :3] = np.array(entry["rotation_matrix"])
        T[:3, 3] = np.array(entry["position"])
        robot_poses.append(T)

        info.append(
            dict(camera_idx=i, robot_idx=idx, filename=fname,
                 target_phi=phi, target_theta=theta,
                 robot_phi=entry["phi"], robot_theta=entry["theta"],
                 distance=dist)
        )

    print(f"Matched {len(robot_poses)}/{len(tjson['frames'])} frames.")
    return robot_poses, cam_c2w, info


def parse_colmap_images_txt(images_txt_path):
    """
    Parse COLMAP images.txt and return a dictionary of image_name → 4x4 cam-to-world transform.
    """
    cam_c2w_dict = {}
    with open(images_txt_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#") or len(line) == 0:
            i += 1
            continue

        # Parse camera line
        tokens = line.split()
        if len(tokens) < 10:
            i += 1
            continue
        image_id = int(tokens[0])
        qw, qx, qy, qz = map(float, tokens[1:5])
        tx, ty, tz = map(float, tokens[5:8])
        cam_id = int(tokens[8])
        image_name = tokens[9]

        # Convert to 4x4 cam-to-world matrix
        # R = quat_to_rot_matrix([qw, qx, qy, qz])  # rotation matrix
        R = quat_to_rot_matrix_scipy([qw, qx, qy, qz])

        t = np.array([tx, ty, tz])               # translation
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = t
        cam_c2w_dict[image_name] = T
        i += 2  # skip the 2D point line
    return cam_c2w_dict


def load_matched_robot_poses_colmap_images_txt(scan_log_path, images_txt_path, tol=1e-3):
    """
    Load robot poses and match to COLMAP poses from images.txt via φ/θ in the filename.
    """
    with open(scan_log_path, "r") as f:
        scan_log = json.load(f)

    colmap_c2w_dict = parse_colmap_images_txt(images_txt_path)

    robot_poses, cam_c2w, info = [], [], []

    for fname, c2w in colmap_c2w_dict.items():
        if "theta" not in fname:
            continue

        phi, theta = extract_phi_theta_from_filename(fname)
        idx, dist = find_matching_robot_pose(phi, theta, scan_log, tol)
        if idx is None:
            continue

        # COLMAP camera pose
        cam_c2w.append(c2w)

        # Robot gripper → base transform
        entry = scan_log[idx]
        T = np.eye(4)
        T[:3, :3] = np.array(entry["rotation_matrix"])
        T[:3, 3] = np.array(entry["position"])
        robot_poses.append(T)

        info.append(dict(
            robot_idx=idx,
            filename=fname,
            target_phi=phi,
            target_theta=theta,
            robot_phi=entry["phi"],
            robot_theta=entry["theta"],
            distance=dist
        ))

    print(f"Matched {len(robot_poses)}/{len(colmap_c2w_dict)} frames.")
    return robot_poses, cam_c2w, info

# -----------------------------------------------------------------------------#
#  Hand–eye
# -----------------------------------------------------------------------------#
def hand_eye_calibration(robot_T, cam_c2b):
    """
    Tsai hand-eye calibration.

    Parameters
    ----------
    robot_T : list of 4×4   base → holder
    cam_c2b : list of 4×4   camera → base   (same frame as robot_T)

    Returns
    -------
    R_cam2holder, t_cam2holder
    """
    # OpenCV expects:
    #   R_gripper2base, t_gripper2base
    #   R_target2cam ,  t_target2cam
    Rg2b = np.array([T[:3, :3] for T in robot_T])
    tg2b = np.array([T[:3, 3]  for T in robot_T])

    # invert cam_c2b → base2cam, then invert again? No:
    # We need TARGET→CAM, i.e. world→cam.  Here "world" == base.
    Rw2c = np.array([np.linalg.inv(T)[:3, :3] for T in cam_c2b])
    tw2c = np.array([np.linalg.inv(T)[:3, 3]  for T in cam_c2b])

    R_cam2grip, t_cam2grip = cv2.calibrateHandEye(
        Rg2b, tg2b, Rw2c, tw2c, method=cv2.CALIB_HAND_EYE_TSAI
    )
    return R_cam2grip, t_cam2grip

# -----------------------------------------------------------------------------#
#  Build world-to-camera transforms with metric scale only
# -----------------------------------------------------------------------------#
def build_scaled_w2c_using_scale_only(c2w_list, R, t, s):
    """
    Convert COLMAP camera-to-world poses (arbitrary scale) to
    *metric* world-to-camera poses, given the Sim(3) alignment
    (R, t, s) obtained from Umeyama.

    Parameters
    ----------
    c2w_list : list | (N,4,4) array
        Original COLMAP camera-to-world poses.
    R : (3,3) array
        Rotation that brings COLMAP's world frame into the robot
        base frame (output of Umeyama).
    t : (3,) array
        Translation that brings COLMAP's world origin into the
        robot base frame (same Umeyama fit).
    s : float
        Uniform scale factor   (metres / COLMAP-unit).

    Returns
    -------
    R_target2cam : (N,3,3) ndarray
        Rotation matrices of **world/target → camera** (a.k.a. W2C).
    t_target2cam : (N,3) ndarray
        Translation vectors  of **world/target → camera**.
    """
    R_t2c, t_t2c = [], []

    Rg = np.asarray(R)
    tg = np.asarray(t).reshape(3, 1)

    for C2W in c2w_list:
        C2W = np.asarray(C2W)

        # --- decompose original COLMAP pose --------------------------
        R_c2w = C2W[:3, :3]            # 3×3
        t_c2w = C2W[:3,  3].reshape(3,1)

        # --- apply uniform scale & global Sim(3) ---------------------
        # R_c2w_metric = Rg @ R_c2w                      # rotate axes
        # t_c2w_metric = tg + s * (Rg @ t_c2w)           # scale + rotate + shift
        R_c2w_metric = R_c2w
        t_c2w_metric = s * t_c2w

        C2W_metric = np.eye(4)
        C2W_metric[:3,:3] = R_c2w_metric
        C2W_metric[:3, 3] = t_c2w_metric.ravel()

        # --- invert to world-to-camera -------------------------------
        W2C_metric = np.linalg.inv(C2W_metric)

        R_t2c.append(W2C_metric[:3,:3])
        t_t2c.append(W2C_metric[:3, 3])

    return np.stack(R_t2c), np.stack(t_t2c)

# #  Synthetic debug harness
# # -----------------------------------------------------------------------------#
# def debug_with_synthetic_data(robot_T):
#     """
#     Use the first 5 robot poses to create perfect synthetic data and verify
#     that calibrateHandEye recovers the known camera→gripper transform.
#     """
#     if len(robot_T) < 5:
#         raise RuntimeError("Need ≥5 robot poses for the synthetic test")
#     robot_T = robot_T[:15]                               # take first five

#     # --- 1. Ground-truth camera → gripper ------------------------------------
#     T_cg = np.eye(4)
#     # 90° about +Z
#     T_cg[:3, :3] = np.array([[0, -1, 0],
#                              [1,  0, 0],
#                              [0,  0, 1]])
#     # T_cg[:3, :3] = np.array([[1, 0, 0],
#     #                          [0, 1, 0],
#     #                          [0, 0, 1]])
#     T_cg[:3,  3] = np.array([2.0, 2.0, 0.0])            # metres

#     # --- 2. Target pose in base frame ----------------------------------------
#     T_tb = np.eye(4)
#     T_tb[:3, 3] = np.array([100.0, 0.0, 0.0])           # (100,0,0) m

#     # --- 3. Build R_target2cam / t_target2cam --------------------------------
#     R_t2c, t_t2c = [], []
#     for Tg2b in robot_T:
#         # camera pose in base:  Tc2b = Tg2b ⋅ Tcg
#         Tc2b = Tg2b @ T_cg
#         # target → camera:      Tt2c = Tc2b⁻¹ ⋅ Ttb
#         Tt2c = np.linalg.inv(Tc2b) @ T_tb

#         R_t2c.append(Tt2c[:3, :3])
#         t_t2c.append(Tt2c[:3, 3])

#     R_t2c = np.array(R_t2c)
#     t_t2c = np.array(t_t2c)

#     # --- 4. Split robot motions ---------------------------------------------
#     R_g2b = np.array([T[:3, :3] for T in robot_T])
#     t_g2b = np.array([T[:3,  3] for T in robot_T])

#     # --- 5. Hand–eye ---------------------------------------------------------
#     R_syn, t_syn = cv2.calibrateHandEye(
#         R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_TSAI
#     )

#     # --- 6. Compare ----------------------------------------------------------
#     print("=== Synthetic test ===")
#     print("Ground-truth R_c2g:\n", T_cg[:3, :3])
#     print("Estimated   R_c2g:\n", R_syn)
#     print("Angle error (deg):",
#           np.rad2deg(cv2.Rodrigues(R_syn @ T_cg[:3, :3].T)[0].ravel()).round(6))

#     print("Ground-truth t_c2g:", T_cg[:3, 3])
#     print("Estimated   t_c2g:", t_syn.ravel())
#     print("Transl. error (m):", np.linalg.norm(t_syn.ravel() - T_cg[:3, 3]))

def estimate_camera2gripper(scan_log_path, transforms_json_path, phi_theta_tol=1e-3):
    """
    Estimate camera-to-gripper transformation using hand-eye calibration.
    
    Parameters
    ----------
    scan_log_path : str | Path
        Path to robot log with gripper→base poses
    transforms_json_path : str | Path
        Path to COLMAP / NeRF style transforms.json
    phi_theta_tol : float, optional
        Tolerance for matching phi/theta angles (default: 1e-3)
    
    Returns
    -------
    R_c2g : (3,3) array
        Rotation matrix from camera to gripper
    t_c2g : (3,) array
        Translation vector from camera to gripper
    """
    robot_T, cam_c2w, _ = load_matched_robot_poses(
        scan_log_path, transforms_json_path, phi_theta_tol
    )
    
    # Convert COLMAP camera poses to metric scale using similarity transform
    cam_centres_world = camera_centres_from_c2w(cam_c2w)
    gripper_positions = np.array([T[:3, 3] for T in robot_T])
    
    # Estimate similarity transformation (scale + rotation + translation)
    R, t, s = sim3_umeyama(cam_centres_world, gripper_positions)
    
    # Build scaled world-to-camera transforms
    cam_c2b = build_scaled_w2c_using_scale_only(cam_c2w, R, t, s)
    
    # Perform hand-eye calibration
    R_c2g, t_c2g = hand_eye_calibration(robot_T, cam_c2b)
    
    return R_c2g, t_c2g

def estimate_world2base(scan_log_path, transforms_json_path,
                        R_c2g, t_c2g, phi_theta_tol=1e-3):
    """
    Parameters
    ----------
    scan_log_path          : str | Path   (robot log with gripper→base poses)
    transforms_json_path   : str | Path   (COLMAP / NeRF style transforms.json)
    R_c2g, t_c2g           : (3,3) and (3,)   fixed camera→gripper transform

    Returns
    -------
    T_BW : (4,4)  homogeneous matrix [  s·R   t ]
                                    [   0     1 ]
           that maps COLMAP‑world coordinates to robot‑base coordinates.
    """
    # robot_T, cam_c2w, _ = load_matched_robot_poses(
    #     scan_log_path, transforms_json_path, phi_theta_tol
    # )
    robot_T, cam_c2w, _ = load_matched_robot_poses_colmap_images_txt(
        scan_log_path, images_txt_path, phi_theta_tol
    )
    # ---------- per‑frame camera→base -----------------------------------------
    T_c2g = build_4x4(R_c2g, t_c2g)
    cam_centres_base, cam_centres_world = [], []
    for Tg2b, C2W in zip(robot_T, cam_c2w):
        Tc2b = Tg2b @ T_c2g                      # camera→base
        # Tc2b = Tg2b                      # camera→base
        cam_centres_base .append(Tc2b[:3, 3])
        cam_centres_world.append(C2W[:3, 3])    # still arbitrary units

    cam_centres_base  = np.vstack(cam_centres_base)
    cam_centres_world = np.vstack(cam_centres_world)

    # ---------- similarity (scale,R,t)  world → base --------------------------
    s, R, t = sim3_umeyama(cam_centres_world, cam_centres_base)
    s_official, R_official, t_official = official_umeyama(cam_centres_world.transpose(), cam_centres_base.transpose())

    T_BW = np.eye(4)
    T_BW[:3, :3] = s * R
    T_BW[:3,  3] = t
    return T_BW

# def estimate_world2base(scan_log_path, transforms_json_path,
#                         R_c2g, t_c2g, phi_theta_tol=1e-3):
#     """
#     Returns
#     -------
#     T_BW : (4,4)  similarity matrix  world → robot‑base
#              [ s R  t ]
#              [  0   1 ]
#     """
#     # ---------- load & match --------------------------------------------------
#     robot_T, cam_c2w, _ = load_matched_robot_poses(
#         scan_log_path, transforms_json_path, phi_theta_tol
#     )
#     if len(robot_T) < 3:
#         raise RuntimeError("Need ≥3 matched poses; got", len(robot_T))

#     T_c2g = build_4x4(R_c2g, t_c2g)

#     camera_centres_base, camera_centres_world = [], []
#     TBW_candidates = []

#     for Tg2b, C2W in zip(robot_T, cam_c2w):
#         Tc2b = Tg2b @ T_c2g               # camera → base
#         camera_centres_base .append(Tc2b[:3, 3])
#         camera_centres_world.append(C2W[:3, 3])

#         T_bw_i = Tc2b @ np.linalg.inv(C2W)   # world → base   (contains scale)
#         TBW_candidates.append(T_bw_i)

#     camera_centres_base  = np.vstack(camera_centres_base)
#     camera_centres_world = np.vstack(camera_centres_world)

#     # ---------- 1. uniform scale from centre distances -----------------------
#     s, _, _ = sim3_umeyama(camera_centres_world, camera_centres_base)

#     # ---------- 2. split each candidate into R_i / t_i -----------------------
#     R_list, t_list = [], []
#     for T_bw in TBW_candidates:
#         R_scaled = T_bw[:3, :3]           # this is s·R
#         R_list.append(R_scaled / s)       # -> pure rotation
#         t_list.append(T_bw[:3, 3])

#     R_mean = average_rotations(np.stack(R_list))
#     t_mean = np.mean(np.vstack(t_list), axis=0)

#     # ---------- 3. assemble final similarity ---------------------------------
#     T_BW = np.eye(4)
#     T_BW[:3, :3] = s * R_mean
#     T_BW[:3, 3]  = t_mean
#     return T_BW

def transform_mesh_to_base(mesh_path, T_BW, output_path=None):
    """
    Load a mesh and transform it from COLMAP world coordinates to robot base coordinates.
    
    Parameters
    ----------
    mesh_path : str | Path
        Path to the input mesh file (e.g., .ply, .obj)
    T_BW : (4,4) array
        Transformation matrix from COLMAP world to robot base coordinates
    output_path : str | Path, optional
        Path to save the transformed mesh. If None, returns the transformed mesh object.
    
    Returns
    -------
    mesh : open3d.geometry.TriangleMesh
        The transformed mesh object
    """
    import open3d as o3d
    
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.vertices) == 0:
        raise ValueError(f"Failed to load mesh from {mesh_path}")
    
    # Apply the transformation
    mesh.transform(T_BW)
    
    # Save if output path is provided
    if output_path is not None:
        success = o3d.io.write_triangle_mesh(str(output_path), mesh)
        if not success:
            raise RuntimeError(f"Failed to save mesh to {output_path}")
        print(f"Transformed mesh saved to: {output_path}")
    
    return mesh


# -----------------------------------------------------------------------------#
#  Main (unchanged except for the test call at the end)
# -----------------------------------------------------------------------------#
def main(scan_log_path, transforms_json_path, mesh_path, tol=1e-2):
    # R_c2g, t_c2g = estimate_camera2gripper(scan_log_path, transforms_json_path, tol)
    # Use pre-computed hand-eye calibration results
    R_c2g = np.array([[-0.00369406,  0.99992083,  0.01202885],
                      [-0.00272167,  0.01201883, -0.99992407],
                      [-0.99998947, -0.00372652,  0.00267706]])
    t_c2g = np.array([36.68630125, -24.61733549, 27.64501449])

    # # Check if mesh has UV mapping
    # import open3d as o3d
    # mesh = o3d.io.read_triangle_mesh(mesh_path)
    # if len(mesh.triangle_uvs) > 0:
    #     print(f"Mesh has UV mapping with {len(mesh.triangle_uvs)} UV coordinates")
    #     output_path = mesh_path.replace(".ply", "_transformed_with_uv.ply")
    # else:
    #     print("Mesh does not have UV mapping")
    #     output_path = mesh_path.replace(".ply", "_transformed.ply")
    T_BW = estimate_world2base(scan_log_path, transforms_json_path, R_c2g, t_c2g, tol)
    transform_mesh_to_base(mesh_path, T_BW, output_path=mesh_path.replace(".ply", "_corrected_convention_transformed.ply"))


if __name__ == "__main__":
    scan_log_path = (
        "/media/raid/cloth/rot_axis/scan_log_0813_cloth.json"
    )
    transforms_json_path = (
        "/media/raid/cloth/Pipeline_test_images_Aug15/transforms.json"
    )
    images_txt_path = "/media/raid/cloth/Pipeline_test_images_Aug15/undistorted/sparse/0/images.txt"
    mesh_path = "./output/d90427d5-b/train/ours_30000/fuse_post.ply"
    main(scan_log_path, transforms_json_path, mesh_path, tol=0.01)
