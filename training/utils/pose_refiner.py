import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.io import load_camera_metadata

def hat_so3(w):  # (B,3)->(B,3,3)
    x,y,z = w.unbind(-1)
    O = torch.zeros_like(x)
    return torch.stack([
        torch.stack([ O, -z,  y], -1),
        torch.stack([ z,  O, -x], -1),
        torch.stack([-y,  x,  O], -1),
    ], -2)

def so3_exp(w):
    theta = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-12)
    k = w / theta
    K = hat_so3(k)
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand(K.shape[0],3,3)
    s, c = torch.sin(theta)[...,None], torch.cos(theta)[...,None]
    R = I + s*K + (1-c)*(K@K)
    small = (theta.squeeze(-1) < 1e-4).float()[...,None,None]
    return R*(1-small) + (I + hat_so3(w)) * small  # first-order near 0

def se3_exp(xi):  # xi: (N,6)->R:(N,3,3), t:(N,3)
    w, t = xi[...,:3], xi[...,3:]
    R = so3_exp(w)
    return R, t

def mat_from_Rt(R, t):  # (N,3,3),(N,3)->(N,4,4)
    N = R.shape[0]
    M = torch.eye(4, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(N,1,1)
    M[:,:3,:3] = R
    M[:,:3, 3] = t
    return M

def mat_inv(T):  # (N,4,4)->(N,4,4)
    R = T[:,:3,:3]; t = T[:,:3,3]
    Rt = R.transpose(1,2)
    Minv = torch.eye(4, device=T.device, dtype=T.dtype).unsqueeze(0).repeat(T.shape[0],1,1)
    Minv[:,:3,:3] = Rt
    Minv[:,:3, 3] = -(Rt @ t.unsqueeze(-1)).squeeze(-1)
    return Minv

class GlobalHandEyeRefiner(nn.Module):
    """ One global δξ in the gripper frame (hand-eye correction). """
    def __init__(self, sigma_t_mm=0.5, sigma_r_deg=0.1, json_path='data/camera_metadata.json'):
        super().__init__()
        camera_metadata = load_camera_metadata(json_path)
        # Load camera metadata and create g2w transformation matrices
        self.camera_metadata = camera_metadata
        
        # Create g2w (gripper-to-world) transformation matrices for all cameras
        g2w_matrices = []
        for camera_id in sorted(camera_metadata.keys(), key=int):
            camera_info = camera_metadata[camera_id]
            
            # Extract rotation matrix and position
            rotation_matrix = torch.tensor(camera_info['rotation_matrix'], dtype=torch.float32)
            position = torch.tensor(camera_info['position'], dtype=torch.float32)
            
            # Create 4x4 transformation matrix (gripper-to-world)
            g2w = torch.eye(4, dtype=torch.float32)
            g2w[:3, :3] = rotation_matrix
            g2w[:3, 3] = position
            
            g2w_matrices.append(g2w)
        
        # Stack into (N, 4, 4) tensor and register as buffer
        g2w_tensor = torch.stack(g2w_matrices, dim=0)
        self.register_buffer('T_gw', g2w_tensor.cuda())       
        # Initialize sigma parameters as 0 and make them learnable
        self.sigma_t_mm = nn.Parameter(torch.zeros(3).cuda())
        self.sigma_r_deg = nn.Parameter(torch.zeros(3).cuda())

    def prior(self):
        sigma_t = self.sigma_t_mm/1000.0
        sigma_r = self.sigma_r_deg * 3.1415926535/180.0
        return (sigma_t).pow(2).sum() + (sigma_r).pow(2).sum()
    
    def apply_handeye_delta_to_rays(self, rays, camera_ids):
        """
        rays: (N,12) tensor with [rays_o, rays_d, dxdu, dydv] concatenated
        T_gw: (N,4,4) gripper->world for each ray's view (or per-image then indexed)
        delta_cg: (1,4,4) the global hand-eye delta in gripper frame
        """
        sigma_t = self.sigma_t_mm/1000.0
        sigma_r = self.sigma_r_deg * 3.1415926535/180.0
        xi = torch.cat((sigma_t, sigma_r), dim=0)
        rays = rays.squeeze(0)
        camera_ids = camera_ids.squeeze(0)
        N = rays.shape[0]
        R, t = se3_exp(xi.unsqueeze(0))  # (1,3,3),(1,3)
        delta_cg = mat_from_Rt(R, t)
        T_wg = mat_inv(self.T_gw[camera_ids])
        Delta_w = self.T_gw[camera_ids] @ delta_cg @ T_wg            # (N,4,4)

        Rw = Delta_w[:,:3,:3]                        # (N,3,3)
        tw = Delta_w[:,:3, 3]                        # (N,3)

        # Extract components from concatenated rays tensor
        rays_o = rays[..., :3]
        rays_d = rays[..., 3:6]
        dxdu = rays[..., 6:9]
        dydv = rays[..., 9:12]

        # Apply transformation
        o2 = (Rw @ rays_o.unsqueeze(-1)).squeeze(-1) + tw
        d2 = (Rw @ rays_d.unsqueeze(-1)).squeeze(-1)
        dxdu2 = (Rw @ dxdu.unsqueeze(-1)).squeeze(-1)
        dydv2 = (Rw @ dydv.unsqueeze(-1)).squeeze(-1)

        # Concatenate back into single tensor
        out = torch.cat([o2, d2, dxdu2, dydv2], dim=-1)
        return out.unsqueeze(0), self.prior()

