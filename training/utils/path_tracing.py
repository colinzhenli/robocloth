"""Differentiable single-bounce path tracing (paper Eq. 2).

The renderer evaluates L_o = f_r * L_e(y)*Lambda(theta) * V * cos(theta_i)/r^2
over the finite LED area with S Monte-Carlo samples; only f_r is predicted by
the network — all geometric/radiometric factors are computed analytically
from calibrated quantities. Live entry points:
  points_path_tracing_real_area_emitter      - stage 1 (per-point batches)
  batched_path_tracing_tbn_real_area_emitter - stage 2 (per-pixel rays, TBN)
  batched_path_tracing_tbn_preset_emitter    - Bonn/MERL comparison flows
"""
import torch
import torch.nn.functional as NF

# import mitsuba
# mitsuba.set_variant('cuda_ad_rgb')

from .ops import *

def ray_intersect(scene,xs,ds):
    """ warpper of mitsuba ray-mesh intersection 
    Args:
        xs: Bx3 pytorch ray origin
        ds: Bx3 pytorch ray direction
    Return:
        positions: Bx3 intersection location
        normals: Bx3 normals
        uvs: Bx2 uv coordinates
        idx: B triangle indices, -1 indicates no intersection
        valid: B whether a valid intersection
    """

    # convert pytorch tensor to mitsuba
    xs_mi = mitsuba.Point3f(xs[...,0],xs[...,1],xs[...,2])
    ds_mi = mitsuba.Vector3f(ds[...,0],ds[...,1],ds[...,2])
    rays_mi = mitsuba.Ray3f(xs_mi,ds_mi)
    
    ret = scene.ray_intersect_preliminary(rays_mi)
    idx = mitsuba.Int(ret.prim_index).torch().long()
    ret = ret.compute_surface_interaction(rays_mi)
    
    positions = ret.p.torch()
    normals = ret.sh_frame.n.torch()
    normals = NF.normalize(normals,dim=-1)
    
    # check if invalid intersection
    ts  = ret.t.torch()
    valid = (~ts.isinf())
    
    idx[~valid] = -1
    normals = double_sided(-ds,normals)
    return positions,normals,ret.uv.torch(),idx,valid

# def ray_sphere_intersect(self, ray_o, ray_d):
#     """ Ray-sphere intersection test
#     Args:
#         ray_o: Bx3 ray origins
#         ray_d: Bx3 ray directions (normalized)
#     Returns:
#         hit_pos: Bx3 intersection points
#         normals: Bx3 surface normals
#         valid: B whether ray hits sphere
#     """
#     # Solve quadratic equation for ray-sphere intersection
#     oc = ray_o - self.position
#     a = (ray_d * ray_d).sum(-1)
#     b = 2.0 * (oc * ray_d).sum(-1)
#     c = (oc * oc).sum(-1) - self.radius * self.radius
#     disc = b * b - 4 * a * c
    
#     valid = disc > 0
#     t = torch.zeros_like(disc)
#     t[valid] = (-b[valid] - torch.sqrt(disc[valid])) / (2.0 * a[valid])
#     valid = valid & (t > 0)

#     # Compute intersection points and normals
#     hit_pos = ray_o + ray_d * t.unsqueeze(-1)
#     normals = NF.normalize(hit_pos - self.position, dim=-1)
    
#     return hit_pos, normals, valid

def ray_hemisphere_intersect_TBN(xs, ds, center=[0, 0, 0], radius=1.0):
    """
    Ray-hemisphere intersection using mathematical computation.
    Hemisphere is defined by center and radius, with the hemisphere on the positive z half.
    
    Args:
        xs: (N, 3) ray origins
        ds: (N, 3) ray directions (normalized)
        center: [x, y, z] center of the hemisphere (sphere center)
        radius: radius of the hemisphere
    
    Returns:
        positions: (N, 3) intersection points
        normals: (N, 3) surface normals (pointing outward)
        uv: (N, 2) texture coordinates [0,1] using spherical mapping
        dp_du: (N, 3) surface partial derivative wrt u
        dp_dv: (N, 3) surface partial derivative wrt v
        idx: (N,) primitive index (-1 for invalid)
        valid: (N,) boolean mask for valid intersections
        TBN: (N, 3, 3) tangent-bitangent-normal frame
    """
    device = xs.device
    N = xs.shape[0]
    
    center_pt = torch.tensor(center, dtype=xs.dtype, device=device)
    
    # Ray-sphere intersection: solve quadratic equation
    # |xs + t*ds - center|^2 = radius^2
    # oc = xs - center
    # |oc + t*ds|^2 = radius^2
    # t^2*(ds·ds) + 2t*(oc·ds) + (oc·oc) - radius^2 = 0
    oc = xs - center_pt
    a = (ds * ds).sum(-1)  # ds·ds (should be 1 if normalized)
    b = 2.0 * (oc * ds).sum(-1)  # 2*(oc·ds)
    c = (oc * oc).sum(-1) - radius * radius  # oc·oc - r^2
    
    discriminant = b * b - 4 * a * c
    
    # Check if ray intersects sphere
    valid = discriminant >= 0
    
    # Compute both intersection points (t1 < t2)
    sqrt_disc = torch.zeros(N, dtype=xs.dtype, device=device)
    sqrt_disc[valid] = torch.sqrt(discriminant[valid])
    
    t1 = torch.zeros(N, dtype=xs.dtype, device=device)
    t2 = torch.zeros(N, dtype=xs.dtype, device=device)
    t1[valid] = (-b[valid] - sqrt_disc[valid]) / (2.0 * a[valid])
    t2[valid] = (-b[valid] + sqrt_disc[valid]) / (2.0 * a[valid])
    
    # Compute intersection points for both t values
    pos1 = xs + ds * t1.unsqueeze(-1)
    pos2 = xs + ds * t2.unsqueeze(-1)
    
    # Check which intersection is on positive z half (relative to center)
    local_pos1 = pos1 - center_pt
    local_pos2 = pos2 - center_pt
    
    on_hemisphere1 = local_pos1[..., 2] >= -1e-6  # positive z half
    on_hemisphere2 = local_pos2[..., 2] >= -1e-6  # positive z half
    
    # Choose the closer valid intersection that's on the hemisphere and in front of ray
    t1_valid = valid & (t1 > 1e-6) & on_hemisphere1
    t2_valid = valid & (t2 > 1e-6) & on_hemisphere2
    
    # Use t1 if valid, otherwise use t2
    t = torch.where(t1_valid, t1, t2)
    valid = t1_valid | t2_valid
    
    # For cases where both are valid, use the smaller t (closer intersection)
    both_valid = t1_valid & t2_valid
    t[both_valid] = torch.minimum(t1[both_valid], t2[both_valid])
    
    # Compute final intersection points
    positions = xs + ds * t.unsqueeze(-1)
    local_pos = positions - center_pt
    
    # Normals: pointing outward from sphere center
    normals = NF.normalize(local_pos, dim=-1)
    normals = double_sided(-ds, normals)
    
    # Compute spherical coordinates for UV mapping
    # theta: azimuthal angle in xy-plane from +x axis [0, 2pi] -> u [0, 1]
    # phi: polar angle from +z axis [0, pi/2] for hemisphere -> v [0, 1]
    # Note: for hemisphere on +z, phi ranges from 0 (top) to pi/2 (equator)
    
    # Normalize local position to unit sphere
    local_normalized = local_pos / radius
    
    # phi = arccos(z), ranges [0, pi/2] for z in [1, 0]
    phi = torch.acos(torch.clamp(local_normalized[..., 2], -1.0, 1.0))
    
    # theta = atan2(y, x), ranges [-pi, pi]
    theta = torch.atan2(local_normalized[..., 1], local_normalized[..., 0])
    
    # Map to UV coordinates [0, 1]
    uv = torch.zeros(N, 2, dtype=xs.dtype, device=device)
    uv[:, 0] = (theta / (2 * torch.pi)) + 0.5  # u: theta mapped to [0, 1]
    uv[:, 1] = phi / (torch.pi / 2)  # v: phi mapped to [0, 1] for hemisphere
    
    # Surface partials for spherical parameterization:
    # p(u, v) = center + r * [sin(phi)*cos(theta), sin(phi)*sin(theta), cos(phi)]
    # where theta = 2*pi*(u-0.5) and phi = v*pi/2
    # 
    # dp/du = r * sin(phi) * 2*pi * [-sin(theta), cos(theta), 0]
    # dp/dv = r * (pi/2) * [cos(phi)*cos(theta), cos(phi)*sin(theta), -sin(phi)]
    
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    dp_du = torch.zeros(N, 3, dtype=xs.dtype, device=device)
    dp_du[:, 0] = -radius * sin_phi * 2 * torch.pi * sin_theta
    dp_du[:, 1] = radius * sin_phi * 2 * torch.pi * cos_theta
    dp_du[:, 2] = 0
    
    dp_dv = torch.zeros(N, 3, dtype=xs.dtype, device=device)
    dp_dv[:, 0] = radius * (torch.pi / 2) * cos_phi * cos_theta
    dp_dv[:, 1] = radius * (torch.pi / 2) * cos_phi * sin_theta
    dp_dv[:, 2] = -radius * (torch.pi / 2) * sin_phi
    
    # TBN frame: tangent (u-direction), bitangent (v-direction), normal
    # Tangent is along theta direction (normalized dp_du)
    tangent = torch.zeros(N, 3, dtype=xs.dtype, device=device)
    tangent[:, 0] = -sin_theta
    tangent[:, 1] = cos_theta
    tangent[:, 2] = 0
    
    # Bitangent is along phi direction (normalized dp_dv)
    bitangent = torch.zeros(N, 3, dtype=xs.dtype, device=device)
    bitangent[:, 0] = cos_phi * cos_theta
    bitangent[:, 1] = cos_phi * sin_theta
    bitangent[:, 2] = -sin_phi
    
    # Handle degenerate case at pole (phi = 0, sin_phi = 0)
    at_pole = sin_phi.abs() < 1e-6
    tangent[at_pole, 0] = 1.0
    tangent[at_pole, 1] = 0.0
    bitangent[at_pole, 0] = 0.0
    bitangent[at_pole, 1] = 1.0
    bitangent[at_pole, 2] = 0.0
    
    TBN = torch.stack([tangent, bitangent, normals], dim=-1)  # [N, 3, 3]
    
    # Primitive index (0 for valid hits, -1 for invalid)
    idx = torch.zeros(N, dtype=torch.long, device=device)
    idx[~valid] = -1
    
    return positions, normals, uv, dp_du, dp_dv, idx, valid, TBN

def ray_rectangle_intersect_TBN(xs, ds, center=[0, 0, 0], width=0.4, length=0.4):
    """
    Ray-rectangle intersection using mathematical computation.
    Rectangle is defined by center, width (x), length (y), with normal along z-axis.
    
    Args:
        xs: (N, 3) ray origins
        ds: (N, 3) ray directions (normalized)
        center: [x, y, z] center of rectangle
        width: width along x-axis
        length: length along y-axis
    
    Returns:
        positions: (N, 3) intersection points
        normals: (N, 3) surface normals (along z-axis)
        uv: (N, 2) texture coordinates [0,1]
        dp_du: (N, 3) surface partial derivative wrt u
        dp_dv: (N, 3) surface partial derivative wrt v
        idx: (N,) primitive index (-1 for invalid)
        valid: (N,) boolean mask for valid intersections
        TBN: (N, 3, 3) tangent-bitangent-normal frame
    """
    device = xs.device
    N = xs.shape[0]
    
    # Rectangle plane: normal is [0, 0, 1] in local space
    center_pt = torch.tensor(center, dtype=xs.dtype, device=device)
    plane_normal = torch.tensor([0.0, 0.0, 1.0], dtype=xs.dtype, device=device)
    
    # Ray-plane intersection: t = (center - xs) · n / (ds · n)
    numerator = ((center_pt - xs) * plane_normal).sum(-1)
    denominator = (ds * plane_normal).sum(-1)
    
    # Check if ray is parallel to plane
    valid = denominator.abs() > 1e-8
    t = torch.zeros(N, dtype=xs.dtype, device=device)
    t[valid] = numerator[valid] / denominator[valid]
    
    # Check if intersection is in front of ray
    valid = valid & (t > 1e-6)
    
    # Compute intersection points
    positions = xs + ds * t.unsqueeze(-1)
    
    # Check if intersection is within rectangle bounds
    local_pos = positions - center_pt
    half_width = width / 2.0
    half_length = length / 2.0
    
    in_bounds = (local_pos[..., 0].abs() <= half_width) & \
                (local_pos[..., 1].abs() <= half_length)
    valid = valid & in_bounds
    
    # Compute UV coordinates [0, 1]
    uv = torch.zeros(N, 2, dtype=xs.dtype, device=device)
    uv[:, 0] = (local_pos[:, 0] / width) + 0.5  # u: [0, 1]
    uv[:, 1] = (local_pos[:, 1] / length) + 0.5  # v: [0, 1]
    
    # Surface partials: dp/du and dp/dv
    # u maps to x-axis, v maps to y-axis
    dp_du = torch.zeros(N, 3, dtype=xs.dtype, device=device)
    dp_dv = torch.zeros(N, 3, dtype=xs.dtype, device=device)
    dp_du[:, 0] = width   # ∂p/∂u = width along x
    dp_dv[:, 1] = length  # ∂p/∂v = length along y
    
    # Normals (all pointing along z-axis)
    normals = plane_normal.unsqueeze(0).expand(N, 3).clone()
    normals = double_sided(-ds, normals)
    
    # TBN frame: tangent (u-direction), bitangent (v-direction), normal (z)
    tangent = torch.zeros(N, 3, dtype=xs.dtype, device=device)
    tangent[:, 0] = 1.0  # tangent along x (u-direction)
    
    bitangent = torch.zeros(N, 3, dtype=xs.dtype, device=device)
    bitangent[:, 1] = 1.0  # bitangent along y (v-direction)
    
    TBN = torch.stack([tangent, bitangent, normals], dim=-1)  # [N, 3, 3]
    
    # Primitive index (0 for valid hits, -1 for invalid)
    idx = torch.zeros(N, dtype=torch.long, device=device)
    idx[~valid] = -1
    
    return positions, normals, uv, dp_du, dp_dv, idx, valid, TBN

def ray_intersect_with_tbn(scene, xs, ds):
    xs_mi = mitsuba.Point3f(xs[...,0], xs[...,1], xs[...,2])
    ds_mi = mitsuba.Vector3f(ds[...,0], ds[...,1], ds[...,2])
    rays_mi = mitsuba.Ray3f(xs_mi, ds_mi)

    ret = scene.ray_intersect_preliminary(rays_mi)
    idx = mitsuba.Int(ret.prim_index).torch().long()
    ret = ret.compute_surface_interaction(rays_mi)

    positions = ret.p.torch()
    normals   = NF.normalize(ret.n.torch(), dim=-1)

    tangent   = NF.normalize(ret.sh_frame.s.torch(), dim=-1)
    bitangent = NF.normalize(ret.sh_frame.t.torch(), dim=-1)
    TBN = torch.stack([tangent, bitangent, normals], dim=-1)  # [B, 3, 3]

    ts  = ret.t.torch()
    valid = (~ts.isinf())
    idx[~valid] = -1
    normals = double_sided(-ds, normals)

    return positions, normals, ret.uv.torch(), ret.dp_du.torch(), ret.dp_dv.torch(), idx, valid, TBN

def cal_pixel_derivative(rays_o, rays_d, p, dp_du, dp_dv, dx_du, dy_dv, normal, eps=1e-8):
    """
    Calculate UV partial derivatives with respect to pixel coordinates.
    
    Args:
        rays_o: (N,3) ray origins
        rays_d: (N,3) ray directions 
        p: (N,3) intersection points
        dp_du, dp_dv: (N,3) surface partials from mitsuba (∂p/∂u, ∂p/∂v)
        dx_du, dy_dv: (N,3) ray direction differentials wrt screen x,y
        normal: (N,3) surface normals from mitsuba (already normalized)
        eps: small epsilon to avoid division by zero
        
    Returns:
        du_dx, du_dy: (N,) UV partial derivatives wrt pixel x coordinate
        dv_dx, dv_dy: (N,) UV partial derivatives wrt pixel y coordinate
    """
    # Distance t to hit
    d = rays_d
    o = rays_o
    t = ((p - o) * d).sum(-1) / (d * d).sum(-1).clamp_min(eps)  # (N,)
    
    # Use the normal directly from mitsuba (already normalized)
    n = normal  # (N,3)
    
    ndotd = (n * d).sum(-1).clamp(min=-1.0, max=1.0)  # (N,)
    # Debug: Check if ndotd is zero
    if (ndotd.abs() ==0 ).any():
        print(f"Warning: ndotd near zero detected. Count: {(ndotd.abs() < eps).sum().item()}")
    # Avoid near-grazing division
    ndotd = torch.where(ndotd.abs() < eps, ndotd.sign() * eps, ndotd)
    
    def compute_dp_dscreen(dd):
        """Compute dp/dx or dp/dy where dd is dx_du or dy_dv"""
        ndotdd = (n * dd).sum(-1)                # (N,)
        dt = - t * ndotdd / ndotd                # (N,)
        dp = t[..., None] * dd + dt[..., None] * d  # (N,3)
        # Remove normal component for numerical stability
        #dp = dp - (dp * n).sum(-1, keepdim=True) * n
        # Debug: Check for NaN values in dp
        if torch.isinf(dp).any():
            print(f"Warning: Inf values detected in dp. Count: {torch.isinf(dp).sum().item()}")
        return dp
    
    # Calculate dp/dx and dp/dy (3D surface position derivatives)
    dp_dx = compute_dp_dscreen(dx_du)  # (N,3)
    dp_dy = compute_dp_dscreen(dy_dv)  # (N,3)
    '''
    print("rays_d",rays_d)
    print("n",n)
    print("t",t)
    print("dx_du",dx_du)
    print("dy_dv",dy_dv)
    print("dp_du",dp_du)
    print("dp_dv",dp_dv)
    print("dp_dx",dp_dx)
    print("dp_dy",dp_dy)
    '''
    # Now solve for du/dx, dv/dx, du/dy, dv/dy using the chain rule:
    # dp/dx = (∂p/∂u)(du/dx) + (∂p/∂v)(dv/dx)
    # dp/dy = (∂p/∂u)(du/dy) + (∂p/∂v)(dv/dy)
    
    # Stack the surface partials into a matrix: J = [dp_du, dp_dv] shape (N,3,2)
    J = torch.stack([dp_du, dp_dv], dim=-1)  # (N,3,2)
    
    # Analytical solution using 2x2 matrix inversion
    # We need to solve: J^T @ J @ [du/dx, dv/dx]^T = J^T @ dp/dx
    # where J^T @ J is a 2x2 matrix that we can invert analytically
    
    # Compute Gram matrix: G = J^T @ J  (N,2,2)
    JT = J.transpose(-1, -2)  # (N,2,3)
    G = torch.bmm(JT, J)  # (N,2,2)
    
    # Compute right-hand side: rhs_x = J^T @ dp/dx, rhs_y = J^T @ dp/dy
    rhs_x = torch.bmm(JT, dp_dx.unsqueeze(-1)).squeeze(-1)  # (N,2)
    rhs_y = torch.bmm(JT, dp_dy.unsqueeze(-1)).squeeze(-1)  # (N,2)
    
    # Analytical 2x2 matrix inversion
    # For matrix [[a,b],[c,d]], inverse is (1/det) * [[d,-b],[-c,a]]
    a = G[:, 0, 0]  # (N,)
    b = G[:, 0, 1]  # (N,)  
    c = G[:, 1, 0]  # (N,)
    d = G[:, 1, 1]  # (N,)
    
    # Compute determinant with regularization for numerical stability
    det = a * d - b * c  # (N,)
    det_reg = torch.where(det.abs() < eps, det.sign() * eps, det)  # Avoid division by zero
    
    # Compute inverse elements
    inv_det = 1.0 / det_reg  # (N,)
    G_inv_00 = d * inv_det   # (N,)
    G_inv_01 = -b * inv_det  # (N,)
    G_inv_10 = -c * inv_det  # (N,)
    G_inv_11 = a * inv_det   # (N,)
    
    # Solve for UV derivatives analytically
    # [du/dx, dv/dx]^T = G^{-1} @ rhs_x
    du_dx = G_inv_00 * rhs_x[:, 0] + G_inv_01 * rhs_x[:, 1]  # (N,)
    dv_dx = G_inv_10 * rhs_x[:, 0] + G_inv_11 * rhs_x[:, 1]  # (N,)
    
    # [du/dy, dv/dy]^T = G^{-1} @ rhs_y  
    du_dy = G_inv_00 * rhs_y[:, 0] + G_inv_01 * rhs_y[:, 1]  # (N,)
    dv_dy = G_inv_10 * rhs_y[:, 0] + G_inv_11 * rhs_y[:, 1]  # (N,)
    
    # Handle degenerate cases where determinant is too small
    invalid_mask = det.abs() < eps
    if invalid_mask.any():
        # Fallback to pseudoinverse for degenerate cases
        try:
            J_pinv = torch.linalg.pinv(J[invalid_mask])  # (M,2,3) where M = invalid_mask.sum()
            
            uv_x_fallback = (J_pinv @ dp_dx[invalid_mask].unsqueeze(-1)).squeeze(-1)  # (M,2)
            uv_y_fallback = (J_pinv @ dp_dy[invalid_mask].unsqueeze(-1)).squeeze(-1)  # (M,2)
            
            du_dx[invalid_mask] = uv_x_fallback[:, 0]
            dv_dx[invalid_mask] = uv_x_fallback[:, 1]
            du_dy[invalid_mask] = uv_y_fallback[:, 0]
            dv_dy[invalid_mask] = uv_y_fallback[:, 1]
        except:
            # Ultimate fallback: set to zero for invalid cases
            du_dx[invalid_mask] = 0.0
            dv_dx[invalid_mask] = 0.0
            du_dy[invalid_mask] = 0.0
            dv_dy[invalid_mask] = 0.0
    
    # Debug: Check for NaN values in UV derivatives
    if torch.isnan(du_dx).any() or torch.isnan(du_dy).any() or torch.isnan(dv_dx).any() or torch.isnan(dv_dy).any():
        print(f"Warning: NaN values detected in UV derivatives.")
        print(f"  du_dx NaN count: {torch.isnan(du_dx).sum().item()}")
        print(f"  du_dy NaN count: {torch.isnan(du_dy).sum().item()}")
        print(f"  dv_dx NaN count: {torch.isnan(dv_dx).sum().item()}")
        print(f"  dv_dy NaN count: {torch.isnan(dv_dy).sum().item()}")
    return du_dx, du_dy, dv_dx, dv_dy

def compute_area(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute area of triangle formed by each pair of points in A, B and the origin (0,0).

    Args:
        A: (n,2) torch tensor
        B: (n,2) torch tensor
    Returns:
        areas: (n,) tensor, absolute triangle areas
    """
    cross = A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]  # scalar cross product
    return cross.abs()

def compute_footprint(rays_o, rays_d, position, dp_du, dp_dv, dx_du, dy_dv, normal, eps=1e-8):
    du_dx, du_dy, dv_dx, dv_dy = cal_pixel_derivative(rays_o, rays_d, position, dp_du, dp_dv, dx_du, dy_dv, normal)
    # print("du_dx",torch.mean(du_dx))
    # print("du_dy",torch.mean(du_dy))
    # print("dv_dx",torch.mean(dv_dx))
    # print("dv_dy",torch.mean(dv_dy))
    uv_x=torch.stack((du_dx, dv_dx), dim=1) 
    uv_y=torch.stack((du_dy, dv_dy), dim=1) 
    
    area=compute_area(uv_x,uv_y) 
    return area

def uv_footprint_simple(ro, wi, p, n, dp_du, dp_dv, dx_du, dy_dv, eps=1e-8):
    t = ((p - ro) * wi).sum(-1, keepdim=True).clamp_min(0.0)
    cosi = (n * wi).sum(-1, keepdim=True).abs().clamp_min(eps)
    theta = 0.5 * torch.sqrt((dx_du**2).sum(-1, keepdim=True) + (dy_dv**2).sum(-1, keepdim=True))
    r_surf = (t * theta) / cosi
    ru = r_surf / dp_du.norm(dim=-1, keepdim=True).clamp_min(eps)
    rv = r_surf / dp_dv.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.sqrt(ru * rv)  # isotropic radius in UV

    
def batched_path_tracing_dynamic_emitter(scene,emitter_net,material_net,rays_o,rays_d,dx_du,dy_dv,spp, brdf_sampling, emitter_sampling, gt_params=None, latent=None):
    """ Path trace current scene
    Args:
        scene: mitsuba scene
        emitter_net: emitter object
        material_net: material object
        rays_o: BxNx3 ray origin
        rays_d: BxNx3 ray direction
        dx_du,dy_dv: BxNx3 ray differential
        spp: samples per pixel
        brdf_sampling: boolean flag for BRDF importance sampling
        emitter_sampling: boolean flag for emitter importance sampling
        gt_params: optional ground truth material parameters
        latent: optional batched latent code for material network
    Return:
        L: (B*N)x3 traced results unbatched
    """
    # flatten the rays
    # Create batch mask where each row contains the same batch index
    # For rays with shape B, N, 3, create mask with shape B, N
    batch_mask = torch.arange(len(rays_o), device=rays_o.device).view(rays_o.shape[0], 1).expand(rays_o.shape[0], rays_o.shape[1])
    # batch_mask = torch.zeros(len(rays_o), device=rays_o.device)
    rays_o = rays_o.reshape(-1,3)
    rays_d = rays_d.reshape(-1,3)
    dx_du = dx_du.reshape(-1,3)
    dy_dv = dy_dv.reshape(-1,3)
    batch_mask = batch_mask.reshape(-1)
    N = len(rays_o)
    N_lights = emitter_net.num_lights
    device = rays_o.device
    
    # sample camera ray
    # du,dv = torch.rand(2,len(rays_o),spp,1,device=device)-0.5
    # wi = NF.normalize(rays_d[:,None]+dx_du[:,None]*du+dy_dv[:,None]*dv,dim=-1).reshape(-1,3)
    wi = rays_d
    # Add mask for wi z component
    position = rays_o.repeat_interleave(spp,0)
    
    # compute first intersection
    position,normal,_, _,vis = ray_intersect(scene,position,wi)
    # position, normal, vis = ray_sphere_intersect(scene,position,wi)
    L = torch.zeros(vis.shape[0],3,device=device)
    if not vis.any():
        print("No valid intersection")
        return L.reshape(N,spp,3).mean(1), None, None
    position = position[vis]
    normal = normal[vis]
    batch_mask = batch_mask[vis]
    wo = -wi[vis]
    
    # deterministic sampling
    wi,emit_pdf, emit_position, idx = emitter_net.sample_emitter(position)
    normal = normal.repeat_interleave(emitter_net.num_lights,0)
    position = position.repeat_interleave(emitter_net.num_lights,0)
    wo = wo.repeat_interleave(emitter_net.num_lights,0)
    batch_mask = batch_mask.repeat_interleave(emitter_net.num_lights,0)
    # visibility test
    emit_weight,_,_ = emitter_net.eval_emitter(emit_position, idx)
    emit_vis = (wi*normal).sum(-1,keepdim=True) > 0 # B, 1
    
    # goemetry term (assume double sided area light)
    G = 1 / (emit_position-position).pow(2).sum(-1).clamp_min(1e-6) # B, 1
    emit_weight = emit_weight*emit_vis*G[...,None]/emit_pdf.clamp_min(1e-6)
    
    # Now, reshape and average over light dimension
    emit_brdf,_ = material_net.eval_brdf(gt_params,position, wi,wo,normal, latent, batch_mask)
    L[vis] += (emit_brdf*emit_weight).reshape(-1, N_lights,3).mean(1)
    L = L.reshape(N,spp,3).mean(1)
    ray_params = torch.cat([position, wi, wo], dim=-1)
    return L, vis, ray_params

def batched_path_tracing_preset_emitter(scene,emitter_net,material_net,rays_o,rays_d,dx_du,dy_dv, light_id, spp,  brdf_sampling, emitter_sampling, gt_params=None, latent=None):
    """ Path trace aligned with real capture
    Args:
        scene: mitsuba scene
        emitter_net: emitter object
        material_net: material object
        rays_o: BxNx3 ray origin
        rays_d: BxNx3 ray direction
        dx_du,dy_dv: BxNx3 ray differential
        spp: samples per pixel
        brdf_sampling: boolean flag for BRDF importance sampling
        emitter_sampling: boolean flag for emitter importance sampling
        gt_params: optional ground truth material parameters
        latent: optional batched latent code for material network
    Return:
        L: (B*N)x3 traced results unbatched
    """
    # flatten the rays
    # Create batch mask where each row contains the same batch index
    # For rays with shape B, N, 3, create mask with shape B, N
    batch_mask = torch.arange(len(rays_o), device=rays_o.device).view(rays_o.shape[0], 1).expand(rays_o.shape[0], rays_o.shape[1])
    # batch_mask = torch.zeros(len(rays_o), device=rays_o.device)
    rays_o = rays_o.reshape(-1,3)
    rays_d = rays_d.reshape(-1,3)
    light_id = light_id.reshape(-1)
    dx_du = dx_du.reshape(-1,3)
    dy_dv = dy_dv.reshape(-1,3)
    batch_mask = batch_mask.reshape(-1)
    N = len(rays_o)
    device = rays_o.device
    
    # sample camera ray
    du,dv = torch.rand(2,len(rays_o),spp,1,device=device)-0.5
    wi = NF.normalize(rays_d[:,None]+dx_du[:,None]*du+dy_dv[:,None]*dv,dim=-1).reshape(-1,3)
    # wi = rays_d
    # Add mask for wi z component
    position = rays_o.repeat_interleave(spp,0)
    
    # compute first intersection
    position,normal,uv, _,vis = ray_intersect(scene,position,wi)
    # position, normal, vis = ray_sphere_intersect(scene,position,wi)
    L = torch.zeros(vis.shape[0],3,device=device)
    if not vis.any():
        print("No valid intersection")
        return L.reshape(N,spp,3).mean(1), None, None
    position = position[vis]
    normal_raw=normal
    normal = normal[vis]
    uv=uv[vis]
    batch_mask = batch_mask[vis]
    wo = -wi[vis]
    light_id = light_id[vis]
    
    # deterministic sampling
    wi,emit_pdf, emit_position, idx = emitter_net.sample_emitter(position, light_id)
    # visibility test
    emit_weight,_,_ = emitter_net.eval_emitter(emit_position, idx)
    emit_vis = (wi*normal).sum(-1,keepdim=True) > 0 # B, 1
    
    # goemetry term (assume double sided area light)
    G = 1 / (emit_position-position).pow(2).sum(-1).clamp_min(1e-6) # B, 1
    emit_weight = emit_weight*emit_vis*G[...,None]/emit_pdf.clamp_min(1e-6)
    
    # Now, reshape and average over light dimension
    emit_brdf,_ = material_net.eval_brdf(None, position, wi,wo,normal,uv, latent, batch_mask)
    debug_mode = False
    if not debug_mode:
        L[vis] += (emit_brdf*emit_weight)
    else:
        L[vis] += torch.tensor([1.0, 0.0, 0.0], device=L.device).expand_as(L[vis])
    ray_params = torch.cat([position, wi, wo], dim=-1)

    emit_vis_raw = torch.zeros((normal_raw.shape[0],1),dtype=emit_vis.dtype, device=emit_vis.device) 
    emit_vis_raw[vis] = emit_vis
    #return emit_vis_raw, vis, ray_params
    #return (normal_raw+1)/2, vis, ray_params
    return L, vis, ray_params

def batched_path_tracing_dynamic_emitter(scene,emitter_net,material_net,rays_o,rays_d,dx_du,dy_dv,spp, brdf_sampling, emitter_sampling, gt_params=None, latent=None):
    """ Path trace current scene
    Args:
        scene: mitsuba scene
        emitter_net: emitter object
        material_net: material object
        rays_o: BxNx3 ray origin
        rays_d: BxNx3 ray direction
        dx_du,dy_dv: BxNx3 ray differential
        spp: samples per pixel
        brdf_sampling: boolean flag for BRDF importance sampling
        emitter_sampling: boolean flag for emitter importance sampling
        gt_params: optional ground truth material parameters
        latent: optional batched latent code for material network
    Return:
        L: (B*N)x3 traced results unbatched
    """
    # flatten the rays
    # Create batch mask where each row contains the same batch index
    # For rays with shape B, N, 3, create mask with shape B, N
    batch_mask = torch.arange(len(rays_o), device=rays_o.device).view(rays_o.shape[0], 1).expand(rays_o.shape[0], rays_o.shape[1])
    # batch_mask = torch.zeros(len(rays_o), device=rays_o.device)
    rays_o = rays_o.reshape(-1,3)
    rays_d = rays_d.reshape(-1,3)
    dx_du = dx_du.reshape(-1,3)
    dy_dv = dy_dv.reshape(-1,3)
    batch_mask = batch_mask.reshape(-1)
    N = len(rays_o)
    N_lights = emitter_net.num_lights
    device = rays_o.device
    
    # sample camera ray
    # du,dv = torch.rand(2,len(rays_o),spp,1,device=device)-0.5
    # wi = NF.normalize(rays_d[:,None]+dx_du[:,None]*du+dy_dv[:,None]*dv,dim=-1).reshape(-1,3)
    wi = rays_d
    # Add mask for wi z component
    position = rays_o.repeat_interleave(spp,0)
    
    # compute first intersection
    position,normal,_, _,vis = ray_intersect(scene,position,wi)
    # position, normal, vis = ray_sphere_intersect(scene,position,wi)
    L = torch.zeros(vis.shape[0],3,device=device)
    if not vis.any():
        print("No valid intersection")
        return L.reshape(N,spp,3).mean(1), None, None
    position = position[vis]
    normal = normal[vis]
    batch_mask = batch_mask[vis]
    wo = -wi[vis]
    
    # deterministic sampling
    wi,emit_pdf, emit_position, idx = emitter_net.sample_emitter(position)
    normal = normal.repeat_interleave(emitter_net.num_lights,0)
    position = position.repeat_interleave(emitter_net.num_lights,0)
    wo = wo.repeat_interleave(emitter_net.num_lights,0)
    batch_mask = batch_mask.repeat_interleave(emitter_net.num_lights,0)
    # visibility test
    emit_weight,_,_ = emitter_net.eval_emitter(emit_position, idx)
    emit_vis = (wi*normal).sum(-1,keepdim=True) > 0 # B, 1
    
    # goemetry term (assume double sided area light)
    G = 1 / (emit_position-position).pow(2).sum(-1).clamp_min(1e-6) # B, 1
    emit_weight = emit_weight*emit_vis*G[...,None]/emit_pdf.clamp_min(1e-6)
    
    # Now, reshape and average over light dimension
    emit_brdf,_ = material_net.eval_brdf(gt_params,position, wi,wo,normal, latent, batch_mask)
    L[vis] += (emit_brdf*emit_weight).reshape(-1, N_lights,3).mean(1)
    L = L.reshape(N,spp,3).mean(1)
    ray_params = torch.cat([position, wi, wo], dim=-1)
    return L, vis, ray_params

def batched_path_tracing_tbn_preset_emitter(scene,emitter_net,material_net,rays_o,rays_d,dx_du,dy_dv, light_id, spp,  brdf_sampling, emitter_sampling, gt_params=None, latent=None):
    """ Path trace aligned with real capture
    Args:
        scene: mitsuba scene
        emitter_net: emitter object
        material_net: material object
        rays_o: BxNx3 ray origin
        rays_d: BxNx3 ray direction
        dx_du,dy_dv: BxNx3 ray differential
        spp: samples per pixel
        brdf_sampling: boolean flag for BRDF importance sampling
        emitter_sampling: boolean flag for emitter importance sampling
        gt_params: optional ground truth material parameters
        latent: optional batched latent code for material network
    Return:
        L: (B*N)x3 traced results unbatched
    """
    # flatten the rays
    # Create batch mask where each row contains the same batch index
    # For rays with shape B, N, 3, create mask with shape B, N
    batch_mask = torch.arange(len(rays_o), device=rays_o.device).view(rays_o.shape[0], 1).expand(rays_o.shape[0], rays_o.shape[1])
    # batch_mask = torch.zeros(len(rays_o), device=rays_o.device)
    rays_o = rays_o.reshape(-1,3)
    rays_d = rays_d.reshape(-1,3)
    light_id = light_id.reshape(-1)
    dx_du = dx_du.reshape(-1,3)
    dy_dv = dy_dv.reshape(-1,3)
    batch_mask = batch_mask.reshape(-1)
    N = len(rays_o)
    device = rays_o.device
    
    # sample camera ray
    du,dv = torch.rand(2,len(rays_o),spp,1,device=device)-0.5
    wi = NF.normalize(rays_d[:,None]+dx_du[:,None]*du+dy_dv[:,None]*dv,dim=-1).reshape(-1,3)
    # wi = rays_d
    # Add mask for wi z component
    position = rays_o.repeat_interleave(spp,0)
    
    # compute first intersection
    position,normal,uv, _,vis, TBN = ray_intersect_with_tbn(scene,position,wi)
    # position, normal, vis = ray_sphere_intersect(scene,position,wi)
    L = torch.zeros(vis.shape[0],3,device=device)
    if not vis.any():
        print("No valid intersection")
        return L.reshape(N,spp,3).mean(1), None, None
    position = position[vis]
    normal_raw=normal
    normal = normal[vis]
    uv=uv[vis]
    batch_mask = batch_mask[vis]
    wo = -wi[vis]
    light_id = light_id[vis]
    TBN = TBN[vis]
    
    # deterministic sampling
    wi,emit_pdf, emit_position, idx = emitter_net.sample_emitter(position, light_id)
    # visibility test
    emit_weight,_,_ = emitter_net.eval_emitter(emit_position, idx)
    emit_vis = (wi*normal).sum(-1,keepdim=True) > 0 # B, 1
    
    # goemetry term (assume double sided area light)
    G = 1 / (emit_position-position).pow(2).sum(-1).clamp_min(1e-6) # B, 1
    emit_weight = emit_weight*emit_vis*G[...,None]/emit_pdf.clamp_min(1e-6)
    
    # Now, reshape and average over light dimension
    emit_brdf,_ = material_net.eval_brdf(None, position, wi,wo,normal,uv, TBN, latent, batch_mask)
    debug_mode = False
    if not debug_mode:
        L[vis] += (emit_brdf*emit_weight)
    else:
        L[vis] += torch.tensor([1.0, 0.0, 0.0], device=L.device).expand_as(L[vis])
    ray_params = torch.cat([position, wi, wo], dim=-1)

    emit_vis_raw = torch.zeros((normal_raw.shape[0],1),dtype=emit_vis.dtype, device=emit_vis.device) 
    emit_vis_raw[vis] = emit_vis
    #return emit_vis_raw, vis, ray_params
    #return (normal_raw+1)/2, vis, ray_params
    return L, vis, ray_params

def path_tracing_envmap_emitter(scene,emitter_net,material_net,rays_o,rays_d,dx_du,dy_dv,spp, brdf_sampling, emitter_sampling, gt_params=None, latent=None):
    """ Path trace current scene
    Args:
        scene: mitsuba scene
        emitter_net: emitter object
        material_net: material object
        rays_o: Bx3 ray origin
        rays_d: Bx3 ray direction
        dx_du,dy_dv: Bx3 ray differential
        spp: sampler per pixel
        indir_depth: indirect illumination depth
    Return:
        L: Bx3 traced results
    """
    B = len(rays_o)
    device = rays_o.device
    
    # sample camera ray
    # Set fixed random seed for reproducibility
    # torch.manual_seed(42)    
    # Generate random offsets for ray sampling
    du,dv = torch.rand(2,len(rays_o),spp,1,device=device)-0.5
    wi = NF.normalize(rays_d[:,None]+dx_du[:,None]*du+dy_dv[:,None]*dv,dim=-1).reshape(-1,3)
    
    # Add mask for wi z component
    position = rays_o.repeat_interleave(spp,0)
    
    # compute first intersection
    position,normal,_, _,vis = ray_intersect(scene,position,wi)
    L,_,valid_next = emitter_net.eval_emitter(position,wi)
    L[vis] = 0 # set the radiance to 0 for the valid intersection
    valid_next = vis
    # drop invalid intersection
    if not valid_next.any():
        return L.reshape(B,spp,3).mean(1)
    position = position[valid_next]
    normal = normal[valid_next]
    wo = -wi[valid_next]
    active_next = valid_next.clone()
    # Create batch mask for all samples (all batch 0s since this is not batched)
    batch_size = position.shape[0]
    batch_mask = torch.zeros(batch_size, dtype=torch.long, device=position.device)

    # Sample the environment map instead of a point emitter
    if emitter_sampling:
        wi, emit_pdf, _ = emitter_net.sample_emitter(torch.rand_like(position[..., :2]), position)

        # Evaluate the environment map along sampled directions
        emit_weight, emit_pdf, _ = emitter_net.eval_emitter(position, wi)
        emit_weight = emit_weight / emit_pdf.clamp_min(1e-6)
        # emit brdf
        emit_brdf,brdf_pdf = material_net.eval_brdf(gt_params,position, wi,wo,normal,latent, batch_mask) # gt_params will not be used in neural brdf model
        w_mis = torch.where((emit_pdf>0)&(~brdf_pdf.isinf()),emit_pdf*emit_pdf/(emit_pdf*emit_pdf+brdf_pdf*brdf_pdf),0)
        w_mis[emit_pdf.isinf()|(brdf_pdf==0)] = 1
        L[active_next] += emit_brdf*emit_weight

    # sample brdf
    if brdf_sampling:
        wi,brdf_pdf,brdf_weight = material_net.sample_brdf(
            gt_params,
            position,
            torch.rand(len(normal),device=device),
            torch.rand(len(normal),2,device=device),
            wo,normal,
            latent,
            batch_mask
        ) # ground truth roughness will be used in brdf sampling
    
        # Evaluate Le from environment map
        Le, emit_pdf, valid_next = emitter_net.eval_emitter(position, wi)

        # Update BRDF PDF
        brdf_pdf = brdf_pdf 
        
        w_mis = torch.where((brdf_pdf>0)&(~emit_pdf.isinf()),brdf_pdf*brdf_pdf/(emit_pdf*emit_pdf+brdf_pdf*brdf_pdf),0)
        w_mis[brdf_pdf.isinf()|(emit_pdf==0)] = 1
        w_mis[w_mis.isnan()] = 0
        L[active_next] += brdf_weight*Le * w_mis

    L = L.reshape(B,spp,3).mean(1)
    return L


def batched_path_tracing_tbn_real_area_emitter(scene,emitter_net,material_net,rays_o,rays_d,dx_du,dy_dv, light_id, spp, brdf_sampling, emitter_sampling, gt_params=None, latent=None):
    """ Path trace with real capture
    Args:
        scene: mitsuba scene
        emitter_net: emitter object
        material_net: material object
        rays_o: BxNx3 ray origin
        rays_d: BxNx3 ray direction
        dx_du,dy_dv: BxNx3 ray differential
        spp: samples per pixel
        brdf_sampling: boolean flag for BRDF importance sampling
        emitter_sampling: boolean flag for emitter importance sampling
        gt_params: optional ground truth material parameters
        latent: optional batched latent code for material network
    Return:
        L: (B*N)x3 traced results unbatched
    """
    is_graypatch = material_net.__class__.__name__ == 'GreyPatchBRDF'
    # flatten the rays
    # Create batch mask where each row contains the same batch index
    # For rays with shape B, N, 3, create mask with shape B, N
    batch_mask = torch.arange(len(rays_o), device=rays_o.device).view(rays_o.shape[0], 1).expand(rays_o.shape[0], rays_o.shape[1])
    # batch_mask = torch.zeros(len(rays_o), device=rays_o.device)
    rays_o = rays_o.reshape(-1,3)
    rays_d = rays_d.reshape(-1,3)
    dx_du = dx_du.reshape(-1,3)
    dy_dv = dy_dv.reshape(-1,3)
    light_id = light_id.reshape(-1)
    batch_mask = batch_mask.reshape(-1)
    N = len(rays_o)
    device = rays_o.device
    
    # sample camera ray
    du,dv = torch.rand(2,len(rays_o),spp,1,device=device)-0.5
    rays_d = (rays_d[:,None]+dx_du[:,None]*du+dy_dv[:,None]*dv).reshape(-1,3)
    wi = NF.normalize(rays_d,dim=-1)
    
    # wi = rays_d.repeat_interleave(spp, 0)
    # Add mask for wi z component
    position = rays_o.repeat_interleave(spp,0)
    light_id = light_id.repeat_interleave(spp,0)
    
    # compute first intersection
    # Check if scene is a dictionary (scene parameters) or a Mitsuba scene object
    if isinstance(scene, dict):
        if 'radius' in scene:
            # Use mathematical ray-hemisphere intersection
            position, normal, uv, dp_du, dp_dv, _, vis, TBN = ray_hemisphere_intersect_TBN(
                position, wi, 
                center=scene.get('center', [0, 0, 0]),
                radius=scene.get('radius', 1.0)
            )
        else:
            # Use mathematical ray-rectangle intersection
            position, normal, uv, dp_du, dp_dv, _, vis, TBN = ray_rectangle_intersect_TBN(
                position, wi, 
                center=scene.get('center', [0, 0, 0]),
                width=scene.get('width', 0.4),
                length=scene.get('length', 0.4)
            )
    else:
        # Use Mitsuba scene intersection
        position, normal, uv, dp_du, dp_dv, _, vis, TBN = ray_intersect_with_tbn(scene, position, wi)
    footprint_vis = compute_footprint(rays_o.repeat_interleave(spp,0)[vis], rays_d[vis], position[vis], dp_du[vis], dp_dv[vis], dx_du.repeat_interleave(spp,0)[vis], dy_dv.repeat_interleave(spp,0)[vis], normal[vis])
    # Debug: Check for NaN values in footprint_vis
    if torch.isnan(footprint_vis).any():
        print(f"Warning: NaN values detected in footprint_vis. Count: {torch.isnan(footprint_vis).sum().item()}")
    
    # position, normal, vis = ray_sphere_intersect(scene,position,wi)
    L = torch.zeros(vis.shape[0],3,device=device)
    if not vis.any():
        print("No valid intersection")
        return L.reshape(N,spp,3).mean(1), None, None
    batch_size = position.shape[0]
    batch_mask = torch.zeros(batch_size, dtype=torch.long, device=position.device)
    batch_mask = batch_mask[vis]
    position = position[vis]

    normal = normal[vis]
    uv=uv[vis]
    dp_du = dp_du[vis]
    dp_dv = dp_dv[vis]

    wo = -wi[vis]
    light_id = light_id[vis]
    TBN = TBN[vis]
    
    if is_graypatch:
        angle_ok_all = torch.zeros(N*spp, dtype=torch.bool, device=device)
        cosine_emitter_angle_all = torch.zeros(N*spp, dtype=torch.float32, device=device)
    else:
        uv_offset = torch.zeros(vis.shape[0], 2, device=device)
    
    # deterministic sampling
    if emitter_sampling:
        wi, emit_pdf, emit_position, emitter_normal= emitter_net.sample_emitter(torch.rand_like(position[..., :2]), position, light_id)
        # visibility test
        emit_weight,emit_pdf, _ = emitter_net.eval_emitter(position, wi, light_id)
        # emit brdf
        brdf_result = material_net.eval_brdf(
            gt_params=None,
            pos=position,
            wi=wi,
            wo=wo,
            normal=normal,
            uv=uv,
            TBN=TBN,
            latent=latent,
            batch_mask=batch_mask,
            footprint_vis=footprint_vis,
            dp_du=dp_du,
            dp_dv=dp_dv
        )
        if is_graypatch:
            emit_brdf, normal, brdf_pdf, angle_ok, _ = brdf_result
            angle_ok_all[vis] = angle_ok
            cosine_emitter_angle_all[vis] = (-wi*emitter_normal).sum(-1).abs()
        else:
            emit_brdf, normal, brdf_pdf, uv_offset[vis] = brdf_result

        G = (wi*normal).sum(-1).abs() / (emit_position-position).pow(2).sum(-1).clamp_min(1e-6) # B, 1
        emit_weight = emit_weight*G[...,None]/emit_pdf.clamp_min(1e-6)
        w_mis = torch.where((emit_pdf>0)&(~brdf_pdf.isinf()),emit_pdf*emit_pdf/(emit_pdf*emit_pdf+brdf_pdf*brdf_pdf),0)
        w_mis[emit_pdf.isinf()|(brdf_pdf==0)] = 1
        # Avoid in-place indexed operation for cleaner autograd graph
        contribution = emit_brdf * emit_weight
        L_update = torch.zeros_like(L)
        L_update[vis] = contribution
        L = L + L_update
        # L[vis] += emit_weight * emit_brdf
    # sample brdf
    if brdf_sampling:
        wi,brdf_pdf,brdf_weight = material_net.sample_brdf(
            gt_params,
            position,
            torch.rand(len(normal),device=device),
            torch.rand(len(normal),2,device=device),
            wo,normal,
            latent,
            batch_mask,
            dp_du,
            dp_dv
        ) # ground truth roughness will be used in brdf sampling
    
        # Evaluate Le
        Le, emit_pdf, _ = emitter_net.eval_emitter(position, wi, light_id)
        G = (wi*normal).sum(-1).abs() * (-wi*emitter_normal).sum(-1).abs() / (emit_position-position).pow(2).sum(-1).clamp_min(1e-6) # B, 1
        Le = Le*G[...,None]/emit_pdf.clamp_min(1e-6)
        
        w_mis = torch.where((brdf_pdf>0)&(~emit_pdf.isinf()),brdf_pdf*brdf_pdf/(emit_pdf*emit_pdf+brdf_pdf*brdf_pdf),0)
        w_mis[brdf_pdf.isinf()|(emit_pdf==0)] = 1
        w_mis[w_mis.isnan()] = 0
        # Avoid in-place indexed operation for cleaner autograd graph
        contribution = brdf_weight * Le * w_mis
        L_update = torch.zeros_like(L)
        L_update[vis] = contribution
        L = L + L_update
    ray_params = torch.cat([position, wi, wo], dim=-1)
    L = L.reshape(N,spp,3).mean(1)
    vis_reshaped = vis.reshape(N, spp)
    
    if is_graypatch:
        angle_ok = angle_ok_all.reshape(N, spp)
        cosine_emitter_angle = cosine_emitter_angle_all.reshape(N, spp)
        cosine_emitter_angle = cosine_emitter_angle.mean(dim=1)
        pixel_all_ok = angle_ok.all(dim=1)
        vis = vis_reshaped.any(dim=1)
        return L, vis, ray_params, (pixel_all_ok, cosine_emitter_angle)
    else:
        uv_offset = uv_offset.reshape(N, spp, 2).mean(1)
        vis = vis_reshaped.all(dim=1)
        return L, vis, ray_params, uv_offset 

def points_path_tracing_real_area_emitter(scene,emitter_net,material_net,rays_o,rays_d, xyz, dx_du,dy_dv, light_id, material_id, point_ids, spp, brdf_sampling, emitter_sampling, gt_params=None, latent=None):
    """ Path trace with real capture
    Args:
        scene: mitsuba scene
        emitter_net: emitter object
        material_net: material object
        rays_o: BxNx3 ray origin
        rays_d: BxNx3 ray direction
        xyz: BxNx3 intersection points (already computed, always visible)
        dx_du,dy_dv: BxNx3 ray differential
        spp: samples per pixel
        brdf_sampling: boolean flag for BRDF importance sampling
        emitter_sampling: boolean flag for emitter importance sampling
        gt_params: optional ground truth material parameters
        latent: optional batched latent code for material network
    Return:
        L: (B*N)x3 traced results unbatched
    """
    is_graypatch = material_net.__class__.__name__ == 'GreyPatchBRDF'
    # flatten the inputs
    # Create batch mask where each row contains the same batch index
    # For rays with shape B, N, 3, create mask with shape B, N
    batch_mask = torch.arange(len(rays_o), device=rays_o.device).view(rays_o.shape[0], 1).expand(rays_o.shape[0], rays_o.shape[1])
    rays_o = rays_o.reshape(-1,3)
    rays_d = rays_d.reshape(-1,3)
    xyz = xyz.reshape(-1,3)
    light_id = light_id.reshape(-1)
    material_id = material_id.reshape(-1)
    point_ids = point_ids.reshape(-1)
    batch_mask = batch_mask.reshape(-1)
    N = len(rays_o)
    device = rays_o.device
    
    # Use xyz directly as position, repeat for spp samples
    # All points are visible, no intersection needed
    position = xyz.repeat_interleave(spp, 0)
    light_id = light_id.repeat_interleave(spp, 0)
    material_id = material_id.repeat_interleave(spp, 0)
    point_ids = point_ids.repeat_interleave(spp, 0 )
    batch_mask = batch_mask.repeat_interleave(spp, 0)
    
    # Compute wo from rays_d (viewing direction is opposite of ray direction)
    wo = -NF.normalize(rays_d.repeat_interleave(spp, 0), dim=-1)
    
    # Set normal, TBN, uv, derivatives to None
    normal = None
    TBN = None
    uv = None
    dp_du = None
    dp_dv = None
    footprint_vis = None
    
    # Initialize output
    L = torch.zeros(N * spp, 3, device=device)
    
    if is_graypatch:
        angle_ok_all = torch.zeros(N*spp, dtype=torch.bool, device=device)
        cosine_emitter_angle_all = torch.zeros(N*spp, dtype=torch.float32, device=device)
    else:
        uv_offset = torch.zeros(N * spp, 2, device=device)
    
    # emitter sampling with randomness for spp variation
    if emitter_sampling:
        wi, emit_pdf, emit_position, emitter_normal = emitter_net.sample_emitter(torch.rand_like(position[..., :2]), position, light_id, material_id)
        # visibility test
        emit_weight, emit_pdf, _ = emitter_net.eval_emitter(position, wi, light_id, material_id)

        # emit brdf
        brdf_result = material_net.eval_brdf(
            pos=position,
            wi=wi,
            wo=wo,
            normal=normal,
            latent=latent,
            point_ids=point_ids,
            material_ids=material_id,
        )
        if is_graypatch:
            emit_brdf, brdf_pdf, angle_ok, _ = brdf_result
            angle_ok_all = angle_ok
            cosine_emitter_angle_all = (-wi * emitter_normal).sum(-1).abs()
        else:
            emit_brdf, normal, brdf_pdf, smooth_loss = brdf_result
        G = (wi*normal).sum(-1).abs() / (emit_position - position).pow(2).sum(-1).clamp_min(1e-6)
        emit_weight = emit_weight * G[..., None] / emit_pdf.clamp_min(1e-6)
        w_mis = torch.where((emit_pdf > 0) & (~brdf_pdf.isinf()), emit_pdf * emit_pdf / (emit_pdf * emit_pdf + brdf_pdf * brdf_pdf), 0)
        w_mis[emit_pdf.isinf() | (brdf_pdf == 0)] = 1
        # All points are visible, direct assignment
        L = emit_brdf * emit_weight
        if torch.isnan(L).any():
            print("L is nan")
    else:
        smooth_loss = torch.tensor(0.0, device=device)
    
    # brdf_sampling is not used since normal is None
    # if brdf_sampling:
    #     pass
    
    ray_params = torch.cat([position, wi, wo], dim=-1)
    L = L.reshape(N, spp, 3).mean(1)
    
    if is_graypatch:
        angle_ok = angle_ok_all.reshape(N, spp)
        cosine_emitter_angle = cosine_emitter_angle_all.reshape(N, spp)
        cosine_emitter_angle = cosine_emitter_angle.mean(dim=1)
        pixel_all_ok = angle_ok.all(dim=1)
        # All points are visible
        vis = torch.ones(N, dtype=torch.bool, device=device)
        return L, vis, ray_params, (pixel_all_ok, cosine_emitter_angle), smooth_loss
    else:
        uv_offset = uv_offset.reshape(N, spp, 2).mean(1)
        # All points are visible
        vis = torch.ones(N, dtype=torch.bool, device=device)
        return L, vis, ray_params, uv_offset, smooth_loss
    
# def batched_path_tracing_tbn_real_area_emitter(scene,emitter_net,material_net,rays_o,rays_d,dx_du,dy_dv, light_id, spp, brdf_sampling, emitter_sampling, gt_params=None, latent=None):
#     """ Path trace with real capture
#     Args:
#         scene: mitsuba scene
#         emitter_net: emitter object
#         material_net: material object
#         rays_o: BxNx3 ray origin
#         rays_d: BxNx3 ray direction
#         dx_du,dy_dv: BxNx3 ray differential
#         spp: samples per pixel
#         brdf_sampling: boolean flag for BRDF importance sampling
#         emitter_sampling: boolean flag for emitter importance sampling
#         gt_params: optional ground truth material parameters
#         latent: optional batched latent code for material network
#     Return:
#         L: (B*N)x3 traced results unbatched
#     """
#     is_graypatch = material_net.__class__.__name__ == 'GreyPatchBRDF'
#     # flatten the rays
#     # Create batch mask where each row contains the same batch index
#     # For rays with shape B, N, 3, create mask with shape B, N
#     batch_mask = torch.arange(len(rays_o), device=rays_o.device).view(rays_o.shape[0], 1).expand(rays_o.shape[0], rays_o.shape[1])
#     # batch_mask = torch.zeros(len(rays_o), device=rays_o.device)
#     rays_o = rays_o.reshape(-1,3)
#     rays_d = rays_d.reshape(-1,3)
#     dx_du = dx_du.reshape(-1,3)
#     dy_dv = dy_dv.reshape(-1,3)
#     light_id = light_id.reshape(-1)
#     batch_mask = batch_mask.reshape(-1)
#     N = len(rays_o)
#     device = rays_o.device
    
#     # sample camera ray
#     du,dv = torch.rand(2,len(rays_o),spp,1,device=device)-0.5
#     rays_d = (rays_d[:,None]+dx_du[:,None]*du+dy_dv[:,None]*dv).reshape(-1,3)
#     wi = NF.normalize(rays_d,dim=-1)
    
#     # wi = rays_d.repeat_interleave(spp, 0)
#     # Add mask for wi z component
#     position = rays_o.repeat_interleave(spp,0)
#     light_id = light_id.repeat_interleave(spp,0)
    
#     # compute first intersection
#     # Check if scene is a dictionary (scene parameters) or a Mitsuba scene object
#     if isinstance(scene, dict):
#         # Use mathematical ray-rectangle intersection
#         position, normal, uv, dp_du, dp_dv, _, vis, TBN = ray_rectangle_intersect_TBN(
#             position, wi, 
#             center=scene.get('center', [0, 0, 0]),
#             width=scene.get('width', 0.4),
#             length=scene.get('length', 0.4)
#         )
#     else:
#         # Use Mitsuba scene intersection
#         position, normal, uv, dp_du, dp_dv, _, vis, TBN = ray_intersect_with_tbn(scene, position, wi)
#     footprint_vis = compute_footprint(rays_o.repeat_interleave(spp,0)[vis], rays_d[vis], position[vis], dp_du[vis], dp_dv[vis], dx_du.repeat_interleave(spp,0)[vis], dy_dv.repeat_interleave(spp,0)[vis], normal[vis])
#     # Debug: Check for NaN values in footprint_vis
#     if torch.isnan(footprint_vis).any():
#         print(f"Warning: NaN values detected in footprint_vis. Count: {torch.isnan(footprint_vis).sum().item()}")
    
#     # position, normal, vis = ray_sphere_intersect(scene,position,wi)
#     L = torch.zeros(vis.shape[0],3,device=device)
#     if not vis.any():
#         print("No valid intersection")
#         return L.reshape(N,spp,3).mean(1), None, None
#     batch_size = position.shape[0]
#     batch_mask = torch.zeros(batch_size, dtype=torch.long, device=position.device)
#     batch_mask = batch_mask[vis]
#     position = position[vis]

#     normal = normal[vis]
#     uv=uv[vis]
#     dp_du = dp_du[vis]
#     dp_dv = dp_dv[vis]

#     wo = -wi[vis]
#     light_id = light_id[vis]
#     TBN = TBN[vis]
    
#     if is_graypatch:
#         angle_ok_all = torch.zeros(N*spp, dtype=torch.bool, device=device)
#         cosine_emitter_angle_all = torch.zeros(N*spp, dtype=torch.float32, device=device)
#     else:
#         uv_offset = torch.zeros(vis.shape[0], 2, device=device)
    
#     # deterministic sampling
#     if emitter_sampling:
#         wi, emit_pdf, emit_position, emitter_normal= emitter_net.sample_emitter(torch.rand_like(position[..., :2]), position, light_id)
#         # visibility test
#         emit_weight,emit_pdf, _ = emitter_net.eval_emitter(position, wi, light_id)
#         G = (wi*normal).sum(-1).abs()/ (emit_position-position).pow(2).sum(-1).clamp_min(1e-6) # B, 1
#         # G = (wi*normal).sum(-1).abs() * (-wi*emitter_normal).sum(-1).abs() / (emit_position-position).pow(2).sum(-1).clamp_min(1e-6) # B, 1
#         emit_weight = emit_weight*G[...,None]/emit_pdf.clamp_min(1e-6)
#         # emit brdf
#         brdf_result = material_net.eval_brdf(None, position, wi,wo,normal,uv, TBN, latent, batch_mask, footprint_vis, dp_du, dp_dv)
#         if is_graypatch:
#             emit_brdf, brdf_pdf, angle_ok, _ = brdf_result
#             angle_ok_all[vis] = angle_ok
#             cosine_emitter_angle_all[vis] = (-wi*emitter_normal).sum(-1).abs()
#         else:
#             emit_brdf, brdf_pdf, uv_offset[vis] = brdf_result
#         w_mis = torch.where((emit_pdf>0)&(~brdf_pdf.isinf()),emit_pdf*emit_pdf/(emit_pdf*emit_pdf+brdf_pdf*brdf_pdf),0)
#         w_mis[emit_pdf.isinf()|(brdf_pdf==0)] = 1
#         # Avoid in-place indexed operation for cleaner autograd graph
#         contribution = emit_brdf * emit_weight
#         L_update = torch.zeros_like(L)
#         L_update[vis] = contribution
#         L = L + L_update
#         # L[vis] += emit_weight * emit_brdf
#     # sample brdf
#     if brdf_sampling:
#         wi,brdf_pdf,brdf_weight = material_net.sample_brdf(
#             gt_params,
#             position,
#             torch.rand(len(normal),device=device),
#             torch.rand(len(normal),2,device=device),
#             wo,normal,
#             latent,
#             batch_mask,
#             dp_du,
#             dp_dv
#         ) # ground truth roughness will be used in brdf sampling
    
#         # Evaluate Le
#         Le, emit_pdf, _ = emitter_net.eval_emitter(position, wi, light_id)
#         G = (wi*normal).sum(-1).abs() * (-wi*emitter_normal).sum(-1).abs() / (emit_position-position).pow(2).sum(-1).clamp_min(1e-6) # B, 1
#         Le = Le*G[...,None]/emit_pdf.clamp_min(1e-6)
        
#         w_mis = torch.where((brdf_pdf>0)&(~emit_pdf.isinf()),brdf_pdf*brdf_pdf/(emit_pdf*emit_pdf+brdf_pdf*brdf_pdf),0)
#         w_mis[brdf_pdf.isinf()|(emit_pdf==0)] = 1
#         w_mis[w_mis.isnan()] = 0
#         # Avoid in-place indexed operation for cleaner autograd graph
#         contribution = brdf_weight * Le * w_mis
#         L_update = torch.zeros_like(L)
#         L_update[vis] = contribution
#         L = L + L_update
#     ray_params = torch.cat([position, wi, wo], dim=-1)
#     L = L.reshape(N,spp,3).mean(1)
#     vis_reshaped = vis.reshape(N, spp)
    
#     if is_graypatch:
#         angle_ok = angle_ok_all.reshape(N, spp)
#         cosine_emitter_angle = cosine_emitter_angle_all.reshape(N, spp)
#         cosine_emitter_angle = cosine_emitter_angle.mean(dim=1)
#         pixel_all_ok = angle_ok.all(dim=1)
#         vis = vis_reshaped.any(dim=1)
#         return L, vis, ray_params, (pixel_all_ok, cosine_emitter_angle)
#     else:
#         uv_offset = uv_offset.reshape(N, spp, 2).mean(1)
#         vis = vis_reshaped.all(dim=1)
#         return L, vis, ray_params, uv_offset 