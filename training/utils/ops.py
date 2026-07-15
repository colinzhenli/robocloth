import torch
import torch.nn.functional as NF
import math


def get_normal_space(normal):
    """ get matrix transform shading space to normal spanned space
    Args:
        normal: Bx3
    Return:
        Bx3x3 transformation matrix
    """
    v1 = torch.zeros_like(normal)
    tangent = v1.clone()
    v1[...,0] = 1.0
    tangent[...,1] = 1.0
    
    mask = (v1*normal).sum(-1).abs() <= 1e-1
    tangent[mask] = NF.normalize(torch.cross(v1[mask],normal[mask],dim=-1),dim=-1)
    mask = ~mask
    tangent[mask] = NF.normalize(torch.cross(tangent[mask],normal[mask],dim=-1),dim=-1)
    
    bitangent = torch.cross(normal,tangent,dim=-1)
    return torch.stack([tangent,bitangent,normal],dim=-1)

def angle2xyz(theta,phi):
    """ spherical coordinates to euclidean 
    Args:
        theta,phi: B
    Return:
        Bx3 euclidean coordinates
    """
    sin_theta = torch.sin(theta)
    x = sin_theta*torch.cos(phi)
    y = sin_theta*torch.sin(phi)
    z = torch.cos(theta)
    ret = torch.stack([x,y,z],dim=-1)
    return NF.normalize(ret,dim=-1)

def G1_GGX_Schlick(NoV, eta):
    """ G term of schlick GGX
    eta: roughness
    """
    r = eta
    k = (r+1)
    k = k*k/8
    denom = NoV*(1-k)+k
    return 1 /(denom + 1e-6)

def G_Smith(NoV,NoL,eta):
    """ Smith shadow masking divided by (NoV*NoL)
    eta: roughness 
    """
    g1_l = G1_GGX_Schlick(NoL,eta)
    g1_v = G1_GGX_Schlick(NoV,eta)
    return g1_l*g1_v

def fresnelSchlick(VoH,F0):
    """ schlick fresnel """
    x = (1-VoH).pow(5)
    return F0 + (1-F0)*x

def fresnelSchlick_sep(VoH):
    """ two terms of schlick fresnel """
    x = (1-VoH).pow(5)
    return (1-x),x

def D_GGX(cos_h,eta):
    """GGX normal distribution
    eta: roughness
    """
    alpha = eta*eta
    alpha2 = alpha*alpha
    denom = (cos_h*cos_h*(alpha2-1.0)+1.0)
    denom = math.pi * denom*denom
    return alpha2/(denom + 1e-6)

def D_GGX_aniso(h, N, T, B, ax, ay):
    Ht = (h * T).sum(-1, keepdim=True)
    Hb = (h * B).sum(-1, keepdim=True)
    Hn = (h * N).sum(-1, keepdim=True)
    denom = (Ht**2)/(ax**2) + (Hb**2)/(ay**2) + Hn**2      # ( .. )² in paper
    return 1.0 / (math.pi * ax * ay * denom**2 + 1e-6)

# ----------  Geometry terms --------------------------------------------------
def G1_aniso(v, N, T, B, ax, ay):
    Vn = (v * N).sum(-1, keepdim=True).clamp(1e-6)
    Vt = (v * T).sum(-1, keepdim=True)
    Vb = (v * B).sum(-1, keepdim=True)
    lam = torch.sqrt(ax**2 * Vt**2 + ay**2 * Vb**2 + Vn**2) / Vn - 1.0
    return 1.0 / (1.0 + lam)                                # Smith masking

def G_Smith_aniso(wi, wo, N, T, B, ax, ay):
    return G1_aniso(wi, N, T, B, ax, ay) * G1_aniso(wo, N, T, B, ax, ay)


def double_sided(V,N):
    """ double sided normal 
    Args:
        V: Bx3 viewing direction
        N: Bx3 normal direction
    Return:
        Bx3 flipped normal towards camera direction
    """
    NoV = (N*V).sum(-1)
    flipped = NoV<0
    tmp = -N[flipped]
    N[flipped] = tmp
    return N

    
def lerp_specular(specular,roughness):
    """ interpolate specular shadings by roughness
    Args:
        specular: Bx6x3 specular shadings
        roughness: Bx1 roughness in [0.02,1.0]
    Return:
        Bx3 interpolated specular shading
    """
    # remap roughness from to [0,1]
    r_min,r_max = 0.02,1.0 
    r_num = specular.shape[-2]
    r = (roughness-r_min)/(r_max-r_min)*(r_num-1)
    
    
    r1 = r.ceil().long()
    r0 = r.floor().long()
    r_ = (r-r0)
    s0 = torch.gather(specular,1,r0[...,None].expand(r0.shape[0],1,3))[:,0]
    s1 = torch.gather(specular,1,r1[...,None].expand(r1.shape[0],1,3))[:,0]
    s = s0*(1-r_) + s1*r_
    return s

MAX_SH_DEGREE = 6

def components_from_spherical_harmonics(
    degree: int, directions: torch.Tensor
) -> torch.Tensor:
    """
    Returns value for each component of spherical harmonics.

    Args:
        degree: Number of spherical harmonic degrees to compute.
        directions: Spherical harmonic coefficients
    """
    num_components = num_sh_bases(degree)
    components = torch.zeros((*directions.shape[:-1], num_components), device=directions.device)

    assert 0 <= degree <= MAX_SH_DEGREE, f"SH degree must be in [0, {MAX_SH_DEGREE}], got {degree}"
    assert directions.shape[-1] == 3, f"Direction input should have three dimensions. Got {directions.shape[-1]}"

    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]

    xx = x**2
    yy = y**2
    zz = z**2

    # l0
    components[..., 0] = 0.28209479177387814

    # l1
    if degree > 0:
        components[..., 1] = 0.4886025119029199 * y
        components[..., 2] = 0.4886025119029199 * z
        components[..., 3] = 0.4886025119029199 * x

    # l2
    if degree > 1:
        components[..., 4] = 1.0925484305920792 * x * y
        components[..., 5] = 1.0925484305920792 * y * z
        components[..., 6] = 0.9461746957575601 * zz - 0.31539156525251999
        components[..., 7] = 1.0925484305920792 * x * z
        components[..., 8] = 0.5462742152960396 * (xx - yy)

    # l3
    if degree > 2:
        components[..., 9] = 0.5900435899266435 * y * (3 * xx - yy)
        components[..., 10] = 2.890611442640554 * x * y * z
        components[..., 11] = 0.4570457994644658 * y * (5 * zz - 1)
        components[..., 12] = 0.3731763325901154 * z * (5 * zz - 3)
        components[..., 13] = 0.4570457994644658 * x * (5 * zz - 1)
        components[..., 14] = 1.445305721320277 * z * (xx - yy)
        components[..., 15] = 0.5900435899266435 * x * (xx - 3 * yy)

    # l4
    if degree > 3:
        components[..., 16] = 2.5033429417967046 * x * y * (xx - yy)
        components[..., 17] = 1.7701307697799304 * y * z * (3 * xx - yy)
        components[..., 18] = 0.9461746957575601 * x * y * (7 * zz - 1)
        components[..., 19] = 0.6690465435572892 * y * z * (7 * zz - 3)
        components[..., 20] = 0.10578554691520431 * (35 * zz * zz - 30 * zz + 3)
        components[..., 21] = 0.6690465435572892 * x * z * (7 * zz - 3)
        components[..., 22] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1)
        components[..., 23] = 1.7701307697799304 * x * z * (xx - 3 * yy)
        components[..., 24] = 0.6258357354491761 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

    # l5
    if degree > 4:
        components[..., 25] = 0.6563820568401703 * y * (15 * xx**2 - 10 * xx * yy + 3 * yy**2)
        components[..., 26] = 1.7701307697799304 * x * y * z * (3 * xx - yy)
        components[..., 27] = 0.5291677740499537 * y * (21 * zz * xx - 7 * xx - 21 * zz * yy + 7 * yy)
        components[..., 28] = 0.4570457994644658 * y * z * (9 * zz - 1)
        components[..., 29] = 0.3731763325901154 * z * (21 * zz**2 - 14 * zz + 1)
        components[..., 30] = 0.4570457994644658 * x * z * (9 * zz - 1)
        components[..., 31] = 0.26458388702338646 * (xx - yy) * (21 * zz**2 - 14 * zz + 1)
        components[..., 32] = 1.7701307697799304 * x * z * (xx - 3 * yy)
        components[..., 33] = 0.5291677740499537 * x * (21 * zz * xx - 7 * xx - 21 * zz * yy + 7 * yy)
        components[..., 34] = 0.6563820568401703 * x * (xx**2 - 10 * xx * yy + 15 * yy**2)

    # l6
    if degree > 5:
        components[..., 35] = 1.3663682103838286 * x * y * (5 * xx**2 - 10 * xx * yy + yy**2)
        components[..., 36] = 2.366619162231752 * y * z * (5 * xx**2 - 10 * xx * yy + yy**2)
        components[..., 37] = 0.5268565240848685 * x * y * (33 * zz * xx - 11 * xx - 33 * zz * yy + 11 * yy)
        components[..., 38] = 0.5349652792439522 * y * z * (33 * zz * xx - 11 * xx - 33 * zz * yy + 11 * yy)
        components[..., 39] = 0.4641322034408583 * x * y * (33 * zz**2 - 18 * zz + 1)
        components[..., 40] = 0.6690465435572892 * y * z * (11 * zz**2 - 3 * zz)
        components[..., 41] = 0.10578554691520431 * z * (231 * zz**3 - 315 * zz**2 + 105 * zz - 5)
        components[..., 42] = 0.6690465435572892 * x * z * (11 * zz**2 - 3 * zz)
        components[..., 43] = 0.23206610172042916 * (xx - yy) * (33 * zz**2 - 18 * zz + 1)
        components[..., 44] = 0.5349652792439522 * x * z * (33 * zz * xx - 11 * xx - 33 * zz * yy + 11 * yy)
        components[..., 45] = 0.13171413102121713 * (xx**2 - 6 * xx * yy + yy**2) * (33 * zz**2 - 18 * zz + 1)
        components[..., 46] = 2.366619162231752 * x * z * (xx**2 - 10 * xx * yy + 5 * yy**2)
        components[..., 47] = 0.6831841051919143 * (xx - yy) * (xx**2 - 10 * xx * yy + 5 * yy**2)
        components[..., 48] = 1.3663682103838286 * x * (xx**2 - 15 * xx * yy + 15 * yy**2)

    return components

def num_sh_bases(degree: int) -> int:
    """
    Returns the number of spherical harmonic bases for a given degree.
    """
    assert degree <= MAX_SH_DEGREE, f"We don't support degree greater than {MAX_SH_DEGREE}."
    return (degree + 1) ** 2


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5

def rotate_to_canonical_frame(wi, wo):
    """
    Rotate wi and wo around z-axis by the same angle to make wi.x = 0.
    
    This exploits isotropy by rotating the coordinate frame so that the incoming 
    direction lies in the yz-plane (x component = 0). The outgoing direction is 
    rotated by the same amount.
    
    Args:
        wi: [B, 3] incoming direction vectors
        wo: [B, 3] outgoing direction vectors
    
    Returns:
        wi_rotated: [B, 3] rotated incoming vectors with x component = 0
        wo_rotated: [B, 3] rotated outgoing vectors
    """
    # Calculate rotation angle: negative of wi's azimuthal angle
    # phi_wi = atan2(wi.y, wi.x)
    # We want to rotate by -phi_wi to make wi.x = 0
    phi_wi = torch.atan2(wi[:, 1], wi[:, 0])  # [B]
    
    # Rotation matrix around z-axis by angle -phi_wi:
    # [cos(-phi)  -sin(-phi)  0]   [cos(phi)   sin(phi)  0]
    # [sin(-phi)   cos(-phi)  0] = [-sin(phi)  cos(phi)  0]
    # [   0           0       1]   [   0          0      1]
    
    cos_phi = torch.cos(phi_wi)  # [B]
    sin_phi = torch.sin(phi_wi)  # [B]
    
    # Apply rotation to wi
    wi_rotated = torch.zeros_like(wi)
    wi_rotated[:, 0] = cos_phi * wi[:, 0] + sin_phi * wi[:, 1]
    wi_rotated[:, 1] = -sin_phi * wi[:, 0] + cos_phi * wi[:, 1]
    wi_rotated[:, 2] = wi[:, 2]
    
    # Apply same rotation to wo
    wo_rotated = torch.zeros_like(wo)
    wo_rotated[:, 0] = cos_phi * wo[:, 0] + sin_phi * wo[:, 1]
    wo_rotated[:, 1] = -sin_phi * wo[:, 0] + cos_phi * wo[:, 1]
    wo_rotated[:, 2] = wo[:, 2]
    
    return wi_rotated, wo_rotated