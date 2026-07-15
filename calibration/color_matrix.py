#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ColorChecker-based 3×3 color correction matrix for HDR (linear) images.

- Parses CGATS-like Lab(D50) text (post-2014 ColorChecker).
- Click 4 corners (TL->TR->BR->BL) to define board pose.
- Builds outer and inset (center) quads directly via bilinear mapping.
- Samples each patch mean (from inset quads).
- Converts Lab(D50) -> XYZ(D50) -> XYZ(D65) [Bradford] -> linear sRGB(D65).
- Solves least-squares 3x3 (camera RGB -> linear sRGB).
- Reports ΔE00.
- Applies M and saves corrected image.

Usage:
  python calibrate_cc_hdr.py \
      --lab_file ColorChecker24_After_Nov2014.txt \
      --images_folder /path/to/folder \
      --out_suffix _ccorr.png
"""

import argparse
import os
import re
from typing import Tuple, List

import cv2
import numpy as np

# =========================
# Configuration
# =========================
GRID_ROWS = 4
GRID_COLS = 6
CENTER_MARGIN = 0.2  # fraction of cell size inset on each side (0..0.49)

# Physical layout used to index CGATS values (post-2014, rotated 180° here)
PATCH_ORDER = [
    "F4","E4","D4","C4","B4","A4",
    "F3","E3","D3","C3","B3","A3",
    "F2","E2","D2","C2","B2","A2",
    "F1","E1","D1","C1","B1","A1",
]

# =========================
# Geometry helpers
# =========================
def _ensure_ccw(quad: np.ndarray) -> np.ndarray:
    # quad: (4,2) -> TL,TR,BR,BL
    x, y = quad[:, 0], quad[:, 1]
    area2 = (x[0]*y[1]+x[1]*y[2]+x[2]*y[3]+x[3]*y[0]
            -y[0]*x[1]-y[1]*x[2]-y[2]*x[3]-y[3]*x[0])
    return quad if area2 > 0 else quad[::-1].copy()

def _bilerp_quad(pts_tl_tr_br_bl: np.ndarray, u0: float, v0: float, u1: float, v1: float) -> np.ndarray:
    """
    Build a quad via bilinear mapping from canonical (u,v) in [0,1]^2 to image pixels.
    pts_tl_tr_br_bl: (4,2) in order TL,TR,BR,BL for the whole board.
    Returns quad in TL,TR,BR,BL for the subcell.
    """
    TL, TR, BR, BL = [pts_tl_tr_br_bl[i].astype(np.float64) for i in (0,1,2,3)]
    def bilerp(u, v):
        return (1-u)*(1-v)*TL + u*(1-v)*TR + u*v*BR + (1-u)*v*BL
    q = np.vstack([
        bilerp(u0, v0),  # TL
        bilerp(u1, v0),  # TR
        bilerp(u1, v1),  # BR
        bilerp(u0, v1),  # BL
    ])
    return _ensure_ccw(q)

def make_quads(pts_img: np.ndarray, rows: int, cols: int, margin: float
               ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    From board corners (TL,TR,BR,BL), build per-cell outer quads and inset-center quads.

    margin: fraction of the cell size (0..0.49). e.g., 0.2 -> keep central 60%.
    """
    outer, inner = [], []
    for r in range(rows):
        v0, v1 = r/rows, (r+1)/rows
        dv = margin * (v1 - v0)
        for c in range(cols):
            u0, u1 = c/cols, (c+1)/cols
            du = margin * (u1 - u0)

            q_outer = _bilerp_quad(pts_img, u0, v0, u1, v1)
            q_inner = _bilerp_quad(pts_img, u0+du, v0+dv, u1-du, v1-dv)
            outer.append(q_outer)
            inner.append(q_inner)
    return outer, inner

def draw_quads_on_original(
    img_linear: np.ndarray,
    quads_outer: List[np.ndarray],
    quads_inner: List[np.ndarray],
    flip_bgr: bool = False,
    pct: float = 99.5,
    outer_color: Tuple[int, int, int] = (0, 255, 0),
    inner_color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """
    Tone-map a linear HDR RGB image for visualization and draw outer/inner quads.

    Args:
        img_linear: float image in RGB (linear, HDR magnitude).
        quads_outer: list of (4,2) TL,TR,BR,BL quads for cell borders.
        quads_inner: list of (4,2) TL,TR,BR,BL quads for inset centers.
        flip_bgr: if True, return BGR (for OpenCV imwrite/show). Otherwise RGB.
        pct: percentile for simple tone-map (higher -> darker preview).
        outer_color: BGR/RGB color for outer quads (matches flip_bgr).
        inner_color: BGR/RGB color for inner quads (matches flip_bgr).
        thickness: polyline thickness in pixels.

    Returns:
        uint8 preview image with polylines drawn (BGR if flip_bgr else RGB).
    """
    img = np.asarray(img_linear, dtype=np.float64)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    denom = max(np.percentile(img, pct), 1e-12)
    prev01 = np.clip(img / denom, 0.0, 1.0)
    vis = (prev01[..., ::-1] if flip_bgr else prev01).copy()
    vis = (vis * 255.0).astype(np.uint8)

    for q in quads_outer:
        cv2.polylines(vis, [q.astype(np.int32)], isClosed=True, color=outer_color, thickness=thickness, lineType=cv2.LINE_AA)
    for q in quads_inner:
        cv2.polylines(vis, [q.astype(np.int32)], isClosed=True, color=inner_color, thickness=thickness, lineType=cv2.LINE_AA)
    return vis

# =========================
# UI / Sampling
# =========================
def click_four_points(preview_bgr: np.ndarray, window_name="ClickCorners") -> np.ndarray:
    """
    Click TL -> TR -> BR -> BL.
    """
    pts: List[Tuple[int, int]] = []
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            cv2.circle(preview_bgr, (x, y), 6, (0, 255, 0), -1)
            cv2.imshow(window_name, preview_bgr)

    cv2.imshow(window_name, preview_bgr)
    cv2.setMouseCallback(window_name, on_mouse)
    print("Click 4 corners: TL(black/F4) -> TR(white/A4) -> BR(brown/A1) -> BL(green-blue/F1). Press 'q' to cancel.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if len(pts) == 4:
            break
        if key == ord('q'):
            pts.clear()
            break
    cv2.setMouseCallback(window_name, lambda *args: None)
    cv2.destroyWindow(window_name)
    return np.asarray(pts, dtype=np.float32)

def draw_quads(img_linear: np.ndarray, quads_outer: List[np.ndarray],
               quads_inner: List[np.ndarray], flip_bgr=False) -> np.ndarray:
    denom = np.percentile(img_linear, 99.5) + 1e-12
    prev01 = np.clip(img_linear / denom, 0, 1)
    overlay = (prev01[..., ::-1]*255).astype(np.uint8) if flip_bgr else (prev01*255).astype(np.uint8)
    for q in quads_outer:
        cv2.polylines(overlay, [q.astype(np.int32)], True, (0, 255, 0), 2)
    for q in quads_inner:
        cv2.polylines(overlay, [q.astype(np.int32)], True, (0, 0, 255), 2)
    return overlay

def sample_means(img_linear_rgb: np.ndarray, quads: List[np.ndarray], rows: int, cols: int) -> np.ndarray:
    """
    Returns (rows*cols, 3) means; if a quad is empty, returns NaNs for that entry.
    """
    H, W = img_linear_rgb.shape[:2]
    means = np.full((rows*cols, 3), np.nan, dtype=np.float64)
    mask = np.zeros((H, W), dtype=np.uint8)
    for i, q in enumerate(quads):
        mask.fill(0)
        cv2.fillConvexPoly(mask, q.astype(np.int32), 255)
        m = mask.astype(bool)
        if m.any():
            means[i] = img_linear_rgb[m].mean(axis=0)
    return means

# =========================
# CGATS parser (Lab D50)
# =========================
def parse_cgats_lab_d50(txt_path: str) -> np.ndarray:
    """
    Parse CGATS-like text listing SAMPLE_NAME and Lab_L Lab_a Lab_b for patches A1..F4.
    Returns (24,3) array in PATCH_ORDER.
    """
    lab_map = {}
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        for ln in f:
            m = re.match(r'^\s*([A-F]\d)\s+(.+)', ln)
            if not m:
                continue
            name = m.group(1)
            fields = re.split(r'\t+|\s{2,}', m.group(2).strip())
            if len(fields) >= 3:
                L = float(fields[0].replace(',', '.'))
                a = float(fields[1].replace(',', '.'))
                b = float(fields[2].replace(',', '.'))
                lab_map[name] = (L, a, b)
    vals, missing = [], []
    for p in PATCH_ORDER:
        (vals.append(lab_map[p]) if p in lab_map else missing.append(p))
    if missing:
        print("WARNING: missing patches in CGATS:", missing)
    return np.asarray(vals, dtype=np.float64)

# =========================
# Color space conversions
# =========================
def lab_to_xyz_D50(Lab: np.ndarray) -> np.ndarray:
    Xn, Yn, Zn = 0.9642, 1.0000, 0.8251
    L, a, b = Lab[..., 0], Lab[..., 1], Lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)
    delta = 6/29
    def inv_f(t):
        return np.where(t > delta, t**3, 3*delta**2*(t - 4/29))
    xr = inv_f(fx); yr = inv_f(fy); zr = inv_f(fz)
    return np.stack([xr*Xn, yr*Yn, zr*Zn], axis=-1)

def xyz_to_lab_D50(XYZ_D50: np.ndarray) -> np.ndarray:
    Xn, Yn, Zn = 0.9642, 1.0, 0.8251
    xr = XYZ_D50[...,0] / Xn
    yr = XYZ_D50[...,1] / Yn
    zr = XYZ_D50[...,2] / Zn
    delta3 = (6/29)**3
    def f(t):
        return np.where(t > delta3, np.cbrt(t), (t/(3*(6/29)**2)) + (4/29))
    fx, fy, fz = f(xr), f(yr), f(zr)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return np.stack([L, a, b], axis=-1)

def bradford_D50_to_D65(XYZ_D50: np.ndarray) -> np.ndarray:
    M = np.array([[ 0.8951,  0.2664, -0.1614],
                  [-0.7502,  1.7135,  0.0367],
                  [ 0.0389, -0.0685,  1.0296]])
    M_inv = np.linalg.inv(M)
    D50 = np.array([0.9642, 1.0000, 0.8251])
    D65 = np.array([0.95047, 1.00000, 1.08883])
    rho_D50 = M @ D50
    rho_D65 = M @ D65
    D = np.diag(rho_D65 / rho_D50)
    XYZ = XYZ_D50.reshape(-1, 3).T
    XYZ_adapt = M_inv @ (D @ (M @ XYZ))
    return XYZ_adapt.T.reshape(XYZ_D50.shape)

def bradford_D50_to_illuminant(XYZ_D50: np.ndarray, target_xy: Tuple[float, float]) -> np.ndarray:
    """Bradford chromatic adaptation from D50 to an arbitrary illuminant.

    Args:
        XYZ_D50: input colors in XYZ under D50.
        target_xy: CIE 1931 (x, y) chromaticity of the target illuminant.
    """
    M = np.array([[ 0.8951,  0.2664, -0.1614],
                  [-0.7502,  1.7135,  0.0367],
                  [ 0.0389, -0.0685,  1.0296]])
    M_inv = np.linalg.inv(M)
    D50 = np.array([0.9642, 1.0000, 0.8251])
    x, y = target_xy
    target_wp = np.array([x / y, 1.0, (1.0 - x - y) / y])
    rho_D50 = M @ D50
    rho_tgt = M @ target_wp
    D = np.diag(rho_tgt / rho_D50)
    XYZ = XYZ_D50.reshape(-1, 3).T
    XYZ_adapt = M_inv @ (D @ (M @ XYZ))
    return XYZ_adapt.T.reshape(XYZ_D50.shape)

def bradford_D65_to_D50(XYZ_D65: np.ndarray) -> np.ndarray:
    M = np.array([[ 0.8951,  0.2664, -0.1614],
                  [-0.7502,  1.7135,  0.0367],
                  [ 0.0389, -0.0685,  1.0296]])
    M_inv = np.linalg.inv(M)
    D50 = np.array([0.9642, 1.0000, 0.8251])
    D65 = np.array([0.95047, 1.00000, 1.08883])
    rho_D50 = M @ D50
    rho_D65 = M @ D65
    D = np.diag(rho_D50 / rho_D65)
    XYZ = XYZ_D65.reshape(-1, 3).T
    XYZ_adapt = M_inv @ (D @ (M @ XYZ))
    return XYZ_adapt.T.reshape(XYZ_D65.shape)

def xyz_to_linear_srgb_D65(XYZ: np.ndarray) -> np.ndarray:
    M = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660,  1.8760108,  0.0415560],
                  [ 0.0556434, -0.2040259,  1.0572252]])
    return XYZ @ M.T

def linear_srgb_to_xyz_D65(rgb: np.ndarray) -> np.ndarray:
    M_inv = np.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]])
    return rgb @ M_inv.T

# =========================
# ΔE00
# =========================
def _hp_f(a, b):
    h = np.degrees(np.arctan2(b, a))
    return np.where(h < 0, h + 360, h)

def deltaE2000(Lab1: np.ndarray, Lab2: np.ndarray) -> np.ndarray:
    L1,a1,b1 = Lab1[...,0], Lab1[...,1], Lab1[...,2]
    L2,a2,b2 = Lab2[...,0], Lab2[...,1], Lab2[...,2]
    kL=kC=kH=1.0
    C1 = np.sqrt(a1*a1 + b1*b1); C2 = np.sqrt(a2*a2 + b2*b2)
    Cm = (C1 + C2)/2.0
    G = 0.5*(1 - np.sqrt((Cm**7)/(Cm**7 + 25**7)))
    a1p = (1+G)*a1; a2p = (1+G)*a2
    C1p = np.sqrt(a1p*a1p + b1*b1); C2p = np.sqrt(a2p*a2p + b2*b2)
    h1p = _hp_f(a1p,b1); h2p = _hp_f(a2p,b2)
    dLp = L2 - L1; dCp = C2p - C1p
    dhp = h2p - h1p; dhp = np.where(dhp > 180, dhp - 360, dhp); dhp = np.where(dhp < -180, dhp + 360, dhp)
    dHp = 2*np.sqrt(C1p*C2p)*np.sin(np.radians(dhp/2))
    Lpm = (L1 + L2)/2; Cpm = (C1p + C2p)/2
    hpm = (h1p + h2p)/2; hpm = np.where(np.abs(h1p - h2p) > 180, hpm + 180, hpm); hpm = np.where(hpm >= 360, hpm - 360, hpm)
    T = 1 - 0.17*np.cos(np.radians(hpm - 30)) + 0.24*np.cos(np.radians(2*hpm)) + \
        0.32*np.cos(np.radians(3*hpm + 6)) - 0.20*np.cos(np.radians(4*hpm - 63))
    Sl = 1 + (0.015*(Lpm - 50)**2)/np.sqrt(20 + (Lpm - 50)**2)
    Sc = 1 + 0.045*Cpm
    Sh = 1 + 0.015*Cpm*T
    Rt = -2*np.sqrt((Cpm**7)/(Cpm**7 + 25**7))*np.sin(np.radians(60*np.exp(-((hpm-275)/25)**2)))
    dE = np.sqrt((dLp/(kL*Sl))**2 + (dCp/(kC*Sc))**2 + (dHp/(kH*Sh))**2 + Rt*(dCp/(kC*Sc))*(dHp/(kH*Sh)))
    return dE

# =========================
# Fitting & Apply
# =========================
def fit_color_matrix(cam_rgb_24x3: np.ndarray, ref_rgb_24x3: np.ndarray) -> np.ndarray:
    """Solve cam * M ≈ ref (least squares). Returns 3x3."""
    mask = np.isfinite(cam_rgb_24x3).all(axis=1) & np.isfinite(ref_rgb_24x3).all(axis=1)
    C = cam_rgb_24x3[mask]
    R = ref_rgb_24x3[mask]
    M, _, _, _ = np.linalg.lstsq(C, R, rcond=None)
    return M

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab_file", default="ColorChecker24_After_Nov2014.txt")
    ap.add_argument("--images_folder", required=True)
    ap.add_argument("--illuminant_xy", type=float, nargs=2, default=None,
                    metavar=("X", "Y"),
                    help="CIE 1931 (x, y) of the actual illuminant. "
                         "If omitted, adapts to D65 (standard sRGB). "
                         "E.g. --illuminant_xy 0.3818 0.3797 for CMA1840 40G 4000K")
    args = ap.parse_args()

    # 1) Reference Lab(D50) → linear sRGB in 0..1
    lab_ref = parse_cgats_lab_d50(args.lab_file)
    XYZ_D50 = lab_to_xyz_D50(lab_ref)
    if args.illuminant_xy is not None:
        target_xy = tuple(args.illuminant_xy)
        print(f"Adapting to illuminant (x, y) = {target_xy}")
        XYZ_target = bradford_D50_to_illuminant(XYZ_D50, target_xy)
    else:
        print("Adapting to D65 (standard sRGB white point)")
        XYZ_target = bradford_D50_to_D65(XYZ_D50)
    ref_lin_srgb = xyz_to_linear_srgb_D65(XYZ_target)  # (24,3)

    # 2) List images
    exts = (".png", ".tif", ".tiff", ".exr")
    image_paths = sorted(
        os.path.join(args.images_folder, f)
        for f in os.listdir(args.images_folder)
        if f.lower().endswith(exts)
    )
    if not image_paths:
        print("No images found."); return

    cam_means_all_01 = []  # each (24,3) in 0..1 using fixed 65535 normalization

    # --- PASS 1: click, build quads, sample patches, save overlays ---
    for img_path in image_paths:
        print(f"\n=== Sampling patches: {img_path} ===")
        img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            print("ERROR reading", img_path); continue
        if img_bgr.ndim == 2: img_bgr = np.stack([img_bgr]*3, -1)
        if img_bgr.shape[2] == 4: img_bgr = img_bgr[..., :3]
        img_rgb = cv2.cvtColor(img_bgr.astype(np.float32), cv2.COLOR_BGR2RGB).astype(np.float64)

        # preview for clicking
        prev = np.clip(img_rgb / (np.percentile(img_rgb, 99.5) + 1e-12), 0, 1)
        prev_bgr = (prev[..., ::-1] * 255).astype(np.uint8)
        try:
            cv2.namedWindow("ClickCorners", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ClickCorners", 1200, 800)
        except cv2.error:
            pass
        pts_img = click_four_points(prev_bgr, "ClickCorners")
        if pts_img.shape != (4,2):
            print("Cancelled/invalid clicks; skip"); continue

        # build quads and sample
        outer_quads, inner_quads = make_quads(pts_img, GRID_ROWS, GRID_COLS, CENTER_MARGIN)
        cam_means_counts = sample_means(img_rgb, inner_quads, GRID_ROWS, GRID_COLS)  # (24,3), ~0..65535

        # save overlay image
        overlay = draw_quads_on_original(img_rgb, outer_quads, inner_quads, flip_bgr=True)  # uint8 BGR
        quad_path = os.path.splitext(img_path)[0] + "_overlay.png"
        cv2.imwrite(quad_path, overlay)
        print("Saved quads overlay:", quad_path)

        # fixed normalization by 65535 → 0..1 domain for fitting
        cam_means_all_01.append(cam_means_counts / 65535.0)

    if not cam_means_all_01:
        print("No valid patches; abort."); return

    # --- PASS 2: fit global CCM in 0..1 domain (no scale in M) ---
    cam_means_avg_01 = np.mean(np.stack(cam_means_all_01, axis=0), axis=0)  # (24,3), 0..1
    M_fit = fit_color_matrix(cam_means_avg_01, ref_lin_srgb)
    print("\nGlobal CCM (fit in 0..1 domain, no extra scaling):\n", M_fit)

    # QC: ΔE in the same 0..1 fitting domain
    corr_lin_fit = cam_means_avg_01 @ M_fit
    dE = deltaE2000(
        xyz_to_lab_D50(bradford_D65_to_D50(linear_srgb_to_xyz_D65(corr_lin_fit))),
        lab_ref
    )
    print("ΔE00 (avg patches, fit domain) — mean {:.3f}, median {:.3f}, max {:.3f}"
          .format(np.nanmean(dE), np.nanmedian(dE), np.nanmax(dE)))

    # --- PASS 3: apply SAME M directly to HDR images; save HDR + preview ---
    for img_path in image_paths:
        print(f"\n=== Applying global CCM: {img_path} ===")
        img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img_bgr is None: print("ERROR reading", img_path); continue
        if img_bgr.ndim == 2: img_bgr = np.stack([img_bgr]*3, -1)
        if img_bgr.shape[2] == 4: img_bgr = img_bgr[..., :3]
        img_rgb = cv2.cvtColor(img_bgr.astype(np.float32), cv2.COLOR_BGR2RGB).astype(np.float64)
        # --- Apply global CCM and save 16-bit PNG + preview ---
        img_corr = (img_rgb @ M_fit).astype(np.float32)  # HDR float

        # normalize range to [0, 65535] safely for 16-bit PNG
        img_corr_16 = np.clip(img_corr, 0, 65535).astype(np.uint16)

        # save 16-bit PNG
        png16_path = os.path.splitext(img_path)[0] + "_ccorr_u16.png"
        cv2.imwrite(png16_path, img_corr_16[..., ::-1])  # OpenCV expects BGR order

        # also save a tone-mapped preview (for viewing)
        p = max(np.percentile(img_corr, 99.5), 1e-12)
        prev = np.clip(img_corr / p, 0, 1)
        prev_path = os.path.splitext(img_path)[0] + "_ccorr_preview.png"
        cv2.imwrite(prev_path, (prev[..., ::-1] * 255).astype(np.uint8))

        print("Saved:", png16_path, "|", prev_path)

if __name__ == "__main__":
    main()
