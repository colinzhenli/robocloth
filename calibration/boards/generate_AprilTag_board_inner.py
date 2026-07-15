#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a dense board with AprilTag 36h11 markers around a 250x170 mm cut-out.
- KEEP: ONE inner ring of tags (next to the hole edges).
- REMOVE: All outer ring tags.
- Corner 2x2 clusters remain.
- NEW: Inner grid now fills the hole EXACTLY (no white margin). Gaps are auto-computed to use all space.
- Tags are rotated by TAG_ROTATION_DEG around their centers (default 180°).
- Labels remain upright for edge/corner tags; inner grid has no labels.
- Outputs PNG (300 DPI) and PDF.

Requires: opencv-contrib-python >= 4.7 (cv2.aruco.DICT_APRILTAG_36h11), Pillow, numpy
"""

import sys
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except Exception as e:
    print("ERROR: OpenCV is required. Install with: pip install --upgrade opencv-contrib-python", file=sys.stderr)
    raise

# =================== CONFIG (millimeters) ===================
BOARD_W_MM = 297   # A4 landscape width
BOARD_H_MM = 210   # A4 landscape height

HOLE_W_MM  = 250.0   # central cut-out width
HOLE_H_MM  = 170.0   # central cut-out height

TAG_SIZE_MM    = 16.0  # edge/inner-ring tag square size
INNER_CLEAR_MM = 6.0   # spacing: hole edge -> (inner ring) tag edge
TAG_GAP_MM     = 3.0   # spacing between adjacent tags along an edge (inner ring)

# Spacing between inner & outer rings (NO LONGER USED — outer ring removed)
RING_GAP_MM    = 3.0

# Rotate every tag around its center. 0 for default, 180 for your comparison case.
TAG_ROTATION_DEG = 180

# ---- Corner clusters (2x2) ----
ADD_CORNER_TAGS       = True
CORNER_TAG_SIZE_MM    = 16.0
CORNER_CLUSTER_ROWS   = 2
CORNER_CLUSTER_COLS   = 2
CORNER_CLUSTER_GAP_MM = 3.0   # edge-to-edge gap inside a cluster

# ---- Inner grid INSIDE the hole ----
ADD_INNER_GRID           = True
INNER_GRID_TAG_SIZE_MM   = 16.0   # tag size for inner grid
# NOTE: margins are ignored; grid fills hole exactly.
# INNER_GRID_GAP_MM is treated as a MINIMUM; actual gaps are auto-scaled to fill the space.
INNER_GRID_GAP_MM        = 1.5    # minimum desired gap; will be increased to fit exactly
INNER_GRID_MARGIN_MM     = 0.0    # force zero margin by design

DPI = 300
# ============================================================

def mm_to_px(mm, dpi=DPI):
    return int(round(mm * dpi / 25.4))

def ensure_apriltag_dict():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "OpenCV is installed but the 'aruco' module is missing. "
            "Install: pip install --upgrade opencv-contrib-python"
        )
    if not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
        raise RuntimeError(
            "Your OpenCV build does not include DICT_APRILTAG_36h11.\n"
            "Try: pip install --upgrade opencv-contrib-python (>=4.7)"
        )
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

def compute_layout():
    # Canvas px
    BW = mm_to_px(BOARD_W_MM)
    BH = mm_to_px(BOARD_H_MM)
    hole_w_px = mm_to_px(HOLE_W_MM)
    hole_h_px = mm_to_px(HOLE_H_MM)
    tag_px = mm_to_px(TAG_SIZE_MM)
    inner_px = mm_to_px(INNER_CLEAR_MM)
    gap_px = mm_to_px(TAG_GAP_MM)

    cx, cy = BW // 2, BH // 2
    x0 = cx - hole_w_px // 2
    y0 = cy - hole_h_px // 2
    x1 = x0 + hole_w_px
    y1 = y0 + hole_h_px

    # How many tags fit along each side (maximal packing based on hole span) — for the inner ring
    def max_count(length_px, tag_px, gap_px):
        return max(1, int((length_px + gap_px) // (tag_px + gap_px)))

    top_count  = max_count(hole_w_px, tag_px, gap_px)
    side_count = max_count(hole_h_px, tag_px, gap_px)

    # Centered coordinates along a span
    def centers_along(count, tag_px, gap_px):
        total_span = count * tag_px + (count - 1) * gap_px
        start = -total_span / 2 + tag_px / 2
        return [start + i * (tag_px + gap_px) for i in range(count)]

    x_offsets_top  = centers_along(top_count,  tag_px, gap_px)
    y_offsets_side = centers_along(side_count, tag_px, gap_px)

    # ---- Inner ring centerlines (adjacent to hole) ----
    top_row_y_inner    = y0 - inner_px - tag_px // 2
    bottom_row_y_inner = y1 + inner_px + tag_px // 2
    left_col_x_inner   = x0 - inner_px - tag_px // 2
    right_col_x_inner  = x1 + inner_px + tag_px // 2

    out = {
        "BOARD_W_PX": BW, "BOARD_H_PX": BH,
        "hole_rect": (x0, y0, x1, y1),
        "cx": cx, "cy": cy,
        "tag_px": tag_px,
        "top_count": top_count,
        "side_count": side_count,
        "x_offsets_top": x_offsets_top,
        "y_offsets_side": y_offsets_side,
        # inner ring
        "top_row_y_inner": top_row_y_inner,
        "bottom_row_y_inner": bottom_row_y_inner,
        "left_col_x_inner": left_col_x_inner,
        "right_col_x_inner": right_col_x_inner,
    }

    # Corner cluster geometry (relative to hole corners)
    if ADD_CORNER_TAGS:
        corner_tag_px = mm_to_px(CORNER_TAG_SIZE_MM)
        cluster_gap_px = mm_to_px(CORNER_CLUSTER_GAP_MM)
        corner_dx = inner_px - corner_tag_px // 10
        corner_dy = inner_px - corner_tag_px // 10

        bases = {
            "tl": (x0 - corner_dx, y0 - corner_dy),
            "tr": (x1 + corner_dx, y0 - corner_dy),
            "bl": (x0 - corner_dx, y1 + corner_dy),
            "br": (x1 + corner_dx, y1 + corner_dy),
        }
        signs = { "tl": (-1, -1), "tr": (+1, -1), "bl": (-1, +1), "br": (+1, +1) }

        step = corner_tag_px + cluster_gap_px  # center-to-center
        corner_clusters = {}
        for key in bases:
            bx, by = bases[key]
            sx, sy = signs[key]
            centers = []
            for r in range(CORNER_CLUSTER_ROWS):
                for c in range(CORNER_CLUSTER_COLS):
                    dx = (c + 0.5) * step * sx
                    dy = (r + 0.5) * step * sy
                    centers.append((int(round(bx + dx)), int(round(by + dy))))
            corner_clusters[key] = centers

        out["corner_tag_px"] = corner_tag_px
        out["corner_clusters"] = corner_clusters

    return out

def generate_marker(dict36h11, tag_id, size_px):
    m = cv2.aruco.generateImageMarker(dict36h11, int(tag_id), int(size_px))
    pil = Image.fromarray(m).convert("RGB")
    if TAG_ROTATION_DEG % 360 != 0:
        pil = pil.rotate(TAG_ROTATION_DEG, resample=Image.NEAREST, expand=False)
    return pil

def compute_fill_grid(inner_w_px, inner_h_px, tag_px, min_gap_px):
    """
    Compute rows, cols, and EXACT gaps so the grid fills the hole with zero margin.
    - Ensures non-negative gaps by reducing rows/cols if necessary.
    """
    # Start with the most columns/rows that can fit with the minimum gap
    def max_count_len(length_px, tag_px, gap_px):
        return max(1, int((length_px + gap_px) // (tag_px + gap_px)))

    cols = max_count_len(inner_w_px, tag_px, min_gap_px)
    rows = max_count_len(inner_h_px, tag_px, min_gap_px)

    # Make sure we can solve for a non-negative gap that fills exactly
    while cols > 1 and (inner_w_px - cols * tag_px) < 0:
        cols -= 1
    while rows > 1 and (inner_h_px - rows * tag_px) < 0:
        rows -= 1

    # Exact gaps to use all available space
    gap_x = 0.0 if cols == 1 else (inner_w_px - cols * tag_px) / (cols - 1)
    gap_y = 0.0 if rows == 1 else (inner_h_px - rows * tag_px) / (rows - 1)

    # Enforce minimum desired gap by reducing counts if needed
    # (keeps decreasing until gaps >= min_gap_px or count == 1)
    while cols > 1 and gap_x < min_gap_px:
        cols -= 1
        gap_x = 0.0 if cols == 1 else (inner_w_px - cols * tag_px) / (cols - 1)
    while rows > 1 and gap_y < min_gap_px:
        rows -= 1
        gap_y = 0.0 if rows == 1 else (inner_h_px - rows * tag_px) / (rows - 1)

    return rows, cols, gap_y, gap_x

def main():
    aruco_dict = ensure_apriltag_dict()
    L = compute_layout()

    BW, BH = L["BOARD_W_PX"], L["BOARD_H_PX"]
    x0, y0, x1, y1 = L["hole_rect"]
    tag_px = L["tag_px"]

    canvas = Image.new("RGB", (BW, BH), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=2)

    # Font (robust fallback)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", mm_to_px(3.5))
    except Exception:
        font = ImageFont.load_default()

    def paste_tag(tag_id, center, size_px):
        tag_img = generate_marker(aruco_dict, tag_id, size_px)
        w, h = tag_img.size
        cx2, cy2 = center
        x = int(round(cx2 - w / 2))
        y = int(round(cy2 - h / 2))
        canvas.paste(tag_img, (x, y))

    def paste_tag_with_label(tag_id, center, edge, size_px):
        # edge/corner tags with small label outside
        tag_img = generate_marker(aruco_dict, tag_id, size_px)
        w, h = tag_img.size
        cx2, cy2 = center
        x = int(round(cx2 - w / 2))
        y = int(round(cy2 - h / 2))
        canvas.paste(tag_img, (x, y))
        # label (drawn but hidden text to keep layout consistent; uncomment to show numbers)
        label = f"{tag_id}"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = mm_to_px(1.0)
        if edge == "top":
            lx, ly = cx2 - tw // 2, y - th - pad
        elif edge == "bottom":
            lx, ly = cx2 - tw // 2, y + h + pad
        elif edge == "left":
            lx, ly = x - tw - pad, cy2 - th // 2
        elif edge == "right":
            lx, ly = x + w + pad, cy2 - th // 2
        elif edge == "tl":
            lx, ly = x - tw - pad, y - th - pad
        elif edge == "tr":
            lx, ly = x + w + pad, y - th - pad
        elif edge == "bl":
            lx, ly = x - tw - pad, y + h + pad
        elif edge == "br":
            lx, ly = x + w + pad, y + h + pad
        else:
            lx, ly = cx2 - tw // 2, cy2 - th // 2
        # background to keep it legible if you enable text
        draw.rectangle([lx - 2, ly - 2, lx + tw + 2, ly + th + 2], fill=(255, 255, 255))
        # draw.text((lx, ly), label, fill=(0, 0, 0), font=font)  # << enable if you want labels

    cx, cy = L["cx"], L["cy"]
    tid = 1

    # ---------- INNER RING ONLY ----------
    for xo in L["x_offsets_top"]:
        paste_tag_with_label(tid, (int(round(cx + xo)), int(round(L["top_row_y_inner"]))), "top", tag_px); tid += 1
    for xo in L["x_offsets_top"]:
        paste_tag_with_label(tid, (int(round(cx + xo)), int(round(L["bottom_row_y_inner"]))), "bottom", tag_px); tid += 1
    for yo in L["y_offsets_side"]:
        paste_tag_with_label(tid, (int(round(L["left_col_x_inner"])), int(round(cy + yo))), "left", tag_px); tid += 1
    for yo in L["y_offsets_side"]:
        paste_tag_with_label(tid, (int(round(L["right_col_x_inner"])), int(round(cy + yo))), "right", tag_px); tid += 1

    # ---------- CORNER CLUSTERS (2x2) ----------
    if ADD_CORNER_TAGS:
        cpx = L["corner_tag_px"]
        for edge in ["tl", "tr", "bl", "br"]:
            for center in L["corner_clusters"][edge]:
                paste_tag_with_label(tid, center, edge, cpx); tid += 1

    # ---------- INNER GRID INSIDE THE HOLE (fills exactly) ----------
    if ADD_INNER_GRID:
        ig_tag_px = mm_to_px(INNER_GRID_TAG_SIZE_MM)
        min_gap_px = mm_to_px(INNER_GRID_GAP_MM)
        # zero margin by design
        inner_w = (x1 - x0)
        inner_h = (y1 - y0)

        rows, cols, gap_y, gap_x = compute_fill_grid(inner_w, inner_h, ig_tag_px, min_gap_px)

        # Top-left placement so the grid exactly spans [x0, x1] × [y0, y1]
        start_x = x0
        start_y = y0

        for r in range(rows):
            for c in range(cols):
                cxg = start_x + ig_tag_px/2 + c * (ig_tag_px + gap_x)
                cyg = start_y + ig_tag_px/2 + r * (ig_tag_px + gap_y)
                paste_tag(tid, (int(round(cxg)), int(round(cyg))), ig_tag_px)
                tid += 1

    # Header text (small)
    try:
        info_font = ImageFont.truetype("DejaVuSans.ttf", mm_to_px(4))
    except Exception:
        info_font = ImageFont.load_default()
    info_text = (f"AprilTag 36h11 - inner ring {TAG_SIZE_MM}mm, inner grid {INNER_GRID_TAG_SIZE_MM}mm "
                 f"(fills hole; min gap {INNER_GRID_GAP_MM}mm) - rotation {TAG_ROTATION_DEG}°")
    draw.text((10, 20), info_text, fill=(0, 0, 0), font=info_font)

    # Save
    png_path = "apriltag_board_inner_ring_filled_grid_rot{}.png".format(TAG_ROTATION_DEG)
    pdf_path = "apriltag_board_inner_ring_filled_grid_rot{}.pdf".format(TAG_ROTATION_DEG)
    canvas.save(png_path, "PNG", dpi=(DPI, DPI))
    canvas.save(pdf_path, "PDF", resolution=DPI)
    print("Saved:", png_path, "and", pdf_path)

if __name__ == "__main__":
    main()
