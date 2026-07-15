#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a dense board with AprilTag 36h11 markers around a 250x170 mm cut-out.
- TWO rows/columns (double lines) of tags around each edge of the hole.
- Each corner has a 2x2 cluster (four tags) placed outside the hole.
- NEW: Fills the hole with a centered grid of smaller AprilTags (for pose/axis estimation).
- Tags are rotated by TAG_ROTATION_DEG around their centers (default 180°).
- Labels remain upright for readability (edge/corner tags only; inner grid has no labels).
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
BOARD_W_MM = 297   # printing size
BOARD_H_MM = 210   # printing size

HOLE_W_MM  = 250.0   # central cut-out width
HOLE_H_MM  = 170.0   # central cut-out height

TAG_SIZE_MM    = 16.0  # edge/outer tags square size
INNER_CLEAR_MM = 6.0   # spacing: hole edge -> (inner ring) tag edge
TAG_GAP_MM     = 3.0   # spacing between adjacent tags along an edge

# Spacing between the inner and outer rings (edge-to-edge).
RING_GAP_MM    = 3.0

# Rotate every tag around its center. 0 for default, 180 for your comparison case.
TAG_ROTATION_DEG = 180

# ---- Corner clusters (2x2) ----
ADD_CORNER_TAGS       = True
CORNER_TAG_SIZE_MM    = 16.0
CORNER_CLUSTER_ROWS   = 2
CORNER_CLUSTER_COLS   = 2
CORNER_CLUSTER_GAP_MM = 3.0   # edge-to-edge gap inside a cluster

# ---- NEW: Inner grid INSIDE the hole ----
ADD_INNER_GRID           = True
INNER_GRID_TAG_SIZE_MM   = 16.0   # smaller tags inside the hole
INNER_GRID_GAP_MM        = 3.0    # gap between inner grid tags
INNER_GRID_MARGIN_MM     = 5.0    # white margin from the hole edge to nearest inner tag

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
    ring_gap_px = mm_to_px(RING_GAP_MM)

    cx, cy = BW // 2, BH // 2
    x0 = cx - hole_w_px // 2
    y0 = cy - hole_h_px // 2
    x1 = x0 + hole_w_px
    y1 = y0 + hole_h_px

    # How many tags fit along each side (maximal packing based on hole span)
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

    # ---- Inner ring centerlines ----
    top_row_y_inner    = y0 - inner_px - tag_px // 2
    bottom_row_y_inner = y1 + inner_px + tag_px // 2
    left_col_x_inner   = x0 - inner_px - tag_px // 2
    right_col_x_inner  = x1 + inner_px + tag_px // 2

    # ---- Outer ring centerlines ----
    ring_offset_px = tag_px + ring_gap_px
    top_row_y_outer    = top_row_y_inner    - ring_offset_px
    bottom_row_y_outer = bottom_row_y_inner + ring_offset_px
    left_col_x_outer   = left_col_x_inner   - ring_offset_px
    right_col_x_outer  = right_col_x_inner  + ring_offset_px

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
        # outer ring
        "top_row_y_outer": top_row_y_outer,
        "bottom_row_y_outer": bottom_row_y_outer,
        "left_col_x_outer": left_col_x_outer,
        "right_col_x_outer": right_col_x_outer,
    }

    # Corner cluster geometry
    if ADD_CORNER_TAGS:
        corner_tag_px = mm_to_px(CORNER_TAG_SIZE_MM)
        cluster_gap_px = mm_to_px(CORNER_CLUSTER_GAP_MM)
        # Slightly tighter to the hole as in your code
        corner_dx = inner_px - corner_tag_px // 10
        corner_dy = inner_px - corner_tag_px // 10

        bases = {
            "tl": (x0 - corner_dx, y0 - corner_dy),
            "tr": (x1 + corner_dx, y0 - corner_dy),
            "bl": (x0 - corner_dx, y1 + corner_dy),
            "br": (x1 + corner_dx, y1 + corner_dy),
        }
        signs = {
            "tl": (-1, -1),
            "tr": (+1, -1),
            "bl": (-1, +1),
            "br": (+1, +1),
        }

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
        # label
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
        draw.rectangle([lx - 2, ly - 2, lx + tw + 2, ly + th + 2], fill=(255, 255, 255))
        # draw.text((lx, ly), label, fill=(0, 0, 0), font=font)

    cx, cy = L["cx"], L["cy"]
    tid = 1

    # ---------- INNER RING ----------
    for xo in L["x_offsets_top"]:
        paste_tag_with_label(tid, (int(round(cx + xo)), int(round(L["top_row_y_inner"]))), "top", tag_px); tid += 1
    for xo in L["x_offsets_top"]:
        paste_tag_with_label(tid, (int(round(cx + xo)), int(round(L["bottom_row_y_inner"]))), "bottom", tag_px); tid += 1
    for yo in L["y_offsets_side"]:
        paste_tag_with_label(tid, (int(round(L["left_col_x_inner"])), int(round(cy + yo))), "left", tag_px); tid += 1
    for yo in L["y_offsets_side"]:
        paste_tag_with_label(tid, (int(round(L["right_col_x_inner"])), int(round(cy + yo))), "right", tag_px); tid += 1

    # ---------- OUTER RING ----------
    for xo in L["x_offsets_top"]:
        paste_tag_with_label(tid, (int(round(cx + xo)), int(round(L["top_row_y_outer"]))), "top", tag_px); tid += 1
    for xo in L["x_offsets_top"]:
        paste_tag_with_label(tid, (int(round(cx + xo)), int(round(L["bottom_row_y_outer"]))), "bottom", tag_px); tid += 1
    for yo in L["y_offsets_side"]:
        paste_tag_with_label(tid, (int(round(L["left_col_x_outer"])), int(round(cy + yo))), "left", tag_px); tid += 1
    for yo in L["y_offsets_side"]:
        paste_tag_with_label(tid, (int(round(L["right_col_x_outer"])), int(round(cy + yo))), "right", tag_px); tid += 1

    # ---------- CORNER CLUSTERS (2x2) ----------
    if ADD_CORNER_TAGS:
        cpx = L["corner_tag_px"]
        for edge in ["tl", "tr", "bl", "br"]:
            for center in L["corner_clusters"][edge]:
                paste_tag_with_label(tid, center, edge, cpx); tid += 1

    # ---------- NEW: INNER GRID INSIDE THE HOLE ----------
    if ADD_INNER_GRID:
        ig_tag_px   = mm_to_px(INNER_GRID_TAG_SIZE_MM)
        ig_gap_px   = mm_to_px(INNER_GRID_GAP_MM)
        ig_margin_px= mm_to_px(INNER_GRID_MARGIN_MM)

        inner_w = (x1 - x0) - 2 * ig_margin_px
        inner_h = (y1 - y0) - 2 * ig_margin_px
        if inner_w > ig_tag_px and inner_h > ig_tag_px:
            # counts along width/height
            def max_count_len(length_px, tag_px, gap_px):
                return max(1, int((length_px + gap_px) // (tag_px + gap_px)))

            cols = max_count_len(inner_w, ig_tag_px, ig_gap_px)
            rows = max_count_len(inner_h, ig_tag_px, ig_gap_px)

            # center the grid
            grid_w = cols * ig_tag_px + (cols - 1) * ig_gap_px
            grid_h = rows * ig_tag_px + (rows - 1) * ig_gap_px
            start_x = x0 + ig_margin_px + (inner_w - grid_w) // 2
            start_y = y0 + ig_margin_px + (inner_h - grid_h) // 2

            for r in range(rows):
                for c in range(cols):
                    cxg = start_x + c * (ig_tag_px + ig_gap_px) + ig_tag_px // 2
                    cyg = start_y + r * (ig_tag_px + ig_gap_px) + ig_tag_px // 2
                    paste_tag(tid, (int(cxg), int(cyg)), ig_tag_px)
                    tid += 1

    # Header text (small)
    try:
        info_font = ImageFont.truetype("DejaVuSans.ttf", mm_to_px(4))
    except Exception:
        info_font = ImageFont.load_default()
    info_text = (f"AprilTag 36h11 - edge tags {TAG_SIZE_MM}mm, inner grid {INNER_GRID_TAG_SIZE_MM}mm - "
                 f"double ring (gap {TAG_GAP_MM}mm, ring gap {RING_GAP_MM}mm)")
    draw.text((10, 20), info_text, fill=(0, 0, 0), font=info_font)

    # Save
    png_path = "apriltag_board_double_ring_innergrid_rot{}.png".format(TAG_ROTATION_DEG)
    pdf_path = "apriltag_board_double_ring_innergrid_rot{}.pdf".format(TAG_ROTATION_DEG)
    canvas.save(png_path, "PNG", dpi=(DPI, DPI))
    canvas.save(pdf_path, "PDF", resolution=DPI)
    print("Saved:", png_path, "and", pdf_path)

if __name__ == "__main__":
    main()
