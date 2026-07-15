"""
Post-hoc quality checks for materials that finished shape matching.

Flags materials that finished successfully but have suspicious metrics that
warrant human review. Separate from `registration_check.py`, which is a
pre-flight gate that decides whether COLMAP succeeded at all.

Detected warnings:
  - CATASTROPHIC_TRANS : all top-per-image translation errors > 20 mm on the
    images that survived the 16 mm filter (Umeyama alignment is bad and the
    filter could not rescue this material)
  - MANY_EXCLUDED      : > EXCLUDED_FRACTION_THRESHOLD of registered images
    were dropped by the 16 mm per-image filter. Umeyama Sim(3) fit failed
    globally, so the obs file looks full but is almost all zeros.
  - MANY_UNMATCHED     : more than UNMATCHED_THRESHOLD scans are missing
    from sparse/images.bin (COLMAP did register >= 90 % of scans but more
    than UNMATCHED_THRESHOLD frames are still unregistered)

Used by: recon/scheduler/job_scheduler.py
"""
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

CATASTROPHIC_TRANS_MM = 20.0
UNMATCHED_THRESHOLD = 50
EXCLUDED_FRACTION_THRESHOLD = 0.50


def parse_top10_translation_errors_mm(log_path: str) -> Optional[List[float]]:
    """
    Parse the 'Top 10 Maximum Translation Errors' block from shape_matching.log.
    Returns a list of per-image translation errors in mm, or None on failure.
    """
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path, 'r', errors='ignore') as f:
            txt = f.read()
    except (IOError, OSError):
        return None
    txt = txt.replace('\r', '\n')

    m = re.search(
        r'Top 10 Maximum Translation Errors:\s*\n={10,}\s*\n(.*?)'
        r'(?:\n={10,}|\nTop 10 Maximum Angular)',
        txt, re.DOTALL)
    if not m:
        return None
    errs_mm = []
    for line in m.group(1).split('\n'):
        mm = re.search(r'(\d+\.\d+)\s*mm', line)
        if mm:
            errs_mm.append(float(mm.group(1)))
    return errs_mm or None


def parse_filter_removal(log_path: str) -> Optional[Tuple[int, int]]:
    """
    Parse '[filter] Removing N/M images with per-image translation error > X mm'
    from shape_matching.log. Returns (n_removed, n_total) or None on failure.
    """
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path, 'r', errors='ignore') as f:
            txt = f.read()
    except (IOError, OSError):
        return None
    m = re.search(r'\[filter\] Removing (\d+)/(\d+) images', txt)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def count_unmatched_scans(folder_path: str) -> Optional[int]:
    """Return number of scans in unmatched_scan_ids.json, or None if missing."""
    p = Path(folder_path) / "unmatched_scan_ids.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return len(json.load(f))
    except (IOError, OSError, ValueError):
        return None


def detect_quality_warnings(folder_path: str) -> List[Dict[str, str]]:
    """
    Inspect the finished outputs of a material and return a list of warnings.

    Each warning is a dict with keys:
      - code: short machine-readable tag (CATASTROPHIC_TRANS, MANY_UNMATCHED)
      - msg : human-readable detail string (one line)

    Returns empty list if the material is healthy (or if expected inputs are
    missing — we never want a missing input file to be reported as a warning,
    since that means shape matching probably didn't finish cleanly and another
    code path already handles it).
    """
    warnings: List[Dict[str, str]] = []

    log_path = os.path.join(folder_path, "shape_matching.log")
    errs_mm = parse_top10_translation_errors_mm(log_path)
    if errs_mm:
        if all(e > CATASTROPHIC_TRANS_MM for e in errs_mm):
            warnings.append({
                "code": "CATASTROPHIC_TRANS",
                "msg": (f"all {len(errs_mm)} top translation errors > "
                        f"{CATASTROPHIC_TRANS_MM:.0f} mm "
                        f"(max {max(errs_mm):.1f} mm, min {min(errs_mm):.1f} mm)"),
            })

    filt = parse_filter_removal(log_path)
    if filt is not None:
        n_removed, n_total = filt
        if n_total > 0 and (n_removed / n_total) > EXCLUDED_FRACTION_THRESHOLD:
            warnings.append({
                "code": "MANY_EXCLUDED",
                "msg": (f"{n_removed}/{n_total} images filtered as high-error "
                        f"({100*n_removed/n_total:.1f}%, > "
                        f"{100*EXCLUDED_FRACTION_THRESHOLD:.0f}% threshold) — "
                        f"Umeyama Sim(3) fit likely failed"),
            })

    n_unmatched = count_unmatched_scans(folder_path)
    if n_unmatched is not None and n_unmatched > UNMATCHED_THRESHOLD:
        warnings.append({
            "code": "MANY_UNMATCHED",
            "msg": (f"{n_unmatched} scans unmatched "
                    f"(>{UNMATCHED_THRESHOLD} threshold)"),
        })

    return warnings
