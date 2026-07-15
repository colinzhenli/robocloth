#!/usr/bin/env python3
"""Collect eval_stage2_ubo.sh results and compare against the paper table.

Usage: python collect_eval_results_ubo.py <eval_results_ubo_dir>

Prints the reproduced "Cross-dataset transfer to UBO2014" table (12 held-out
materials) next to the values reported in the paper.
"""
import glob
import json
import os
import sys

# Paper: Table "Cross-dataset transfer to UBO2014" (12 held-out materials).
# Columns: stage-1 decoder training source (+ Disney-PBR baseline).
PAPER = {
    "fabric02": {"Ours": 34.57, "Bonn": 28.56, "MERL": 27.88, "PBR": 30.96},
    "fabric04": {"Ours": 34.00, "Bonn": 28.50, "MERL": 26.66, "PBR": 30.09},
    "fabric09": {"Ours": 39.50, "Bonn": 36.08, "MERL": 33.96, "PBR": 36.96},
    "fabric11": {"Ours": 34.33, "Bonn": 29.45, "MERL": 27.46, "PBR": 31.24},
    "felt01":   {"Ours": 38.76, "Bonn": 33.60, "MERL": 32.81, "PBR": 35.05},
    "felt03":   {"Ours": 40.30, "Bonn": 34.20, "MERL": 33.27, "PBR": 36.66},
    "felt05":   {"Ours": 35.39, "Bonn": 32.13, "MERL": 30.32, "PBR": 32.63},
    "felt10":   {"Ours": 44.36, "Bonn": 38.52, "MERL": 36.38, "PBR": 41.10},
    "carpet02": {"Ours": 38.87, "Bonn": 34.82, "MERL": 32.56, "PBR": 33.75},
    "carpet07": {"Ours": 34.61, "Bonn": 30.17, "MERL": 28.85, "PBR": 29.48},
    "carpet09": {"Ours": 33.17, "Bonn": 31.23, "MERL": 29.42, "PBR": 29.97},
    "carpet12": {"Ours": 33.25, "Bonn": 29.93, "MERL": 28.13, "PBR": 28.70},
}
PAPER_AVG = {"Ours": 36.76, "Bonn": 32.27, "MERL": 30.64, "PBR": 33.05}
MODELS = ["Ours", "Bonn", "MERL", "PBR"]


def main(results_dir: str) -> None:
    got = {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        with open(path) as f:
            r = json.load(f)
        if r.get("val_psnr") is not None:
            got.setdefault(str(r["material"]), {})[r["model"]] = r["val_psnr"]

    header = f"{'material':>9} | " + " | ".join(f"{m:>21}" for m in MODELS)
    sub = f"{'':>9} | " + " | ".join(f"{'repro / paper / diff':>21}" for _ in MODELS)
    print("\n=== Cross-dataset transfer to UBO2014 — PSNR (dB) ===")
    print(header)
    print(sub)
    print("-" * len(header))

    sums, counts = {m: 0.0 for m in MODELS}, {m: 0 for m in MODELS}
    for mat in PAPER:
        cells = []
        for m in MODELS:
            repro = got.get(mat, {}).get(m)
            paper = PAPER[mat][m]
            if repro is None:
                cells.append(f"{'--':>7} / {paper:5.2f} /    --")
            else:
                sums[m] += repro
                counts[m] += 1
                cells.append(f"{repro:7.2f} / {paper:5.2f} / {repro - paper:+5.2f}")
        print(f"{mat:>9} | " + " | ".join(cells))

    avg_cells = []
    for m in MODELS:
        if counts[m] == len(PAPER):
            avg = sums[m] / counts[m]
            avg_cells.append(f"{avg:7.2f} / {PAPER_AVG[m]:5.2f} / {avg - PAPER_AVG[m]:+5.2f}")
        else:
            avg_cells.append(f"{'--':>7} / {PAPER_AVG[m]:5.2f} /    -- ({counts[m]}/{len(PAPER)})")
    print("-" * len(header))
    print(f"{'average':>9} | " + " | ".join(avg_cells))
    print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval_results_ubo")
