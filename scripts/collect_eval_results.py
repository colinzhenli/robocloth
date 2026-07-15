#!/usr/bin/env python3
"""Collect eval_stage2.sh results and compare against the paper table.

Usage: python collect_eval_results.py <eval_results_dir>

Reads every <mat>_<model>.json produced by eval_stage2.sh and prints the
reproduced "Per-material reconstruction PSNR" table (top block: in-domain on
our held-out test set) next to the values reported in the paper.
"""
import glob
import json
import os
import sys

# Paper: Table "Per-material reconstruction PSNR", top block (our test set).
# Rows: material; columns: stage-1 decoder source (+ PBR baseline).
PAPER = {
    "226": {"Ours": 29.24, "Bonn": 25.63, "MERL": 24.24, "PBR": 25.35},
    "314": {"Ours": 26.60, "Bonn": 24.42, "MERL": 22.93, "PBR": 23.98},
    "370": {"Ours": 29.71, "Bonn": 29.21, "MERL": 27.75, "PBR": 29.29},
    "145": {"Ours": 28.44, "Bonn": 24.45, "MERL": 23.39, "PBR": 24.28},
    "452": {"Ours": 34.15, "Bonn": 32.91, "MERL": 30.89, "PBR": 32.06},
}
PAPER_AVG = {"Ours": 29.63, "Bonn": 27.32, "MERL": 25.84, "PBR": 26.99}
MODELS = ["Ours", "Bonn", "MERL", "PBR"]


def main(results_dir: str) -> None:
    got = {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        with open(path) as f:
            r = json.load(f)
        if r.get("val_psnr") is not None:
            got.setdefault(str(r["material"]), {})[r["model"]] = r["val_psnr"]

    header = f"{'material':>8} | " + " | ".join(f"{m:>21}" for m in MODELS)
    sub = f"{'':>8} | " + " | ".join(f"{'repro / paper / diff':>21}" for _ in MODELS)
    print("\n=== Per-material reconstruction PSNR (dB) — our held-out test set ===")
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
        print(f"{mat:>8} | " + " | ".join(cells))

    avg_cells = []
    for m in MODELS:
        if counts[m] == len(PAPER):
            avg = sums[m] / counts[m]
            avg_cells.append(f"{avg:7.2f} / {PAPER_AVG[m]:5.2f} / {avg - PAPER_AVG[m]:+5.2f}")
        else:
            avg_cells.append(f"{'--':>7} / {PAPER_AVG[m]:5.2f} /    -- ({counts[m]}/{len(PAPER)})")
    print("-" * len(header))
    print(f"{'average':>8} | " + " | ".join(avg_cells))
    print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval_results")
