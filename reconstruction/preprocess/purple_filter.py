import os
import cv2
import numpy as np
from tqdm import tqdm

import numpy as np, cv2

def is_too_purple(image,
                  # HSV magenta detection
                  v_min=180, s_min=40,            # ignore dark/unsaturated pixels
                  hue_lo=135, hue_hi=170,         # OpenCV hue range for purple/magenta (≈270–340°)
                  area_frac_thresh=0.02,          # % of pixels needed to flag (2%)
                  # Overexposed-magenta detection in RGB
                  blow_vmin=245,                  # very bright (near clipping)
                  blow_delta=20                   # (R+B-2G) margin for magenta cast
                  ):
    bgr = image[:, :, :3]                         # drop alpha if present
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]

    # bright & saturated region
    bright = V > v_min
    saturated = S > s_min
    mask = bright & saturated

    # magenta hue fraction (pixel-level)
    magenta = mask & (H >= hue_lo) & (H <= hue_hi)
    frac_magenta = float(magenta.mean())

    # blown-out magenta (handles overexposure)
    b, g, r = [bgr[...,i].astype(np.int16) for i in range(3)]
    near_clip = (np.maximum.reduce([r,b,g]) > blow_vmin)
    magenta_blow = near_clip & ((r + b - 2*g) > blow_delta)
    frac_blow = float(magenta_blow.mean())

    # flag if either condition covers enough of the image
    return (frac_magenta >= area_frac_thresh) or (frac_blow >= area_frac_thresh)


def delete_purple_images(folder_path):
    import json
    import re
    from tqdm import tqdm
    
    # Get list of image files first to show progress
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    deleted_count = 0
    filtered_ids = []
    
    for filename in tqdm(image_files, desc="Processing images"):
        filepath = os.path.join(folder_path, filename)
        img = cv2.imread(filepath)

        if img is None:
            continue

        if is_too_purple(img):
            # Extract scan ID from filename
            match = re.search(r'scan-(\d+)', filename)
            if match:
                scan_id = int(match.group(1))
                filtered_ids.append(scan_id)
            
            os.remove(filepath)
            print(f"Deleted: {filename}")
            deleted_count += 1

    # Save filtered IDs to JSON file
    json_path = os.path.join(folder_path, "filtered_purple_ids.json")
    with open(json_path, 'w') as f:
        json.dump(sorted(list(set(filtered_ids))), f, indent=2)
    
    print(f"\nTotal deleted: {deleted_count}")
    print(f"Filtered scan IDs saved to: {json_path}")

import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    delete_purple_images(os.path.join(cfg.exp_folder, "masks", "filtered_purple_ldr"))

if __name__ == "__main__":
    main()