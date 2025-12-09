#!/usr/bin/env python3
"""
Redraw the original pointing ray onto a sample of images using the .txt xyz_1 and xyz_2.

Assumptions
-----------
- Data layout (configurable with --data_dir, default: ../data):
      ../data/
          <name>.jpg
          <name>.txt

- .txt format:
    line 1: 0 or 1  (not pointing / pointing)
    remaining lines contain at least 6 floats:
        x1 y1 z1 x2 y2 z2
      (they can be split across lines; we just flatten all numbers and
       take the first 6)

    xyz_1 = (x1, y1, z1): wrist position in *meters* in COLOR camera frame
    xyz_2 = (x2, y2, z2): “pointed-to” 3D point (tag center) in *meters*
                          in COLOR camera frame

- We recreate the live visualization logic:

      v = (xyz_2 - xyz_1) / ||xyz_2 - xyz_1||
      point_on_ray = xyz_1 + L * v

  but now xyz are meters, so we use L = 3.0 (3 meters) instead of 300.

- We then use Azure Kinect calibration:

      calib.convert_3d_to_2d( (X_mm, Y_mm, Z_mm),
                              CalibrationType.COLOR,
                              CalibrationType.COLOR )

  to project both xyz_1 and point_on_ray, and draw a red line between them.

- Output:
    For each sampled <name>.jpg with a matching <name>.txt, we write:

        <data_dir>/<out_subdir>/<name>_gt.jpg

    with:
      - red ray from wrist to “far” point along v,
      - cyan circle at wrist,
      - text with label and |xyz_2 - xyz_1| distance (meters).

Only the first N images (sorted) are processed (default: N=20).
"""

import os
import argparse
import cv2
import numpy as np

from pyk4a import PyK4A, Config, CalibrationType


# ==========================
# CONFIG
# ==========================

DEFAULT_DATA_DIR = "data"
DEFAULT_OUT_SUBDIR = "gt_debug_ray"

IMAGE_EXT = ".jpg"
LABEL_EXT = ".txt"

# Length of ray in meters (xyz in .txt are meters)
RAY_LENGTH_M = 3.0

# How many images to process
DEFAULT_NUM_IMAGES = 20


# ==========================
# CALIBRATION
# ==========================

def get_k4a_calibration():
    """
    Open the Azure Kinect, grab its calibration, then close it.

    Returns:
        calib: k4a calibration object usable with convert_3d_to_2d
    """
    k4a = PyK4A(Config())
    k4a.start()
    calib = k4a.calibration
    k4a.stop()
    return calib


# ==========================
# TXT LOADING
# ==========================

def load_points_from_txt(txt_path):
    """
    Load label and two 3D points (meters) from txt.

    Returns:
        label : int
        p1    : np.array shape (3,)  (wrist)
        p2    : np.array shape (3,)  (pointed-to point)
    """
    with open(txt_path, "r") as f:
        lines = [l.strip() for l in f if l.strip() != ""]

    if len(lines) < 2:
        raise ValueError(f"Not enough lines in {txt_path}")

    label = int(lines[0])

    nums = []
    for line in lines[1:]:
        for token in line.replace(",", " ").split():
            nums.append(float(token))

    if len(nums) < 6:
        raise ValueError(f"Need at least 6 numbers in {txt_path}, got {len(nums)}")

    x1, y1, z1, x2, y2, z2 = nums[:6]
    p1 = np.array([x1, y1, z1], dtype=np.float32)
    p2 = np.array([x2, y2, z2], dtype=np.float32)
    return label, p1, p2


# ==========================
# RAY DRAWING
# ==========================

def draw_ray_from_txt_points(
    img_path,
    txt_path,
    out_path,
    calib,
    ray_length_m=RAY_LENGTH_M,
):
    """
    Recreate the pointing ray using xyz_1 and xyz_2 from txt:

        v = (xyz_2 - xyz_1) / ||xyz_2 - xyz_1||
        point_on_ray = xyz_1 + ray_length_m * v

    Project both xyz_1 and point_on_ray using Azure Kinect calibration
    (COLOR -> COLOR) and draw a red line between them.

    Also overlay the 3D distance ||xyz_2 - xyz_1|| in meters.

    Returns:
        True if a ray was successfully drawn (and image saved),
        False otherwise.
    """
    label, p1, p2 = load_points_from_txt(txt_path)

    # Load image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image {img_path}")

    # Compute direction vector v = (p2 - p1) normalized
    v = p2 - p1
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        print(f"[WARN] Zero or tiny direction vector in {os.path.basename(txt_path)} "
              f"p1={p1}, p2={p2}")
        return False
    v /= norm

    # Far point along ray in meters
    far_point = p1 + ray_length_m * v

    # Convert meters -> millimeters for Kinect calibration
    p1_mm = (float(p1[0] * 1000.0), float(p1[1] * 1000.0), float(p1[2] * 1000.0))
    far_mm = (float(far_point[0] * 1000.0),
              float(far_point[1] * 1000.0),
              float(far_point[2] * 1000.0))

    try:
        uv_wrist = calib.convert_3d_to_2d(p1_mm, CalibrationType.COLOR, CalibrationType.COLOR)
        uv_far = calib.convert_3d_to_2d(far_mm, CalibrationType.COLOR, CalibrationType.COLOR)
    except Exception as e:
        print(f"[WARN] convert_3d_to_2d failed for {os.path.basename(txt_path)}: {e}")
        return False

    if uv_wrist is None or uv_far is None:
        print(f"[WARN] Projection returned None for {os.path.basename(txt_path)} "
              f"(p1_mm={p1_mm}, far_mm={far_mm})")
        return False

    u1, v1 = map(int, map(round, uv_wrist))
    u2, v2 = map(int, map(round, uv_far))

    h, w, _ = img.shape
    u1 = max(0, min(w - 1, u1))
    v1 = max(0, min(h - 1, v1))
    u2 = max(0, min(w - 1, u2))
    v2 = max(0, min(h - 1, v2))

    # Draw wrist point (cyan)
    cv2.circle(img, (u1, v1), 8, (255, 255, 0), 2)

    # Draw ray (red)
    cv2.line(img, (u1, v1), (u2, v2), (0, 0, 255), 3)

    # 3D distance between stored p1 and p2 (meters)
    dist_m = float(np.linalg.norm(p2 - p1))

    text = f"GT ray (label={label}, ||p2-p1||={dist_m:.3f} m)"
    cv2.putText(
        img,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        lineType=cv2.LINE_AA,
    )

    cv2.imwrite(out_path, img)
    return True


# ==========================
# MAIN
# ==========================

def main():
    parser = argparse.ArgumentParser(
        description="Redraw pointing rays on a sample of images using xyz_1 and xyz_2 "
                    "from .txt files and Azure Kinect calibration."
    )
    parser.add_argument(
        "--data_dir",
        "-d",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing <name>{IMAGE_EXT} and <name>{LABEL_EXT} "
             f"(default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--out_subdir",
        "-o",
        type=str,
        default=DEFAULT_OUT_SUBDIR,
        help=f"Subfolder inside data_dir for new GT images "
             f"(default: {DEFAULT_OUT_SUBDIR})",
    )
    parser.add_argument(
        "--num_images",
        "-n",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help=f"Number of images to process (default: {DEFAULT_NUM_IMAGES})",
    )

    args = parser.parse_args()

    base = args.data_dir
    if not os.path.isdir(base):
        raise NotADirectoryError(f"{base} is not a directory")

    out_dir = os.path.join(base, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # Grab calibration once
    print("Acquiring Azure Kinect calibration...")
    calib = get_k4a_calibration()
    print("Calibration acquired.")

    # Gather all .jpg names
    all_names = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(base)
        if f.endswith(IMAGE_EXT)
    )

    if not all_names:
        print(f"No {IMAGE_EXT} images found in {base}")
        return

    # Take only the first N
    names = all_names[:args.num_images]
    print(f"Processing {len(names)} image(s) (sample from {len(all_names)} total). "
          f"Output -> {out_dir}")

    for name in names:
        img_path = os.path.join(base, name + IMAGE_EXT)
        txt_path = os.path.join(base, name + LABEL_EXT)
        out_path = os.path.join(out_dir, name + "_gt.jpg")

        if not os.path.exists(txt_path):
            print(f"[WARN] Skipping {name}: no txt file")
            continue

        try:
            ok = draw_ray_from_txt_points(img_path, txt_path, out_path, calib)
            if ok:
                print(f"[OK] {name}: wrote {out_path}")
            else:
                print(f"[SKIP] {name}: could not draw ray")
        except Exception as e:
            print(f"[ERR] {name}: {e}")


if __name__ == "__main__":
    main()
