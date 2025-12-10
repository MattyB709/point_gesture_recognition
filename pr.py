#!/usr/bin/env python3
"""
Interactive GT pointing pipeline.

Behavior
--------
For each base name <name> in data_dir with <name>.jpg and <name>.txt:

- .txt format:
    line 1: 0 or 1  (not pointing / pointing)
    remaining lines: at least 6 floats:
        x1 y1 z1 x2 y2 z2   (in METERS, camera frame of the RGB image)

    xyz_1 = (x1, y1, z1): wrist position (meters)
    xyz_2 = (x2, y2, z2): pointed-to position (meters)

Pipeline
--------
- If label == 0 (no pointing):
    * Automatically MOVE <name>.jpg, <name>.txt, <name>.npy (if exists)
      into data_dir/no_pointing/

- If label == 1 (pointing):
    * Build an overlay image with:
        - cyan dot at wrist
        - red line from wrist -> pointed-to
          (if p2 behind camera, we "clip"/shorten the line to z>0)
        - text overlay with mode ("good" or "clipped")
    * Show overlay and wait for key:

        SPACE (' '):
          - APPROVE:
              If mode == "clipped":
                  -> data_dir/approved_mod/
                     * copy original <name>.jpg (no overlay)
                     * copy original <name>.npy (if exists)
                     * write new <name>.txt with the clipped coordinates
              If mode == "good":
                  -> data_dir/approved_norm/
                     * copy original <name>.jpg, <name>.txt, <name>.npy (if exists)

        'r':
          - REJECT:
              If mode == "clipped":
                  -> data_dir/remove_mod/
                     * copy original <name>.jpg, <name>.txt, <name>.npy (if exists)
              If mode == "good":
                  -> data_dir/remove_norm/
                     * copy original <name>.jpg, <name>.txt, <name>.npy (if exists)

        'l':
          - REVIEW:
              If mode == "clipped":
                  -> data_dir/review_mod/
              If mode == "good":
                  -> data_dir/review_norm/
              (copy original <name>.jpg, <name>.txt, <name>.npy if exists)

        'q' or ESC:
          - Quit the program.

- If wrist cannot be projected, the image is skipped.

Only the first N images (default N=500) are processed (sorted by filename).
"""

import os
import argparse
import cv2
import numpy as np
import shutil

# ==========================
# CONFIG
# ==========================

DEFAULT_DATA_DIR = "../data"
DEFAULT_NUM_IMAGES = 500

IMAGE_EXT = ".jpg"
LABEL_EXT = ".txt"
NPY_EXT = ".npy"

# Camera intrinsics (pixels) for the camera whose images are the .jpgs
FX = 919.76178
FY = 919.8909
CX = 962.6875
CY = 550.9944

# Output subdirectories (inside data_dir)
NO_POINTING_SUBDIR = "no_pointing"
APPROVED_MOD_SUBDIR = "approved_mod"      # shortened / clipped
APPROVED_NORM_SUBDIR = "approved_norm"    # no modification
REMOVE_MOD_SUBDIR = "remove_mod"
REMOVE_NORM_SUBDIR = "remove_norm"
REVIEW_MOD_SUBDIR = "review_mod"
REVIEW_NORM_SUBDIR = "review_norm"


# ==========================
# HELPERS
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
    p1 = np.array([x1, y1, z1], dtype=np.float32)  # wrist
    p2 = np.array([x2, y2, z2], dtype=np.float32)  # pointed-to
    return label, p1, p2


def write_points_to_txt(txt_path, label, p1, p2):
    """
    Write label and two 3D points (meters) to txt in the same style as input.
    """
    with open(txt_path, "w") as f:
        f.write(f"{int(label)}\n")
        f.write(
            "{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                float(p1[0]), float(p1[1]), float(p1[2]),
                float(p2[0]), float(p2[1]), float(p2[2]),
            )
        )


def project_point_intrinsics(pt, fx, fy, cx, cy, width, height):
    """
    Project 3D camera point (meters) -> pixel coordinates using pinhole intrinsics.

    pt: np.array([X, Y, Z]) in meters.

    Returns:
        (u, v) as ints, clamped to image bounds, or None if Z <= 0 or invalid.
    """
    X, Y, Z = pt

    # Require point to be in front of the camera
    if not np.isfinite(Z) or Z <= 0:
        return None

    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy

    u = int(round(u))
    v = int(round(v))

    u = int(np.clip(u, 0, width - 1))
    v = int(np.clip(v, 0, height - 1))

    return (u, v)


def clip_point_to_positive_z(p1, p2, margin=0.9):
    """
    Given two 3D points (meters) p1, p2, and assuming p1.z > 0 but p2.z <= 0,
    compute a point along the line from p1 to p2 that still has z > 0.

    We parametrize:
        p(t) = p1 + t * (p2 - p1),  t in [0, 1]

    z(t) = z1 + t * (z2 - z1).
    Solve for z(t) = 0:
        t0 = z1 / (z1 - z2)

    Then we choose t_clip = margin * t0 with margin < 1, so that z(t_clip) > 0
    but close to the boundary.

    Returns:
        p_clip: np.array shape (3,) or None if we can't find a valid t.
    """
    z1 = float(p1[2])
    z2 = float(p2[2])

    if z1 <= 0:
        return None

    denom = z1 - z2
    if abs(denom) < 1e-8:
        return None

    t0 = z1 / denom  # where z(t0) = 0
    if t0 <= 0:
        # Line never goes "toward" camera in z>0 region
        return None

    t_clip = margin * t0
    if t_clip <= 0:
        return None

    v = p2 - p1
    p_clip = p1 + t_clip * v
    if p_clip[2] <= 0:
        return None

    return p_clip


def make_overlay_image(
    img_path,
    label,
    p1,
    p2,
    fx=FX,
    fy=FY,
    cx=CX,
    cy=CY,
):
    """
    Build an overlay image for visualization.

    Returns:
        img_overlay : np.array (H, W, 3) with GT line drawn
        mode        : "good" or "clipped"
        p2_effective: 3D endpoint used (p2 for good, clipped point for clipped)
        dist_m      : ||p2 - p1|| in meters (original, for info)

        or (None, None, None, None) if we cannot project.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image {img_path}")

    h, w, _ = img.shape

    p1_px = project_point_intrinsics(p1, fx, fy, cx, cy, w, h)
    p2_px = project_point_intrinsics(p2, fx, fy, cx, cy, w, h)

    # Distance between the two 3D points in meters
    dist_m = float(np.linalg.norm(p2 - p1))

    if p1_px is None:
        # Can't even draw the wrist; bail
        print(
            f"[WARN] Cannot project wrist for {os.path.basename(img_path)} | "
            f"p1={p1}, p2={p2}"
        )
        return None, None, None, None

    mode = None
    endpoint_px = None
    p2_effective = None

    if p2_px is not None:
        # Normal case (no shortening)
        mode = "good"
        endpoint_px = p2_px
        p2_effective = p2
    else:
        # Need to clip: p1 ok, p2 bad -> shortened line
        p_clip = clip_point_to_positive_z(p1, p2, margin=0.9)
        if p_clip is None:
            print(
                f"[WARN] Cannot clip line to positive Z for "
                f"{os.path.basename(img_path)} | p1={p1}, p2={p2}"
            )
            return None, None, None, None

        p_clip_px = project_point_intrinsics(p_clip, fx, fy, cx, cy, w, h)
        if p_clip_px is None:
            print(
                f"[WARN] Clipped point still not projectable for "
                f"{os.path.basename(img_path)} | p1={p1}, p2={p2}, p_clip={p_clip}"
            )
            return None, None, None, None

        mode = "clipped"
        endpoint_px = p_clip_px
        p2_effective = p_clip

    img_overlay = img.copy()

    # Draw wrist point (cyan)
    cv2.circle(img_overlay, p1_px, 8, (255, 255, 0), 2)

    # Draw line (red) from wrist to endpoint_px
    cv2.line(img_overlay, p1_px, endpoint_px, (0, 0, 255), 3)

    # Mark endpoint (white)
    cv2.circle(img_overlay, endpoint_px, 6, (255, 255, 255), 2)

    # Text overlay
    text = f"GT line ({mode}, label={label}, ||p2-p1||={dist_m:.3f} m)"
    cv2.putText(
        img_overlay,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        lineType=cv2.LINE_AA,
    )

    return img_overlay, mode, p2_effective, dist_m


def maybe_move(src_path, dst_dir):
    """
    Move file if it exists.
    """
    if os.path.exists(src_path):
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.move(src_path, dst_path)


def maybe_copy(src_path, dst_dir):
    """
    Copy file if it exists.
    """
    if os.path.exists(src_path):
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)


# ==========================
# MAIN
# ==========================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Interactive pipeline to separate pointing / non-pointing images,\n"
            "draw GT lines, and route files into different folders based on\n"
            "keyboard approval."
        )
    )
    parser.add_argument(
        "--data_dir",
        "-d",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing <name>{IMAGE_EXT}, <name>{LABEL_EXT}, "
             f"and optionally <name>{NPY_EXT} (default: {DEFAULT_DATA_DIR})",
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

    # Collect all jpg base names
    all_names = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(base)
        if f.endswith(IMAGE_EXT)
    )

    if not all_names:
        print(f"No {IMAGE_EXT} images found in {base}")
        return

    names = all_names[:args.num_images]
    print(f"Processing {len(names)} image(s) (sample from {len(all_names)} total).")
    print("Keys: SPACE = approve, 'r' = reject, 'l' = review, 'q'/ESC = quit")

    n_no_pointing = 0
    n_pointing = 0
    n_approved_short = 0
    n_approved_nomod = 0
    n_rejected_mod = 0
    n_rejected_norm = 0
    n_review_mod = 0
    n_review_norm = 0
    n_skipped = 0

    for name in names:
        img_path = os.path.join(base, name + IMAGE_EXT)
        txt_path = os.path.join(base, name + LABEL_EXT)
        npy_path = os.path.join(base, name + NPY_EXT)

        if not os.path.exists(txt_path):
            print(f"[WARN] Skipping {name}: no txt file")
            n_skipped += 1
            continue

        try:
            label, p1, p2 = load_points_from_txt(txt_path)
        except Exception as e:
            print(f"[ERR] {name}: failed to read txt: {e}")
            n_skipped += 1
            continue

        # Case 1: no pointing -> auto-move
        if label == 0:
            dst = os.path.join(base, NO_POINTING_SUBDIR)
            maybe_move(img_path, dst)
            maybe_move(txt_path, dst)
            maybe_move(npy_path, dst)
            n_no_pointing += 1
            print(f"[NO_POINTING] {name} -> {dst}")
            continue

        # Case 2: pointing -> interactive
        n_pointing += 1

        try:
            overlay, mode, p2_effective, dist_m = make_overlay_image(
                img_path, label, p1, p2
            )
        except Exception as e:
            print(f"[ERR] {name}: {e}")
            n_skipped += 1
            continue

        if overlay is None or mode is None:
            print(f"[SKIP] {name}: could not project/clip")
            n_skipped += 1
            continue

        # Show overlay and wait for user decision
        win_name = "GT Overlay"
        cv2.imshow(win_name, overlay)
        print(
            f"[POINTING] {name} (mode={mode}, ||p2-p1||={dist_m:.3f} m) "
            f"-> SPACE=approve, r=reject, l=review, q/ESC=quit"
        )

        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        # Quit
        if key in (27, ord('q'), ord('Q')):  # ESC or q -> quit loop
            print("[INFO] Quit requested, stopping pipeline.")
            break

        # REJECT
        if key in (ord('r'), ord('R')):
            if mode == "clipped":
                dst = os.path.join(base, REMOVE_MOD_SUBDIR)
                n_rejected_mod += 1
            else:
                dst = os.path.join(base, REMOVE_NORM_SUBDIR)
                n_rejected_norm += 1

            maybe_copy(img_path, dst)
            maybe_copy(txt_path, dst)
            maybe_copy(npy_path, dst)
            print(f"[REJECT] {name} (mode={mode}) -> {dst}")
            continue

        # REVIEW
        if key in (ord('l'), ord('L')):
            if mode == "clipped":
                dst = os.path.join(base, REVIEW_MOD_SUBDIR)
                n_review_mod += 1
            else:
                dst = os.path.join(base, REVIEW_NORM_SUBDIR)
                n_review_norm += 1

            maybe_copy(img_path, dst)
            maybe_copy(txt_path, dst)
            maybe_copy(npy_path, dst)
            print(f"[REVIEW] {name} (mode={mode}) -> {dst}")
            continue

        # APPROVE
        if key == ord(' '):
            if mode == "clipped":
                # Shortened line -> new txt with clipped p2
                dst = os.path.join(base, APPROVED_MOD_SUBDIR)
                os.makedirs(dst, exist_ok=True)

                # Copy original image and npy
                maybe_copy(img_path, dst)
                maybe_copy(npy_path, dst)

                # Write new txt with clipped coordinates
                new_txt_path = os.path.join(dst, os.path.basename(txt_path))
                write_points_to_txt(new_txt_path, label, p1, p2_effective)

                n_approved_short += 1
                print(
                    f"[APPROVE_MOD] {name} -> {dst} "
                    f"(shortened coords written to {os.path.basename(new_txt_path)})"
                )
            else:
                # No modification -> copy original triad
                dst = os.path.join(base, APPROVED_NORM_SUBDIR)
                maybe_copy(img_path, dst)
                maybe_copy(txt_path, dst)
                maybe_copy(npy_path, dst)
                n_approved_nomod += 1
                print(f"[APPROVE_NORM] {name} -> {dst}")
            continue

        # Any other key -> skip
        print(f"[SKIP] {name}: unrecognized key {key}")
        n_skipped += 1

    print("\nSummary:")
    print(f"  No pointing moved       : {n_no_pointing}")
    print(f"  Pointing encountered    : {n_pointing}")
    print(f"    Approved (mod)        : {n_approved_short}")
    print(f"    Approved (norm)       : {n_approved_nomod}")
    print(f"    Rejected (mod)        : {n_rejected_mod}")
    print(f"    Rejected (norm)       : {n_rejected_norm}")
    print(f"    Review (mod)          : {n_review_mod}")
    print(f"    Review (norm)         : {n_review_norm}")
    print(f"  Skipped                 : {n_skipped}")


if __name__ == "__main__":
    main()
