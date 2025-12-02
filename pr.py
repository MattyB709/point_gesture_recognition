#!/usr/bin/env python3
"""
Compare Mediapipe fingertip vectors (origin at txt wrist) to ground-truth vectors,
and draw everything from the txt wrist.

Triplets:
    <name>.jpg  - RGB image (aligned with depth)
    <name>.npy  - depth map (H x W, Azure Kinect depth in millimeters)
    <name>.txt  - label + 3D ground-truth points (in meters)

.txt format:
    line 1: 0 or 1 (not pointing / pointing)
    remaining: x1 y1 z1 x2 y2 z2   (in meters, camera frame)

We treat P1 = (x1, y1, z1) from the .txt as the ONLY wrist.

For each pointing image:
    - From Mediapipe, get all hands' wrists + index fingertips in pixels.
    - Use depth + intrinsics to map each fingertip pixel -> 3D camera point Q2_i.
    - Build rays:
        v_gt   = P2 - P1
        v_mp,i = Q2_i - P1
    - Compute angle between v_gt and each v_mp,i (when depth is valid), pick the smallest.

Visualization:
    - Project P1 and P2 to image pixels using either:
        * Azure Kinect calibration (if provided), or
        * plain fx,fy,cx,cy intrinsics (fallback).
    - Draw from the txt wrist pixel:
        * GT line: P1_px -> P2_px (RED).
        * For each Mediapipe fingertip: P1_px -> tip_px_i.
            - If candidate had valid 3D: angle shown; the best one is GREEN, others YELLOW.
            - If depth was invalid: draw YELLOW line, angle label 'x'.
    - Also draw orange dots at all Mediapipe wrists with L/R labels.

We never recompute a new 3D wrist; we only find new fingertip points.
"""

import os
import math
import argparse

import cv2
import numpy as np
import mediapipe as mp

# Optional Azure Kinect calibration (pyk4a)
try:
    from pyk4a import CalibrationType
    HAVE_PYK4A = True
except ImportError:
    HAVE_PYK4A = False

# ==========================
# CONFIG
# ==========================

DATA_DIR = "data/"

IMAGE_EXT = ".jpg"
DEPTH_EXT = ".npy"
LABEL_EXT = ".txt"

# Intrinsics for the camera the JPGs come from (and depth is registered to)
FX = 919.76178
FY = 919.8909
CX = 962.6875
CY = 550.9944

MAX_PAIRS = 50
ANNOTATION_SUBDIR = "annotated"

mp_hands = mp.solutions.hands

# Global calibration hook; set this externally if you want true Kinect 3D->2D.
# Example (in your own setup code):
#   from pyk4a import PyK4A, Config
#   k4a = PyK4A(Config())
#   k4a.start()
#   from this_script import K4A_CALIBRATION
#   K4A_CALIBRATION = k4a.calibration
K4A_CALIBRATION = None


# ==========================
# I/O HELPERS
# ==========================

def load_label_and_points(txt_path):
    """Load label and ground-truth 3D points (meters) from .txt."""
    with open(txt_path, "r") as f:
        lines = [l.strip() for l in f if l.strip() != ""]

    if len(lines) < 2:
        raise ValueError("File has fewer than 2 non-empty lines")

    label = int(lines[0])

    nums = []
    for line in lines[1:]:
        for token in line.replace(",", " ").split():
            nums.append(float(token))

    if len(nums) < 6:
        raise ValueError(f"Found only {len(nums)} numeric values (need at least 6)")

    x1, y1, z1, x2, y2, z2 = nums[:6]
    p1 = np.array([x1, y1, z1], dtype=np.float32)  # wrist (from txt)
    p2 = np.array([x2, y2, z2], dtype=np.float32)  # target (from txt)
    return label, p1, p2


def get_all_hands_wrist_tip(image_bgr, hands_detector):
    """
    Run Mediapipe Hands and return info for all hands:

    Returns:
        hand_infos: list of dicts:
            - wrist_px: (u_wrist, v_wrist)
            - tip_px  : (u_tip, v_tip)
            - label   : "Left" / "Right" / "?" (MP's handedness guess)
    """
    h, w, _ = image_bgr.shape
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)

    hand_infos = []
    if not results.multi_hand_landmarks:
        return hand_infos

    labels = []
    if results.multi_handedness:
        for handedness in results.multi_handedness:
            labels.append(handedness.classification[0].label)
    else:
        labels = ["?"] * len(results.multi_hand_landmarks)

    for i, lm in enumerate(results.multi_hand_landmarks):
        label = labels[i] if i < len(labels) else "?"
        wrist = lm.landmark[0]
        fingertip = lm.landmark[8]

        u_wrist = int(round(wrist.x * w))
        v_wrist = int(round(wrist.y * h))
        u_tip = int(round(fingertip.x * w))
        v_tip = int(round(fingertip.y * h))

        u_wrist = int(np.clip(u_wrist, 0, w - 1))
        v_wrist = int(np.clip(v_wrist, 0, h - 1))
        u_tip = int(np.clip(u_tip, 0, w - 1))
        v_tip = int(np.clip(v_tip, 0, h - 1))

        hand_infos.append({
            "wrist_px": (u_wrist, v_wrist),
            "tip_px": (u_tip, v_tip),
            "label": label,
        })

    return hand_infos


def pixel_to_camera(u, v, depth_mm, fx, fy, cx, cy):
    """
    Pixel (u, v) + depth_mm -> 3D camera coordinates (meters).

    depth_mm: H x W depth map in millimeters (Azure Kinect).
    """
    h, w = depth_mm.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return None

    d_mm = float(depth_mm[v, u])
    if d_mm <= 0 or np.isnan(d_mm):
        return None

    d_m = d_mm / 1000.0  # mm -> m

    X = (u - cx) * d_m / fx
    Y = (v - cy) * d_m / fy
    Z = d_m
    return np.array([X, Y, Z], dtype=np.float32)


# ==========================
# 3D -> 2D PROJECTION
# ==========================

def project_camera_to_pixel_k4a(pt_m, calibration, width, height,
                                src_type=None,
                                dst_type=None):
    """
    Use Azure Kinect calibration to project a 3D point (meters) into 2D pixels.

    Args:
        pt_m       : np.array([X, Y, Z]) in meters, in the 'src_type' camera frame
        calibration: pyk4a calibration object
        width, height: color image size (for clamping)
        src_type   : CalibrationType (DEPTH or COLOR)
        dst_type   : CalibrationType (typically COLOR)

    Returns:
        (u, v) pixel coords (ints in [0, w-1]x[0, h-1]) or None if invalid.
    """
    if calibration is None or not HAVE_PYK4A:
        return None

    if src_type is None or dst_type is None:
        # Default to depth->color, which is the standard mapping case.
        src_type = CalibrationType.DEPTH
        dst_type = CalibrationType.COLOR

    X, Y, Z = pt_m
    if not np.isfinite(Z) or abs(Z) < 1e-6:
        return None

    # pyk4a expects millimeters
    xyz_mm = (X * 1000.0, Y * 1000.0, Z * 1000.0)

    res = calibration.convert_3d_to_2d(xyz_mm, src_type, dst_type)
    if res is None:
        return None

    u, v = res
    u = int(round(u))
    v = int(round(v))
    u = int(np.clip(u, 0, width - 1))
    v = int(np.clip(v, 0, height - 1))
    return (u, v)


def project_camera_to_pixel_intrinsics(pt, fx, fy, cx, cy, width, height):
    """
    Project 3D camera point (meters) -> pixel coordinates using fx,fy,cx,cy.
    """
    X, Y, Z = pt
    if not np.isfinite(Z) or abs(Z) < 1e-6:
        return None
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    u = int(round(u))
    v = int(round(v))
    u = int(np.clip(u, 0, width - 1))
    v = int(np.clip(v, 0, height - 1))
    return (u, v)


def project_camera_to_pixel(pt, fx, fy, cx, cy, width, height):
    """
    Wrapper: try Azure Kinect calibration first, then fall back to intrinsics.
    Assumes pt is in the same frame as 'src_type' in project_camera_to_pixel_k4a.
    """
    # Try K4A if available and set
    if HAVE_PYK4A and K4A_CALIBRATION is not None:
        p = project_camera_to_pixel_k4a(
            pt, K4A_CALIBRATION, width, height,
            src_type=CalibrationType.DEPTH,  # adjust if your txt is in COLOR frame
            dst_type=CalibrationType.COLOR
        )
        if p is not None:
            return p

    # Fallback: plain intrinsics
    return project_camera_to_pixel_intrinsics(pt, fx, fy, cx, cy, width, height)


# ==========================
# CORE LOGIC
# ==========================

def angle_between(v1, v2):
    """Angle (deg) and cos similarity between two 3D vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None, None
    v1u = v1 / n1
    v2u = v2 / n2
    cos_sim = float(np.clip(np.dot(v1u, v2u), -1.0, 1.0))
    angle_rad = math.acos(cos_sim)
    return math.degrees(angle_rad), cos_sim


def annotate_image(image_bgr,
                   p1_px,
                   p2_px,
                   hand_candidates,
                   best_idx,
                   out_path,
                   name):
    """
    Draw:
      - txt wrist (P1) as cyan dot.
      - GT line (P1 -> P2) as RED.
      - For each hand candidate: P1 -> tip_px_i
            best: green
            others: yellow
      - Orange dots at all Mediapipe wrists with L/R labels.
    """
    annotated = image_bgr.copy()

    # txt wrist
    if p1_px is not None:
        cv2.circle(annotated, p1_px, 8, (255, 255, 0), 2)  # cyan

    # GT line from P1 to P2 (original point line) in RED
    if p1_px is not None and p2_px is not None:
        cv2.line(annotated, p1_px, p2_px, (0, 0, 255), 3)  # red

    angle_best = None
    cos_best = None

    # Draw all candidates from P1
    for i, cand in enumerate(hand_candidates):
        wrist_px_mp = cand["wrist_px"]
        tip_px = cand["tip_px"]
        angle_deg = cand["angle_deg"]     # may be None if depth invalid
        cos_sim = cand["cos_sim"]         # may be None
        label = cand["label"]

        # orange wrist (Mediapipe)
        cv2.circle(annotated, wrist_px_mp, 7, (0, 165, 255), 2)
        cv2.putText(
            annotated,
            label[0] if label else "?",
            (wrist_px_mp[0] + 10, wrist_px_mp[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            1,
            lineType=cv2.LINE_AA,
        )

        if p1_px is not None:
            if i == best_idx and angle_deg is not None:
                color = (0, 255, 0)  # green
                thickness = 3
                angle_best = angle_deg
                cos_best = cos_sim
            else:
                color = (0, 255, 255)  # yellow
                thickness = 2

            # line from txt wrist to candidate fingertip
            cv2.line(annotated, p1_px, tip_px, color, thickness)

            # small label at fingertip with angle (or 'x' if no angle)
            if angle_deg is not None:
                angle_text = f"{angle_deg:.1f}"
            else:
                angle_text = "x"
            cv2.putText(
                annotated,
                angle_text,
                (tip_px[0] + 5, tip_px[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                lineType=cv2.LINE_AA,
            )

    if angle_best is None:
        angle_best = 0.0
        cos_best = 0.0
    text = f"{name} ang_best={angle_best:.1f} cos_best={cos_best:.2f} candidates={len(hand_candidates)}"
    cv2.putText(
        annotated,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        lineType=cv2.LINE_AA,
    )

    cv2.imwrite(out_path, annotated)


def process_single_triplet(base_path,
                           name,
                           fx,
                           fy,
                           cx,
                           cy,
                           hands_detector,
                           annot_dir=None):
    """
    Process a single <name> triplet and choose the fingertip giving least angle error,
    using P1 from .txt as the only wrist.
    """
    img_path = os.path.join(base_path, name + IMAGE_EXT)
    depth_path = os.path.join(base_path, name + DEPTH_EXT)
    txt_path = os.path.join(base_path, name + LABEL_EXT)

    if not (os.path.exists(img_path) and os.path.exists(depth_path) and os.path.exists(txt_path)):
        print(f"Skipping {name}: missing one of the files.")
        return None

    try:
        label, p1_gt, p2_gt = load_label_and_points(txt_path)
    except Exception as e:
        print(f"Skipping {name}: bad label/points file ({e})")
        return None

    if label == 0:
        return None  # not pointing

    image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"Skipping {name}: failed to read image {img_path}")
        return None

    depth_mm = np.load(depth_path)
    if depth_mm.ndim != 2:
        print(f"Skipping {name}: depth map is not HxW (ndim={depth_mm.ndim})")
        return None

    h, w, _ = image_bgr.shape

    # Ground-truth vector (all from txt)
    v_gt = p2_gt - p1_gt

    # Project txt P1 and P2 into image for drawing
    p1_px = project_camera_to_pixel(p1_gt, fx, fy, cx, cy, w, h)
    p2_px = project_camera_to_pixel(p2_gt, fx, fy, cx, cy, w, h)

    # Gather all hands
    hands = get_all_hands_wrist_tip(image_bgr, hands_detector)
    if not hands:
        return None

    candidates = []
    best_idx = None
    best_angle = None
    best_cos = None

    for info in hands:
        (u_tip, v_tip) = info["tip_px"]

        # Try to get fingertip 3D point from depth
        q2 = pixel_to_camera(u_tip, v_tip, depth_mm, fx, fy, cx, cy)

        if q2 is not None:
            v_mp = q2 - p1_gt  # origin at txt wrist
            angle_deg, cos_sim = angle_between(v_gt, v_mp)
        else:
            angle_deg, cos_sim = None, None

        cand = {
            "wrist_px": info["wrist_px"],   # MP's wrist (for orange debug only)
            "tip_px": info["tip_px"],       # fingertip pixel
            "label": info["label"],         # MP handedness guess
            "angle_deg": angle_deg,
            "cos_sim": cos_sim,
        }
        candidates.append(cand)

        # Only use candidates with valid angle for "best" selection
        if angle_deg is not None:
            if best_angle is None or angle_deg < best_angle:
                best_angle = angle_deg
                best_cos = cos_sim
                best_idx = len(candidates) - 1

    if not candidates:
        return None

    # Annotate
    if annot_dir is not None:
        out_path = os.path.join(annot_dir, f"{name}_annotated.jpg")
        annotate_image(
            image_bgr=image_bgr,
            p1_px=p1_px,
            p2_px=p2_px,
            hand_candidates=candidates,
            best_idx=best_idx,
            out_path=out_path,
            name=name,
        )

    # If no candidate had valid 3D, we don't return a metric result, but we still
    # wrote an annotated image; that's useful for debugging.
    if best_idx is None:
        return None

    best = candidates[best_idx]
    return {
        "name":      name,
        "angle_deg": best_angle,
        "cos_sim":   best_cos,
        "num_cands": len(candidates),
        "mp_label":  best["label"],
    }


def process_dataset(base_path, fx, fy, cx, cy, annot_dir=None):
    """Loop over jpgs (up to MAX_PAIRS) and process."""
    names = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(base_path)
        if f.endswith(IMAGE_EXT)
    )
    if MAX_PAIRS is not None and len(names) > MAX_PAIRS:
        names = names[:MAX_PAIRS]

    results = []
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    ) as hands_detector:
        for i, name in enumerate(names, start=1):
            res = process_single_triplet(base_path, name, fx, fy, cx, cy,
                                         hands_detector, annot_dir)
            if res is not None:
                results.append(res)
            if i % 50 == 0:
                print(f"Attempted {i} images, valid pointing samples so far: {len(results)}")
    return results


def summarize_results(results):
    if not results:
        print("No valid pointing samples were processed.")
        return

    angles = np.array([r["angle_deg"] for r in results], dtype=float)
    cos_sims = np.array([r["cos_sim"] for r in results], dtype=float)

    print("\n========== SUMMARY ==========")
    print(f"Valid pointing samples     : {len(results)}")
    print(f"Angle (deg): mean = {angles.mean():.3f}, "
          f"median = {np.median(angles):.3f}, "
          f"std = {angles.std():.3f}")
    print(f"|cosine|  : mean = {np.abs(cos_sims).mean():.3f}")
    print("=============================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Mediapipe fingertip vectors (origin at txt wrist) to GT, "
                    "choose the smallest-angle candidate, and draw all lines from the "
                    "txt wrist (GT line in red). Uses Azure Kinect 3D->2D if calibration "
                    "is provided, otherwise falls back to intrinsics."
    )
    parser.add_argument(
        "--data_dir",
        "-d",
        type=str,
        default=DATA_DIR,
        help=f"Directory containing <name>{IMAGE_EXT}, <name>{DEPTH_EXT}, <name>{LABEL_EXT} (default: {DATA_DIR})",
    )
    args = parser.parse_args()

    base_path = args.data_dir
    if not os.path.isdir(base_path):
        raise NotADirectoryError(f"{base_path} is not a valid directory")

    annot_dir = os.path.join(base_path, ANNOTATION_SUBDIR)
    os.makedirs(annot_dir, exist_ok=True)

    print(f"Using data directory: {base_path}")
    print(f"Annotation directory: {annot_dir}")
    print(f"Camera intrinsics: fx={FX}, fy={FY}, cx={CX}, cy={CY}")
    print(f"Max jpg-based triplets to attempt: {MAX_PAIRS}")
    if HAVE_PYK4A and K4A_CALIBRATION is not None:
        print("Azure Kinect calibration: ENABLED for 3D->2D projection")
    else:
        print("Azure Kinect calibration: NOT SET (using plain intrinsics)")

    results = process_dataset(base_path, FX, FY, CX, CY, annot_dir=annot_dir)
    summarize_results(results)

    for r in results[:10]:
        print(f"{r['name']}: angle = {r['angle_deg']:.2f} deg, "
              f"cos_sim = {r['cos_sim']:.3f}, "
              f"mp_label = {r['mp_label']}, "
              f"num_cands = {r['num_cands']}")


if __name__ == "__main__":
    main()
