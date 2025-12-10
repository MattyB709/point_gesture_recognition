#!/usr/bin/env python3

# run by doing python3 geo_validition.py <root_dir_of_val_data>

# green is geometric vector
# the magenta is the original vector from the april tags.

import os
import glob
import numpy as np
import cv2
import mediapipe as mp

# === Camera intrinsics (pixels) for the color camera ===
FX = 919.76178
FY = 919.8909
CX = 962.6875
CY = 550.9944

K_COLOR = np.array([
    [FX,    0.0, CX],
    [0.0,   FY,  CY],
    [0.0,   0.0, 1.0],
], dtype=np.float32)


def pixel_to_camera_mm(u, v, depth_map, K):
    """
    Convert pixel (u, v) + depth_map[v, u] (in mm)
    -> 3D point in camera frame (mm).

    Returns:
        np.array([X, Y, Z]) in mm, or None if depth is invalid.
    """
    h, w = depth_map.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return None

    Z = float(depth_map[v, u])  # mm
    if Z <= 0.0:
        return None

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    return np.array([X, Y, Z], dtype=np.float32)


def camera_to_pixel_mm(point_mm, K):
    """
    Project 3D camera-frame point (X,Y,Z) in mm to pixel (u,v).
    """
    X, Y, Z = point_mm
    if Z <= 0:
        return None

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return np.array([u, v], dtype=np.float32)


def draw_vector_from_wrist_mm(img_bgr, K, wrist_mm, vec_unit, length_mm,
                              color, thickness=2):
    """
    Draws a ray starting at wrist_mm along vec_unit for length_mm (all in mm),
    projected to image using intrinsics K, onto img_bgr with given color.
    """
    start_px = camera_to_pixel_mm(wrist_mm, K)
    end_mm = wrist_mm + vec_unit * float(length_mm)
    end_px = camera_to_pixel_mm(end_mm, K)

    if start_px is None or end_px is None:
        return

    p0 = (int(start_px[0]), int(start_px[1]))
    p1 = (int(end_px[0]), int(end_px[1]))
    cv2.line(img_bgr, p0, p1, color, thickness)


def load_original_pose(orig_txt_path):
    """
    Load original GT wrist and vector from <base>.txt.

    Expected format:
      line 1: 0           -> not pointing
        OR
      line 1: 1
      lines 2–4: wrist (x,y,z) in meters
      lines 5–7: vec (x,y,z) normalized (or nearly)

    Returns:
        status, orig_wrist_m, orig_vec

        status = 0   -> not pointing (first line is '0')
        status = 1   -> pointing and both wrist + vec parsed
        status = -1  -> invalid / cannot parse
    """
    if not os.path.exists(orig_txt_path):
        return -1, None, None

    with open(orig_txt_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]

    if len(lines) == 0:
        return -1, None, None

    # Not pointing case
    if lines[0] == "0":
        return 0, None, None

    # Expect at least 7 lines for 1 + 3 (wrist) + 3 (vector)
    if len(lines) < 7:
        return -1, None, None

    try:
        wx = float(lines[1])
        wy = float(lines[2])
        wz = float(lines[3])

        vx = float(lines[4])
        vy = float(lines[5])
        vz = float(lines[6])
    except ValueError:
        return -1, None, None

    orig_wrist_m = np.array([wx, wy, wz], dtype=np.float32)
    orig_vec = np.array([vx, vy, vz], dtype=np.float32)
    return 1, orig_wrist_m, orig_vec


def main(root_dir="val"):
    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Output folders
    out_txt_dir = os.path.join(root_dir, "geo_txt")
    out_vis_dir = os.path.join(root_dir, "geo_vis")
    os.makedirs(out_txt_dir, exist_ok=True)
    os.makedirs(out_vis_dir, exist_ok=True)

    jpg_paths = sorted(glob.glob(os.path.join(root_dir, "*.jpg")))
    print(f"Found {len(jpg_paths)} .jpg files in {root_dir}")

    for img_path in jpg_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        depth_path = os.path.join(root_dir, base + ".npy")
        orig_txt_path = os.path.join(root_dir, base + ".txt")
        out_txt_path = os.path.join(out_txt_dir, f"geo_{base}.txt")
        out_img_path = os.path.join(out_vis_dir, f"geo_{base}.jpg")

        if not os.path.exists(depth_path):
            print(f"[SKIP] No depth .npy for {base}")
            continue

        # --- Handle original GT txt (0 / 1 / invalid) ---
        status, orig_wrist_m, orig_vec = load_original_pose(orig_txt_path)

        if status == 0:
            # Not pointing: geo_<base>.txt should just have a single 0
            with open(out_txt_path, "w") as fgeo:
                fgeo.write("0\n")
            print(f"[OK] {base}: original is 0 -> wrote {out_txt_path} with '0'")
            # No drawing for non-pointing
            continue

        if status == -1:
            print(f"[SKIP] {base}: invalid or missing original txt")
            continue

        # status == 1 and both orig_wrist_m + orig_vec are available

        # Load RGB image
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[SKIP] Cannot read image {img_path}")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img = rgb.shape[:2]

        # Load depth map (assumed mm)
        depth = np.load(depth_path)
        if depth.ndim != 2:
            print(f"[SKIP] Depth map {depth_path} is not 2D")
            continue

        h_d, w_d = depth.shape

        # Run MediaPipe Pose
        results = pose.process(rgb)
        if not results.pose_landmarks:
            print(f"[SKIP] No pose detected for {base}")
            continue

        lms = results.pose_landmarks.landmark
        lw = lms[mp_pose.PoseLandmark.LEFT_WRIST]
        li = lms[mp_pose.PoseLandmark.LEFT_INDEX]

        # 2D coords in the RGB image
        u_w = int(lw.x * w_img)
        v_w = int(lw.y * h_img)
        u_i = int(li.x * w_img)
        v_i = int(li.y * h_img)

        if not (0 <= u_w < w_img and 0 <= v_w < h_img and
                0 <= u_i < w_img and 0 <= v_i < h_img):
            print(f"[SKIP] Landmarks out of bounds for {base}")
            continue

        # Map to depth resolution if needed
        u_w_d = int(u_w * w_d / w_img)
        v_w_d = int(v_w * h_d / h_img)
        u_i_d = int(u_i * w_d / w_img)
        v_i_d = int(v_i * h_d / h_img)

        # Back-project to 3D in camera frame (mm)
        wrist_cam_mm = pixel_to_camera_mm(u_w_d, v_w_d, depth, K_COLOR)
        finger_cam_mm = pixel_to_camera_mm(u_i_d, v_i_d, depth, K_COLOR)

        if wrist_cam_mm is None or finger_cam_mm is None:
            print(f"[SKIP] Invalid depth for {base}")
            continue

        # Geo vector: wrist -> index (normalized)
        geo_vec = finger_cam_mm - wrist_cam_mm
        geo_norm = np.linalg.norm(geo_vec)
        if geo_norm < 1e-6:
            print(f"[SKIP] Zero-length geo vector for {base}")
            continue
        geo_vec_normalized = geo_vec / geo_norm

        # Normalize original vector in case it's slightly off
        orig_norm = np.linalg.norm(orig_vec)
        if orig_norm < 1e-6:
            print(f"[SKIP] Zero-length original vector for {base}")
            continue
        orig_vec_normalized = orig_vec / orig_norm

        # Angular error (degrees) between original & geo vectors
        cos_val = float(np.dot(orig_vec_normalized, geo_vec_normalized))
        cos_val = max(min(cos_val, 1.0), -1.0)
        angle_rad = np.arccos(cos_val)
        angle_deg = float(np.degrees(angle_rad))

        # Wrist point in meters for output
        wrist_cam_m = wrist_cam_mm / 1000.0  # mm -> m

        # ---- Write geo_<name>.txt into geo_txt/ ----
        with open(out_txt_path, "w") as f:
            # Line 1: 1
            f.write("1\n")
            # Lines 2–4: wrist point (xyz_1) in meters (geo wrist)
            f.write(f"{wrist_cam_m[0]:.6f}\n")
            f.write(f"{wrist_cam_m[1]:.6f}\n")
            f.write(f"{wrist_cam_m[2]:.6f}\n")
            # Lines 5–7: normalized geo vector (xyz_2)
            f.write(f"{geo_vec_normalized[0]:.6f}\n")
            f.write(f"{geo_vec_normalized[1]:.6f}\n")
            f.write(f"{geo_vec_normalized[2]:.6f}\n")
            # Line 8: angular error (degrees)
            f.write(f"{angle_deg:.6f}\n")

        # ---- Visualization ----
        # 1) Original wrist/index in image space
        cv2.circle(bgr, (u_w, v_w), 5, (0, 255, 0), -1)   # wrist: green dot
        cv2.circle(bgr, (u_i, v_i), 5, (0, 0, 255), -1)   # index: red dot
        cv2.line(bgr, (u_w, v_w), (u_i, v_i), (255, 0, 0), 2)  # blue line: 2D wrist->index

        # 2) Geometric normalized vector (green) from geo wrist
        GEO_LEN_MM = 300.0  # length of drawn ray
        draw_vector_from_wrist_mm(
            bgr, K_COLOR, wrist_cam_mm, geo_vec_normalized,
            length_mm=GEO_LEN_MM, color=(0, 255, 0), thickness=2
        )

        # 3) Original normalized vector (magenta) from original wrist
        #    Convert original wrist (meters) -> mm for projection
        orig_wrist_mm = orig_wrist_m * 1000.0
        draw_vector_from_wrist_mm(
            bgr, K_COLOR, orig_wrist_mm, orig_vec_normalized,
            length_mm=GEO_LEN_MM, color=(255, 0, 255), thickness=2
        )

        cv2.imwrite(out_img_path, bgr)

        print(f"[OK] {base}: wrote {out_txt_path} (angle={angle_deg:.2f} deg) "
              f"and {out_img_path}")

    pose.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate geometric wrist->index vectors, compare to original, "
                    "and draw both on images."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="val",
        help="Directory containing .jpg, .npy, and .txt pairs (default: ./val)",
    )
    args = parser.parse_args()
    main(args.root)
