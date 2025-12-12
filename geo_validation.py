#!/usr/bin/env python3

# run by doing:
#   python3 geo_validation.py <root_dir_of_val_data> --start LEFT_WRIST --end LEFT_INDEX
# you can choose what media pipe joints to use for the geometric 
# vector, just change the --start and --end args.
# green is geometric vector (from chosen start->end MP joints)
# magenta is the original vector from the AprilTags.

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


def draw_vector_from_point_mm(img_bgr, K, start_mm, vec_unit, length_mm,
                              color, thickness=2):
    """
    Draws a ray starting at start_mm along vec_unit for length_mm (all in mm),
    projected to image using intrinsics K, onto img_bgr with given color.
    """
    start_px = camera_to_pixel_mm(start_mm, K)
    end_mm = start_mm + vec_unit * float(length_mm)
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


def compute_average_angular_error(txt_folder):
    """
    Equivalent of average_8th_line(folder), but returns (avg, count)
    and skips any AVERAGE_*.txt file.
    """
    total = 0.0
    count = 0

    for name in os.listdir(txt_folder):
        if not name.endswith(".txt"):
            continue
        if name.startswith("AVERAGE_"):
            # Don't re-include old summary files if re-running.
            continue

        path = os.path.join(txt_folder, name)

        try:
            with open(path, "r") as f:
                lines = f.readlines()
        except OSError as e:
            print(f"Skipping {name}: could not open file ({e})")
            continue

        if len(lines) < 8:
            # No 8th line (e.g., non-pointing or invalid file)
            continue

        raw = lines[7].strip()
        if not raw:
            continue

        try:
            value = float(raw)
        except ValueError:
            print(f"Skipping {name}: 8th line is not a number -> {raw!r}")
            continue

        total += value
        count += 1

    if count == 0:
        return None, 0

    avg = total / count
    return avg, count


def main(root_dir="val", start_joint_name="LEFT_WRIST", end_joint_name="LEFT_INDEX"):
    mp_pose = mp.solutions.pose

    # Resolve landmark enums from provided joint names.
    try:
        start_lmk = mp_pose.PoseLandmark[start_joint_name]
        end_lmk = mp_pose.PoseLandmark[end_joint_name]
    except KeyError:
        valid_names = [lm.name for lm in mp_pose.PoseLandmark]
        raise ValueError(
            f"Invalid joint name(s): start={start_joint_name}, end={end_joint_name}. "
            f"Valid names include: {valid_names}"
        )

    start_idx = start_lmk.value
    end_idx = end_lmk.value

    pair_tag = f"{start_joint_name}_TO_{end_joint_name}"
    print(f"Using joint pair: {pair_tag}")

    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Output folders now include the pair tag.
    pair_dir = os.path.join(root_dir, pair_tag)
    out_txt_dir = os.path.join(pair_dir, "geo_txt")
    out_vis_dir = os.path.join(pair_dir, "geo_vis")
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
        start_lm = lms[start_idx]
        end_lm = lms[end_idx]

        # 2D coords in the RGB image
        u_start = int(start_lm.x * w_img)
        v_start = int(start_lm.y * h_img)
        u_end = int(end_lm.x * w_img)
        v_end = int(end_lm.y * h_img)

        if not (0 <= u_start < w_img and 0 <= v_start < h_img and
                0 <= u_end < w_img and 0 <= v_end < h_img):
            print(f"[SKIP] Landmarks out of bounds for {base}")
            continue

        # Map to depth resolution if needed
        u_start_d = int(u_start * w_d / w_img)
        v_start_d = int(v_start * h_d / h_img)
        u_end_d = int(u_end * w_d / w_img)
        v_end_d = int(v_end * h_d / h_img)

        # Back-project to 3D in camera frame (mm)
        start_cam_mm = pixel_to_camera_mm(u_start_d, v_start_d, depth, K_COLOR)
        end_cam_mm = pixel_to_camera_mm(u_end_d, v_end_d, depth, K_COLOR)

        if start_cam_mm is None or end_cam_mm is None:
            print(f"[SKIP] Invalid depth for {base}")
            continue

        # Geo vector: start -> end (normalized)
        geo_vec = end_cam_mm - start_cam_mm
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

        # Start point in meters for output (geo start)
        start_cam_m = start_cam_mm / 1000.0  # mm -> m

        # ---- Write geo_<name>.txt into pair-specific geo_txt/ ----
        with open(out_txt_path, "w") as f:
            # Line 1: 1
            f.write("1\n")
            # Lines 2–4: start point (xyz_1) in meters (geo start joint)
            f.write(f"{start_cam_m[0]:.6f}\n")
            f.write(f"{start_cam_m[1]:.6f}\n")
            f.write(f"{start_cam_m[2]:.6f}\n")
            # Lines 5–7: normalized geo vector (xyz_2)
            f.write(f"{geo_vec_normalized[0]:.6f}\n")
            f.write(f"{geo_vec_normalized[1]:.6f}\n")
            f.write(f"{geo_vec_normalized[2]:.6f}\n")
            # Line 8: angular error (degrees)
            f.write(f"{angle_deg:.6f}\n")

        # ---- Visualization ----
        # 1) 2D start/end joints in image space
        cv2.circle(bgr, (u_start, v_start), 5, (0, 255, 0), -1)  # start: green dot
        cv2.circle(bgr, (u_end, v_end), 5, (0, 0, 255), -1)      # end: red dot
        cv2.line(bgr, (u_start, v_start), (u_end, v_end), (255, 0, 0), 2)  # blue line: 2D start->end

        # 2) Geometric normalized vector (green) from geo start
        GEO_LEN_MM = 300.0  # length of drawn ray
        draw_vector_from_point_mm(
            bgr, K_COLOR, start_cam_mm, geo_vec_normalized,
            length_mm=GEO_LEN_MM, color=(0, 255, 0), thickness=2
        )

        # 3) Original normalized vector (magenta) from original wrist
        #    Convert original wrist (meters) -> mm for projection
        orig_wrist_mm = orig_wrist_m * 1000.0
        draw_vector_from_point_mm(
            bgr, K_COLOR, orig_wrist_mm, orig_vec_normalized,
            length_mm=GEO_LEN_MM, color=(255, 0, 255), thickness=2
        )

        cv2.imwrite(out_img_path, bgr)

        print(f"[OK] {base}: wrote {out_txt_path} (angle={angle_deg:.2f} deg) "
              f"and {out_img_path}")

    # After processing all files, compute average angular error for this joint pair.
    avg, count = compute_average_angular_error(out_txt_dir)
    avg_file_name = f"AVERAGE_{pair_tag}.txt"
    avg_file_path = os.path.join(out_txt_dir, avg_file_name)

    with open(avg_file_path, "w") as f:
        if avg is None:
            f.write("No valid 8th-line values found.\n")
            print(f"[AVG] No valid 8th-line values found in {out_txt_dir}")
        else:
            f.write(f"Joint pair: {pair_tag}\n")
            f.write(f"Used {count} file(s).\n")
            f.write(f"Average angular error (deg): {avg:.6f}\n")
            print(f"[AVG] Wrote {avg_file_path} (count={count}, avg={avg:.6f} deg)")

    pose.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "Generate geometric vectors from chosen MediaPipe joints, compare to "
            "original AprilTag vectors, draw both on images, and compute average "
            "angular error for each joint pair."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="val",
        help="Directory containing .jpg, .npy, and original .txt files (default: ./val)",
    )
    parser.add_argument(
        "--start",
        default="LEFT_WRIST",
        help="MediaPipe PoseLandmark name for start joint "
             "(e.g., LEFT_WRIST, LEFT_ELBOW, RIGHT_WRIST, ...)",
    )
    parser.add_argument(
        "--end",
        default="LEFT_INDEX",
        help="MediaPipe PoseLandmark name for end joint "
             "(e.g., LEFT_INDEX, LEFT_WRIST, RIGHT_INDEX, ...)",
    )
    args = parser.parse_args()
    main(args.root, start_joint_name=args.start, end_joint_name=args.end)
