# Process directory of images to compare distance to correct tag
# Paired t-test for continuous distance metric

import os
import numpy as np
import cv2
from find_tag import get_detections
from pose_estimate import decompose_homography
import json
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS, CalibrationType
import mediapipe as mp
import torch
from torchvision import models
import torch.nn.functional as F
from scipy import stats
from stats_angular_error import run_model, pixel_to_3d

Y_MAX = 1080
X_MAX = 1920
HALF_SIDE_M = 0.10

def find_distance_to_tag(wrist_world, direction_world, tag_id, transformation_map):
    """Find distance from ray intersection to specific tag on z=0 plane"""
    x_w, y_w, z_w = wrist_world
    vx, vy, vz = direction_world
    
    if abs(vz) < 1e-6:
        return None
    
    # Intersect with z=0 plane
    t = -z_w / vz
    x_intersect = x_w + vx * t
    y_intersect = y_w + vy * t
    
    # Get tag position
    tag_matrix = transformation_map[tag_id]
    tag_x = tag_matrix[0, 3]
    tag_y = tag_matrix[1, 3]
    
    # Calculate 2D distance on plane
    dist = np.sqrt((x_intersect - tag_x)**2 + (y_intersect - tag_y)**2)
    
    return dist

# ============================================================================
# SETUP
# ============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config(
    color_resolution=ColorResolution.RES_1080P,
    depth_mode=DepthMode.NFOV_UNBINNED,
    synchronized_images_only=True,
    camera_fps=FPS.FPS_15
)
k4a = PyK4A(cfg)
k4a.start()
calib = k4a.calibration
k4a.stop()

model = models.resnet101()
model.fc = torch.nn.Linear(model.fc.in_features, 4)
state_dict = torch.load(
    "trained_models/ResNet101_augFalse_ampTrue_h_flip_2025-12-12 19:03.pth", 
    map_location="cpu"
)["model_state_dict"]
model.load_state_dict(state_dict, strict=True)
model.to(device).eval()

with open("transformation_map.json", "r") as f:
    loaded = json.load(f)

transformation_map = {int(k): np.array(v) for k, v in loaded.items()}

# Zero out z-coordinates (all tags on same plane)
for matrix in transformation_map.values():
    matrix[2, 3] = 0.0

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# ============================================================================
# PROCESS DIRECTORY
# ============================================================================

DIRECTORY = "grid_data"

geometric_distances = []
predicted_distances = []

for file in sorted(os.listdir(DIRECTORY)):
    if not file.endswith(".txt"):
        continue
    
    print(f"Processing {file}...")
    
    base_name = file[:-4]
    txt_path = os.path.join(DIRECTORY, base_name + ".txt")
    img_path = os.path.join(DIRECTORY, base_name + ".jpg")
    depth_path = os.path.join(DIRECTORY, base_name + ".npy")
    
    # Load data
    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth_in_color = np.load(depth_path)
    
    # Load label file
    with open(txt_path, "r") as f:
        lines = f.readlines()
        label = int(lines[0].strip())
        
        if label != 1:
            continue
        
        # Wrist position in meters (lines 1-3)
        wrist_m = np.array([
            float(lines[1].strip()),
            float(lines[2].strip()),
            float(lines[3].strip())
        ])
        
        # Correct tag ID (line 7)
        correct_tag = int(float(lines[7].strip()))
    
    # ========================================================================
    # Get camera_to_world transformation
    # ========================================================================
    
    detections = get_detections(rgb)
    H = detections[0].homography.astype(np.float64)
    detected_tag_id = detections[0].tag_id
    
    tag_to_world = transformation_map[detected_tag_id]
    tag_to_camera = decompose_homography(H)
    camera_to_tag = np.linalg.inv(tag_to_camera)
    camera_to_world = tag_to_world @ camera_to_tag
    
    # ========================================================================
    # METHOD 1: GEOMETRIC BASELINE (wrist -> index finger)
    # ========================================================================
    
    result_mp = pose.process(rgb)
    if result_mp.pose_landmarks is None:
        print("FAILED: No pose landmarks detected")
        continue
    landmarks = result_mp.pose_landmarks.landmark

    left_finger = landmarks[mp.solutions.pose.PoseLandmark.LEFT_INDEX]
    
    finger_x = int(np.clip(left_finger.x * rgb.shape[1], 0, rgb.shape[1] - 1))
    finger_y = int(np.clip(left_finger.y * rgb.shape[0], 0, rgb.shape[0] - 1))
    
    depth_at_finger = depth_in_color[finger_y, finger_x]
    
    if depth_at_finger > 0:
        finger_3d_m = pixel_to_3d(finger_x, finger_y, depth_at_finger)
        finger_3d_m = np.array(finger_3d_m)
        
        # Geometric direction: wrist -> finger
        geo_direction_cam = finger_3d_m - wrist_m
        geo_direction_cam = geo_direction_cam / np.linalg.norm(geo_direction_cam)
    
    # ========================================================================
    # METHOD 2: DEEP LEARNING MODEL
    # ========================================================================
    
    conf, pred_direction_cam = run_model(model, bgr)
    
    # ========================================================================
    # CONVERT TO WORLD FRAME AND FIND DISTANCES TO CORRECT TAG
    # ========================================================================
    
    # Convert wrist to world frame (in tag units)
    wrist_tag_units = wrist_m / HALF_SIDE_M
    wrist_world_hom = np.array([*wrist_tag_units, 1.0])
    wrist_world = (camera_to_world @ wrist_world_hom)[:3]
    
    # Convert directions to world frame
    rotation_matrix = camera_to_world[:3, :3]
    geo_direction_world = rotation_matrix @ geo_direction_cam
    pred_direction_world = rotation_matrix @ pred_direction_cam
    
    # Find distance to CORRECT tag (not closest tag)
    geo_dist = find_distance_to_tag(
        wrist_world, geo_direction_world, correct_tag, transformation_map
    )
    
    pred_dist = find_distance_to_tag(
        wrist_world, pred_direction_world, correct_tag, transformation_map
    )
    
    if geo_dist is None or pred_dist is None:
        print("FAILED: Could not compute distances")
        continue
    
    # Convert to centimeters for reporting
    geo_dist_cm = geo_dist * HALF_SIDE_M * 100
    pred_dist_cm = pred_dist * HALF_SIDE_M * 100
    
    print(f"  Tag {correct_tag}: Geo={geo_dist_cm:.2f} cm, Model={pred_dist_cm:.2f} cm")
    
    geometric_distances.append(geo_dist_cm)
    predicted_distances.append(pred_dist_cm)

# ============================================================================
# STATISTICAL TESTS
# ============================================================================

geo = np.array(geometric_distances)
pred = np.array(predicted_distances)

n = len(geo)
geo_mean = np.mean(geo)
geo_std = np.std(geo, ddof=1)
pred_mean = np.mean(pred)
pred_std = np.std(pred, ddof=1)

# Paired t-test
t_stat, p_value_t = stats.ttest_rel(pred, geo)

# Wilcoxon signed-rank test (non-parametric alternative)
wilcoxon_stat, p_value_w = stats.wilcoxon(pred, geo)

# Mean difference and 95% CI
diff = pred - geo
mean_diff = np.mean(diff)
se_diff = stats.sem(diff)
ci_95 = stats.t.interval(0.95, len(diff)-1, loc=mean_diff, scale=se_diff)

# Effect size (Cohen's d)
cohens_d = mean_diff / np.std(diff, ddof=1)

print("\n" + "="*70)
print("DISTANCE TO CORRECT TAG RESULTS")
print("="*70)
print(f"Samples: {n}")
print(f"\nGeometric Baseline:")
print(f"  Mean ± SD: {geo_mean:.2f} ± {geo_std:.2f} cm")
print(f"  Range: [{geo.min():.2f}, {geo.max():.2f}] cm")

print(f"\nDeep Learning Model:")
print(f"  Mean ± SD: {pred_mean:.2f} ± {pred_std:.2f} cm")
print(f"  Range: [{pred.min():.2f}, {pred.max():.2f}] cm")

print(f"\nDifference (Model - Geometric):")
print(f"  Mean difference: {mean_diff:.2f} cm")
print(f"  95% CI: [{ci_95[0]:.2f}, {ci_95[1]:.2f}] cm")
print(f"  Cohen's d: {cohens_d:.3f}")

print(f"\nPaired t-test:")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  df: {n-1}")
print(f"  p-value: {p_value_t:.6f}")

print(f"\nWilcoxon signed-rank test (robustness check):")
print(f"  W-statistic: {wilcoxon_stat:.1f}")
print(f"  p-value: {p_value_w:.6f}")

print("\n" + "="*70)
if p_value_t < 0.001:
    if mean_diff < 0:
        print("*** Model has HIGHLY significantly lower distance (p < 0.001)")
    else:
        print("*** Geometric has HIGHLY significantly lower distance (p < 0.001)")
elif p_value_t < 0.01:
    if mean_diff < 0:
        print("**  Model has significantly lower distance (p < 0.01)")
    else:
        print("**  Geometric has significantly lower distance (p < 0.01)")
elif p_value_t < 0.05:
    if mean_diff < 0:
        print("*   Model has significantly lower distance (p < 0.05)")
    else:
        print("*   Geometric has significantly lower distance (p < 0.05)")
else:
    print(f"    No significant difference (p = {p_value_t:.4f})")

# Effect size interpretation
print(f"\nEffect size interpretation:")
if abs(cohens_d) < 0.2:
    print(f"  |d| = {abs(cohens_d):.3f}: Negligible effect")
elif abs(cohens_d) < 0.5:
    print(f"  |d| = {abs(cohens_d):.3f}: Small effect")
elif abs(cohens_d) < 0.8:
    print(f"  |d| = {abs(cohens_d):.3f}: Medium effect")
else:
    print(f"  |d| = {abs(cohens_d):.3f}: Large effect")
print("="*70)