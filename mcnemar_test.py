# Process directory of images to compare predicted vs geometric tag classification
# For McNemar's test

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
from statsmodels.stats.contingency_tables import mcnemar
from stats_angular_error import run_model, pixel_to_3d

Y_MAX = 1080
X_MAX = 1920
HALF_SIDE_M = 0.10

def find_closest_tag_on_plane(wrist_world, direction_world, transformation_map):
    """Find closest tag by intersecting ray with z=0 plane"""
    x_w, y_w, z_w = wrist_world
    vx, vy, vz = direction_world
    
    if abs(vz) < 1e-6:
        return None, None
    
    t = -z_w / vz
    x_intersect = x_w + vx * t
    y_intersect = y_w + vy * t
    
    min_dist = float('inf')
    closest_id = None
    
    for tag_id, matrix in transformation_map.items():
        tag_x = matrix[0, 3]
        tag_y = matrix[1, 3]
        dist = np.sqrt((x_intersect - tag_x)**2 + (y_intersect - tag_y)**2)
        
        if dist < min_dist:
            min_dist = dist
            closest_id = tag_id
    
    return closest_id, min_dist

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

model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, 4)
state_dict = torch.load(
    "trained_models/ResNet50_augTrue_ampFalse_clean_data_2025-12-10 21:59.pth", 
    map_location="cpu"
)["model_state_dict"]
model.load_state_dict(state_dict, strict=True)
model.to(device).eval().to("cuda")

with open("transformation_map.json", "r") as f:
    loaded = json.load(f)

transformation_map = {int(k): np.array(v) for k, v in loaded.items()}

# Zero out z-coordinates (all tags on same plane)
for matrix in transformation_map.values():
    matrix[2, 3] = 0.0

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# ============================================================================
# PROCESS DIRECTORY
# ============================================================================

DIRECTORY = "grid_data"

geometric_correct = []
predicted_correct = []

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
    # CONVERT TO WORLD FRAME AND FIND CLOSEST TAGS
    # ========================================================================
    
    # Convert wrist to world frame (in tag units)
    wrist_tag_units = wrist_m / HALF_SIDE_M
    wrist_world_hom = np.array([*wrist_tag_units, 1.0])
    wrist_world = (camera_to_world @ wrist_world_hom)[:3]
    
    # Convert directions to world frame
    rotation_matrix = camera_to_world[:3, :3]
    geo_direction_world = rotation_matrix @ geo_direction_cam
    pred_direction_world = rotation_matrix @ pred_direction_cam
    
    # Find closest tags
    geo_tag_id, geo_dist = find_closest_tag_on_plane(
        wrist_world, geo_direction_world, transformation_map
    )
    
    pred_tag_id, pred_dist = find_closest_tag_on_plane(
        wrist_world, pred_direction_world, transformation_map
    )
    
    # Check correctness
    geo_correct = (geo_tag_id == correct_tag)
    pred_correct = (pred_tag_id == correct_tag)
    
    print(f"  True: {correct_tag}, Geo: {geo_tag_id} {'✓' if geo_correct else '✗'}, Model: {pred_tag_id} {'✓' if pred_correct else '✗'}")
    
    geometric_correct.append(1 if geo_correct else 0)
    predicted_correct.append(1 if pred_correct else 0)

# ============================================================================
# McNEMAR'S TEST
# ============================================================================

geo = np.array(geometric_correct)
pred = np.array(predicted_correct)

# Build contingency table
a = np.sum((geo == 1) & (pred == 1))  # Both correct
b = np.sum((geo == 1) & (pred == 0))  # Geo correct, model wrong
c = np.sum((geo == 0) & (pred == 1))  # Geo wrong, model correct
d = np.sum((geo == 0) & (pred == 0))  # Both wrong

table = np.array([[a, b], [c, d]])

geo_acc = np.mean(geo) * 100
pred_acc = np.mean(pred) * 100

print("\n" + "="*70)
print("TAG CLASSIFICATION RESULTS")
print("="*70)
print(f"Samples: {len(geo)}")
print(f"Geometric: {geo_acc:.1f}% ({geo.sum()}/{len(geo)})")
print(f"Model: {pred_acc:.1f}% ({pred.sum()}/{len(pred)})")
print(f"\nContingency Table:")
print(f"                     Model")
print(f"           Correct  Wrong")
print(f"Geo Correct   {a}      {b}")
print(f"Geo Wrong     {c}      {d}")

if b + c < 25:
    result = mcnemar(table, exact=True)
else:
    result = mcnemar(table, exact=False, correction=True)

print(f"\nMcNemar's Test:")
print(f"  Statistic: {result.statistic:.3f}")
print(f"  P-value: {result.pvalue:.4f}")

if result.pvalue < 0.05:
    winner = "Model" if c > b else "Geometric"
    print(f"  → {winner} is significantly better")
else:
    print(f"  → No significant difference")
print("="*70)