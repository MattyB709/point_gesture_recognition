# 1. using a transformation map of a flat wall, put all tags on the 
# same plane by zeroing their z coordinate (~3-5 cm error,
#  should be fine for this experiment)
# 2. Compute wrist 3D coords and vector in camera's frame then convert to world frame
# 3. Find the point on the plane (z=0) that intersects with this vector
# 4. Find the closest tag to this intersection point based on x,y 
# euclidean distance 

import os
from datetime import datetime
import numpy as np
import cv2
from find_tag import get_detections
from pose_estimate import decompose_homography
import json
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS
from pyk4a import CalibrationType
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
from torchvision import models
from test_model_live import preprocess_exact
import torch.nn.functional as F
from collect_data import draw_detections

Y_MAX = 1080
X_MAX = 1920
CONF_THRESHOLD = 0.0
HALF_SIDE_M = 0.10  # same as your other file
device = "cuda" if torch.cuda.is_available() else "cpu"
state_dict = torch.load("train/best_model.pth", map_location="cpu")["model_state_dict"]
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 4)  # 1 for confidence + 3 for vector
model.load_state_dict(state_dict, strict=True)
model.to(device).eval()
torch.backends.cudnn.benchmark = True

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

PATH = 'pose_landmarker.task'
base_options = python.BaseOptions(model_asset_path=PATH)
options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
cfg = Config(
    color_resolution=ColorResolution.RES_1080P,       # 1920x1080
    depth_mode=DepthMode.NFOV_UNBINNED,               # 640x576 depth
    synchronized_images_only=True,                     # depth+color in same capture
    camera_fps= FPS.FPS_15
)
k4a = PyK4A(cfg)
k4a.start()
pointed_to_id = -1
half_side_m = 10 / 100.0  # meters per "tag unit" (tag family canonical: half-side = 1 unit)
with open("transformation_map.json", "r") as f:
    loaded = json.load(f)

# Convert lists back to numpy arrays
transformation_map = {int(k): np.array(v) for k, v in loaded.items()}

for matrix in transformation_map.values():
    matrix[2, 3] = 0.0  # zero z coordinate to put all tags on same plane

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

while True:
    cap = k4a.get_capture()          # blocking
    color_bgra = cap.color                # numpy uint8, shape (1080,1920,4) BGRA
    depth = cap.depth                # numpy uint16, shape (576,640), units = millimeters

    calib = k4a.calibration                    # pyk4a Calibration object (intrinsics+extrinsics)
    depth_in_color = cap.transformed_depth

    rgb = cv2.cvtColor(color_bgra, cv2.COLOR_BGRA2RGB)
    result = pose.process(rgb)
    detections = get_detections(rgb)
    if detections is not None:
        draw_detections(rgb, detections)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
            left_finger = landmarks[mp.solutions.pose.PoseLandmark.LEFT_INDEX]
            finger_x, finger_y = int(left_finger.x * rgb.shape[1]), int(left_finger.y * rgb.shape[0])
            # Coordinates are normalized (0–1 range)
            x,y = left_wrist.x, left_wrist.y
            x *= rgb.shape[1]
            y *= rgb.shape[0]
            x,y = int(x), int(y)
            rgb = cv2.circle(rgb, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
            if x < X_MAX and x > 0 and y < Y_MAX and y > 0:
                depth_point = depth_in_color[y,x]
                depth_point_finger = depth_in_color[finger_y, finger_x]
                if depth_point == 0:
                    continue
                xmm, ymm, zmm = calib.convert_2d_to_3d((x, y), depth_point, 
                                                    CalibrationType.COLOR)
                xm = xmm / 1000
                ym = ymm / 1000
                zm = zmm / 1000
                wrist_mm = np.array([xmm, ymm, zmm])

                finger_xmm, finger_ymm, finger_zmm = calib.convert_2d_to_3d((finger_x, finger_y), depth_point_finger,
                                                    CalibrationType.COLOR)
                finger_mm = np.array([finger_xmm, finger_ymm, finger_zmm])
                finger_mm = finger_mm - wrist_mm

                finger_mm = finger_mm / np.linalg.norm(finger_mm)
                with torch.no_grad():
                    inp = preprocess_exact(rgb).to(device)
                    out = model(inp)                           # (1,4)
                    conf = torch.sigmoid(out[:, :1]).item()    # scalar in [0,1]
                    vec = F.normalize(out[:, 1:], p=2, dim=1)[0].detach().cpu().numpy()  # (3,)

                if detections is not None and len(detections) > 0:
                    H = detections[0].homography.astype(np.float64)
                    id = detections[0].tag_id

                    tag_to_world = transformation_map[id]
                    tag_to_camera = decompose_homography(H)
                    camera_to_tag = np.linalg.inv(tag_to_camera)
                    camera_to_world = tag_to_world @ camera_to_tag

                    x_t, y_t, z_t = xm / HALF_SIDE_M, ym / HALF_SIDE_M, zm / HALF_SIDE_M
                    wrist_coords = np.array([x_t, y_t, z_t, 1])
                    wrist_coords_world = camera_to_world @ wrist_coords
                    x_w, y_w, z_w, _ = wrist_coords_world
                    # convert to tag coordinates

                    if conf > CONF_THRESHOLD:
                        end_mm_pred = wrist_mm + vec * 300
                        end_mm_finger = wrist_mm + finger_mm * 300
                        uv_wrist = calib.convert_3d_to_2d(tuple(wrist_mm.tolist()),
                                                        CalibrationType.COLOR, CalibrationType.COLOR)
                        uv_end = calib.convert_3d_to_2d(tuple(end_mm_pred.tolist()),
                                                        CalibrationType.COLOR, CalibrationType.COLOR)
                        if uv_wrist is not None and uv_end is not None:
                            p0 = tuple(map(int, uv_wrist))
                            p1 = tuple(map(int, uv_end))
                            cv2.line(rgb, p0, p1, (0, 255, 0), 2)
                            cv2.putText(rgb, f"conf={conf:.2f}", (p0[0]+6, p0[1]-6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 1, cv2.LINE_AA)
                        
                        uv_end_finger = calib.convert_3d_to_2d(tuple(end_mm_finger.tolist()),
                                                        CalibrationType.COLOR, CalibrationType.COLOR)
                        if uv_wrist is not None and uv_end_finger is not None:
                            p0 = tuple(map(int, uv_wrist))
                            p1 = tuple(map(int, uv_end_finger))
                            cv2.line(rgb, p0, p1, (255, 0, 0), 2)
                        
                        rotation_matrix = camera_to_world[:3, :3]
                        vec_world = rotation_matrix @ vec
                        vx, vy, vz = vec_world
                        finger_world = rotation_matrix @ finger_mm
                        vx_finger, vy_finger, vz_finger = finger_world

                        # avoid division by zero
                        if abs(vz) < 1e-6:
                            continue

                        t = -z_w / vz
                        x_intersect = x_w + vx * t
                        y_intersect = y_w + vy * t

                        t_finger = -z_w / vz_finger
                        x_intersect_finger = x_w + vx_finger * t_finger
                        y_intersect_finger = y_w + vy_finger * t_finger

                        min_dist_pred = float('inf')
                        min_dist_finger = float('inf')
                        closest_id_pred = -1
                        closest_id_finger = -1
                        for tag_id, matrix in transformation_map.items():
                            tag_x = matrix[0, 3]
                            tag_y = matrix[1, 3]
                            dist_pred = np.sqrt((x_intersect - tag_x)**2 + (y_intersect - tag_y)**2)
                            if dist_pred < min_dist_pred:
                                min_dist_pred = dist_pred
                                closest_id_pred = tag_id
                            dist_finger = np.sqrt((x_intersect_finger - tag_x)**2 + (y_intersect_finger - tag_y)**2)
                            if dist_finger < min_dist_finger:
                                min_dist_finger = dist_finger
                                closest_id_finger = tag_id
                        print(f"Predicted ID: {closest_id_pred} with distance {min_dist_pred*HALF_SIDE_M*100:.2f} cm")
                        print(f"Geometric ID: {closest_id_finger} with distance {min_dist_finger*HALF_SIDE_M*100:.2f} cm")
    cv2.imshow("frame", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) == ord('q'):
        break
