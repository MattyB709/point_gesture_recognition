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

OUTPUT_DIR = "data"   # <-- set this
DETECTION_RADIUS = 15 # find the min depth in a 10x10 pixel square
Y_MAX = 1080
X_MAX = 1920

os.makedirs(OUTPUT_DIR, exist_ok=True)

def _stamp():
    # month-day-time => 05-11-142530
    return datetime.now().strftime("%m-%d-%H%M%S")

def _save_sample(bgr_img, depth_in_color, label, start_dir_cam=None):
    """
    bgr_img: uint8 BGR image (1920x1080x3)
    depth_in_color: uint16 (1080x1920; you pass transformed_depth)
    label: 0 or 1
    start_dir_cam: tuple(start_xyz_m, normalized_dir) in camera frame, each np.array shape (3,), meters
    """
    base = os.path.join(OUTPUT_DIR, _stamp())
    img_path  = base + ".jpg"
    depth_path = base + ".npy"
    txt_path  = base + ".txt"

    # 1) image
    cv2.imwrite(img_path, bgr_img)

    # 2) depth (raw array)
    np.save(depth_path, depth_in_color)

    # 3) label file
    with open(txt_path, "w") as f:
        f.write(f"{int(label)}\n")
        if label == 1 and start_dir_cam is not None:
            start_tag, end_tag = start_dir_cam
            # EXACTLY 7 lines total: 1 + 6 numbers (each on its own line)
            nums = start_tag + list(end_tag.astype(float))
            for v in nums:
                f.write(f"{v:.6f}\n")
    print(f"[saved] {img_path}, {depth_path}, {txt_path}")

PATH = 'pose_landmarker.task'

# mediapipe object
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
print("input a tag to point at: ")
pointed_to_id = int(input())
if __name__ == "__main__":
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
  
    while True:
        cap = k4a.get_capture()          # blocking
        color = cap.color                # numpy uint8, shape (1080,1920,4) BGRA
        depth = cap.depth                # numpy uint16, shape (576,640), units = millimeters

        calib = k4a.calibration                    # pyk4a Calibration object (intrinsics+extrinsics)
        depth_in_color = cap.transformed_depth

        rgb = cv2.cvtColor(color, cv2.COLOR_BGRA2RGB)
        result = pose.process(rgb)
        detections = get_detections(rgb)
        
        if detections is None:
            cv2.imshow("Image", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            continue

        for det in detections:
            corners = det.corners

            # Convert the corners to integer
            corners = corners.astype(int)

            # Draw the corners (a polygon) using polylines
            rgb = cv2.polylines(rgb, [corners], isClosed=True, color=(0, 255, 0), thickness=2)


        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
            # Coordinates are normalized (0–1 range)
            x,y = left_wrist.x, left_wrist.y
            x *= rgb.shape[1]
            y *= rgb.shape[0]
            x,y = int(x), int(y)
            rgb = cv2.circle(rgb, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

            # if pointing to id is set
            if pointed_to_id != -1:
                # use first apriltag detection to get camera->world
                id = detections[0].tag_id
                if id not in transformation_map:
                    print(f"tag {id} not found in transformation map")
                    # pointed_to_id = -1
                    continue
                if pointed_to_id not in transformation_map:
                    print(f"pointed to tag {pointed_to_id} not in transformation map")
                    # pointed_to_id = -1
                    continue
                H = detections[0].homography.astype(np.float64)

                tag_to_world = transformation_map[id]
                tag_to_camera = decompose_homography(H)
                camera_to_tag = np.linalg.inv(tag_to_camera)
                camera_to_world = tag_to_world @ camera_to_tag
                world_to_camera = np.linalg.inv(camera_to_world)

                # get tag->world for pointed to tag from transformation_map and invert it to get world->tag
                pointed_to_tag_to_world = transformation_map[pointed_to_id]
                # pointed_to_id = -1
                pointed_to_tag_to_camera = world_to_camera @ pointed_to_tag_to_world

                # extract tag coordinates from T of tag->camera
                t_pointed_to_tag_to_camera = pointed_to_tag_to_camera[:3, 3].copy()
                t_pointed_to_tag_to_camera *= half_side_m

                if x < X_MAX and x > 0 and y < Y_MAX and y > 0:
                    # y0 = max(y-DETECTION_RADIUS, 0)
                    # y1 = min(y+DETECTION_RADIUS, Y_MAX)
                    # x0 = max(x-DETECTION_RADIUS, 0)
                    # x1 = min(x+DETECTION_RADIUS, X_MAX)
                    # sub_array = depth_in_color[y0:y1, x0:x1]
                    # depth_point = np.min(sub_array)
                    # index = np.argmin(sub_array, keepdims=True)
                    # dy,dx = np.unravel_index(index, sub_array.shape)
                    # x,y = x0+dx, y0+dy
                    depth_point = depth_in_color[y,x]
                    if depth_point == 0:
                        continue
                    xmm, ymm, zmm = calib.convert_2d_to_3d((x, y), depth_point, 
                                                        CalibrationType.COLOR)
                    xm = xmm / 1000
                    ym = ymm / 1000
                    zm = zmm / 1000

                    # calculate vector between tag coordinates and wrist coordinates
                    v = np.array([t_pointed_to_tag_to_camera[0] - xm, t_pointed_to_tag_to_camera[1] - ym, t_pointed_to_tag_to_camera[2] - zm])
                    norm = np.linalg.norm(v)
                    if norm > 1e-8:
                        v /= norm
                    else: 
                        print("issue with vector norm") 
                        continue

                    # calculate 3D point along vector ray from wrist coordinates
                    point_on_ray = np.array([xmm, ymm, zmm]) + (300* v)
                    try:
                        uv = calib.convert_3d_to_2d(point_on_ray, CalibrationType.COLOR, CalibrationType.COLOR)
                        camera_coords_calculated = tuple(map(int, uv))

                        uv = calib.convert_3d_to_2d((xmm, ymm, zmm), CalibrationType.COLOR, CalibrationType.COLOR)
                        camera_coords_wrist = tuple(map(int, uv))
                        cv2.line(rgb, camera_coords_wrist, camera_coords_calculated, (0, 255, 0), 2)
                    except Exception:
                        pass

                    

                    # draw vector on 2D image connecting wrist and calculated point
        cv2.imshow("Image", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            try:
                pointed_to_id = int(input("Enter tag id to point at: "))
            except Exception:
                print("Invalid tag id")
            continue
        elif key == ord('y'):
            # Save positive sample (requires valid wrist + vector this frame)
            bgr_to_save = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
            try:
                _save_sample(
                    bgr_to_save,
                    depth_in_color,
                    label=1,
                    start_dir_cam=(([xm,ym,zm],v))  # computed below
                )
            except NameError:
                print("Cannot save positive: no valid wrist/vector computed this frame.")
            continue
        elif key == ord('n'):
            # Save negative sample (image + depth + label 0)
            bgr_to_save = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
            _save_sample(
                bgr_to_save,
                depth_in_color,
                label=0,
                start_dir_cam=None
            )
            continue