import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS
from pyk4a import CalibrationType
import time
from find_tag import get_detections
from pose_estimate import decompose_homography
import json

PATH = 'pose_landmarker.task'

# mediapipe object
base_options = python.BaseOptions(model_asset_path=PATH)
options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
detector = vision.PoseLandmarker.create_from_options(options)
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
            continue

        corners = detections[0].corners

        # Convert the corners to integer
        corners = corners.astype(int)

        # Draw the corners (a polygon) using polylines
        rgb = cv2.polylines(rgb, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            pointed_to_id = int(input())

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

                if x < 1920 and x > 0 and y < 1080 and y > 0:
                    depth_point = depth_in_color[y,x]

                    if depth_point == 0:
                        continue
                    xmm, ymm, zmm = calib.convert_2d_to_3d((x, y), depth_point, 
                                                        CalibrationType.COLOR)
                    xm = xmm / 1000
                    ym = ymm / 1000
                    zm = zmm / 1000

                    # calculate vector between tag coordinates and wrist coordinates
                    v = np.array([t_pointed_to_tag_to_camera[0] * 1000 - xmm, t_pointed_to_tag_to_camera[1] * 1000 - ymm, t_pointed_to_tag_to_camera[2] * 1000 - zmm])
                    print(np.linalg.norm(v))
                    v = v / np.linalg.norm(v)

                    # calculate 3D point along vector ray from wrist coordinates
                    point_on_ray = np.array([xmm, ymm, zmm]) + (1500.0 * v)
                    try:
                        uv = calib.convert_3d_to_2d(point_on_ray, CalibrationType.COLOR, CalibrationType.COLOR)
                        camera_coords_calculated = tuple(map(int, uv))

                        uv = calib.convert_3d_to_2d((xmm, ymm, zmm), CalibrationType.COLOR, CalibrationType.COLOR)
                        camera_coords_wrist = tuple(map(int, uv))
                        cv2.line(rgb, camera_coords_wrist, camera_coords_calculated, (0, 255, 0), 2)
                    except Exception:
                        pass


                    # project wrist and calculated point back to 2D
                    # print(x,y)
                    # print(camera_coords_wrist)
                    # print(camera_coords_calculated)

                    # draw vector on 2D image connecting wrist and calculated point
        cv2.imshow("Image", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            


  