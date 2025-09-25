import numpy as np
import cv2
from find_tag import get_detections
from pose_estimate import decompose_homography
import json

# This script is to test the transformation_map.json to see if distances are correct

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("Detected AprilTags", cv2.WINDOW_NORMAL)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

half_side_m = 10 / 100.0  # meters per "tag unit" (tag family canonical: half-side = 1 unit)
with open("transformation_map.json", "r") as f:
    loaded = json.load(f)

# Convert lists back to numpy arrays
transformation_map = {int(k): np.array(v) for k, v in loaded.items()}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    detections = get_detections(frame)
    cv2.imshow("Detected AprilTags", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if detections is None:
        print("No tags detected")
        continue

    for det in detections:
        id = det.tag_id
        if id not in transformation_map:
            print(f"tag {id} not found in transformation map")
            continue
        H = det.homography.astype(np.float64)

        homogenous = transformation_map[id]
        tag_to_camera = decompose_homography(H)
        camera_to_tag = np.linalg.inv(tag_to_camera)
        camera_to_world = homogenous @ camera_to_tag
        t_cam = camera_to_world[:3, 3].copy()
        t_cam *= half_side_m

        print(f"Distance of cam to origin: {np.linalg.norm(t_cam)}")
        t = homogenous[:3, 3].copy()
        t *= half_side_m
        # print(f"Distance of {id} to origin: {np.linalg.norm(t)}")


cap.release()
