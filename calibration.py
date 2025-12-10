import numpy as np
import cv2
from find_tag import get_detections
from pose_estimate import decompose_homography
import json
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS
from pyk4a import CalibrationType

if __name__
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("Detected AprilTags", cv2.WINDOW_NORMAL)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cfg = Config(
    color_resolution=ColorResolution.RES_1080P,       # 1920x1080
    depth_mode=DepthMode.NFOV_UNBINNED,               # 640x576 depth
    synchronized_images_only=True,                     # depth+color in same capture
    camera_fps= FPS.FPS_15
)
k4a = PyK4A(cfg)
k4a.start()
calib = k4a.calibration

# Store dictionary transformation_map mapping april tag id : rigid transformation (from tag -> world frame)
transformation_map = {}

# Loop through every frame of the video (if doing individual images, we'd just continually loop while images are taken)
while True:
    # ret, frame = cap.read()
    cap = k4a.get_capture()
    color_bgra = cap.color                # (1080,1920,4) BGRA uint8
    # if not ret:
    #     print("test")
    #     break
    if color_bgra is None:
        print("didn't work")
        continue
    frame = cv2.cvtColor(color_bgra, cv2.COLOR_BGRA2RGB)
    
    # Have dictionary of tag id : rigid transformation (tag -> camera) for all unknown tags
    unknown_map = {}

    # Have (id, camera -> tag transformation) for known tag, this is for video-ing
    known_to_camera = None


    detections = get_detections(frame)
    rgb = frame
    if detections is not None:
        for det in detections:
            corners = det.corners
            # Convert the corners to integer
            corners = corners.astype(int)
            # Draw the corners (a polygon) using polylines
            rgb = cv2.polylines(rgb, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imshow("Detected AprilTags", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # Loop through all the detected tags
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('t'):
        if detections is not None:
            for det in detections:
                # If trasnformation_map is empty, means we have not processed any tags yet
                if not transformation_map:
                    # Set the first tag we detect as the world frame (add it to the dictionary, map it to 4x4 identity matrix)
                    transformation_map[det.tag_id] = np.eye(4)
                    print("Established origin, id:", det.tag_id)

                # Determine the transformation from tag to camera
                H = det.homography.astype(np.float64)
                homogenous = decompose_homography(H)

                # If the tag id is in the dictionary, we know its world frame transformation has already been calculated
                if det.tag_id in transformation_map:
                    # Save the transformation (tag -> camera we know global transform for)
                    known_to_camera = (det.tag_id, homogenous)
                    print("Known tag found, id:", det.tag_id)
                # Else it's new tag
                else:
                    # Save the tag id & transformation (tag -> camera) in dict of tags we need to calculate world trasnform for
                    print(transformation_map)
                    unknown_map[det.tag_id] = homogenous
                    print("Saved unknown tag, id:", det.tag_id)
            
            if known_to_camera is None:
                print("No known tag found")
                continue

            # Loop through the dict of tags needing world transform
            for id, unknown_tag_to_camera in unknown_map.items():
                # Multiply tag -> camera transform by saved camera -> known tag transform, save as tag -> known tag
                camera_to_known = np.linalg.inv(known_to_camera[1])
                unknown_to_known = camera_to_known @ unknown_tag_to_camera
                
                # Multiply tag -> known tag by the known tag -> world frame stored in trasnformation_map (result is the trasnformation of tag -> world frame)
                unknown_to_world = transformation_map[known_to_camera[0]] @ unknown_to_known

                # Save result in trasnformation_map
                transformation_map[id] = unknown_to_world

# cap.release()

cv2.destroyAllWindows()
k4a.stop()



serializable_data = {k: v.tolist() for k, v in transformation_map.items()}

# save to JSON
with open("transformation_map.json", "w") as f:
    json.dump(serializable_data, f, indent=4)

print("Saved to transformation_map.json")

with open("transformation_map.json", "r") as f:
    loaded = json.load(f)

# Convert lists back to numpy arrays
loaded_data = {int(k): np.array(v) for k, v in loaded.items()}

def dicts_equal(d1, d2, tol=False):
    if d1.keys() != d2.keys():
        return False
    
    for k in d1:
        if tol:
            if not np.allclose(d1[k], d2[k]):
                return False
        else:
            if not np.array_equal(d1[k], d2[k]):
                return False
    return True

print(dicts_equal(loaded_data, transformation_map))