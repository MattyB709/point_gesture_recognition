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

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)
  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]
    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
       landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks])
    solutions.drawing_utils.draw_landmarks(annotated_image, 
                                           pose_landmarks_proto, 
                                           solutions.pose.POSE_CONNECTIONS, 
                                           solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image



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
      image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
      # result = detector.detect(image)
      result = pose.process(rgb)

      if result.pose_landmarks:
        # landmarks = result.pose_landmarks[0]  # first detected pose
        landmarks = result.pose_landmarks.landmark
        left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        # Coordinates are normalized (0–1 range)
        x,y = left_wrist.x, left_wrist.y
        # print(rgb.shape)
        # print(depth_in_color.shape)
        x *= rgb.shape[1]
        y *= rgb.shape[0]
        x,y = int(x), int(y)
        # print(x,y)
        rgb = cv2.circle(rgb, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

        if x < 1920 and x > 0 and y < 1080 and y > 0:

          depth_point = depth_in_color[y,x]

          if depth_point == 0:
            continue
          xmm, ymm, zmm = calib.convert_2d_to_3d((x, y), depth_point, 
                                              CalibrationType.COLOR)
        # time.sleep(1)
        
      # draw simple landmarks
        # annotated_image = mp.solutions.drawing_utils.draw_landmarks(rgb, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
      cv2.imshow("Image", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break


  