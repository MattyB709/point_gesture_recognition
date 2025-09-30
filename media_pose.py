import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

PATH = 'point_gesture_recognition/pose_landmarker.task'

# mediapipe object
base_options = python.BaseOptions(model_asset_path=PATH)
options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

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
  
    cap = cv2.VideoCapture(0)

    while True:
      ret, frame = cap.read()
      if not ret: break

      rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
      result = detector.detect(image)

      if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]  # first detected pose
        left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        # Coordinates are normalized (0–1 range)
        print(
            f"Left wrist (x={left_wrist.x}, y={left_wrist.y}, z={left_wrist.z}, visibility={left_wrist.visibility})"
            # x is horizontal axis, y is vertical (0,0 is at the top left corner)
            # y increases as you go down.
        )
      # draw simple landmarks
      annotated_image = draw_landmarks_on_image(image.numpy_view(), result)
      cv2.imshow("Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

    cap.release()
    cv2.destroyAllWindows()

  