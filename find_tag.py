import cv2
from pupil_apriltags import Detector
import numpy as np

at_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

def run_cam():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the AprilTags in the current frame
        detections = at_detector.detect(gray)

        # Loop through each detection and draw it
        for detection in detections:
            print(detection)
            # Get the corners of the detected tag
            corners = detection.corners

            # Convert the corners to integer
            corners = corners.astype(int)

            # Draw the corners (a polygon) using polylines
            frame = cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)

            # Optionally, you can mark the center as well
            center = tuple(map(int, detection.center))
            frame = cv2.circle(frame, center, 5, (0, 0, 255), -1)  # Red dot at center

        # Show the frame with the drawn tags
        cv2.imshow("Detected AprilTags", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_homographies(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = at_detector.detect(gray)
    if len(detections) == 0:
        return None
    homographies = []
    for detection in detections:
        homographies.append(detection.homography)
        
    return homographies

def get_detections(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = at_detector.detect(gray)
    if len(detections) == 0:
        return None
    return detections
