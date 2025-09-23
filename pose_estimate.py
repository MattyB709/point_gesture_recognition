import numpy as np
from find_tag import get_homographies, get_detections
import cv2

# cm per tag unit
cm2tag = 8.7


intrinsics = np.array([[919.76178, 0, 962.6875],
                       [0, 919.8909, 550.9944],
                       [0,0,1]])
    
inverse = np.linalg.inv(intrinsics)



# checking here
identity_check = np.matmul(intrinsics, inverse)
print(identity_check)



def get_full_T(homography): # input is numpy matrix
    t = np.matmul(inverse, homography)
    scale = np.linalg.norm(t[:3, 0])
    t = t / scale
    full_t = np.eye(4, dtype=np.float32)
    r_3 = np.cross(t[:, 0], t[:, 1])
    full_t[:3, 0] = t[:, 0]
    full_t[:3, 1] = t[:, 1]
    full_t[:3, 2] = r_3
    full_t[:3, 3] = t[:, 2]
    full_t *= (cm2tag/100.0)  # convert to meters
    return full_t


def reprojection_error(full_t, tag_size, corners_2d, K):
    """
    full_t : (4,4) tag->camera transform (from your get_full_T)
    tag_size: side length of the april tag in same unit as translation (meters)
    corners_2d: (4,2) detected corners in pixel coords, ordered to match 3D corners
    K : (3,3) intrinsics
    returns: mean_error, errors_per_corner, projected_points (4,2)
    """
    # define tag corners in tag frame (z=0). order: e.g. top-left, top-right, bottom-right, bottom-left
    s = tag_size 
    corners_3d = np.array([[-s, -s, 0.0],
                           [ s, -s, 0.0],
                           [ s,  s, 0.0],
                           [-s,  s, 0.0]], dtype=np.float32)  # adjust order to match your detector

    # Extract R and t from full_t (tag->camera)
    R = full_t[:3, :3]
    t = full_t[:3, 3].reshape(3, 1)

    # Project using K [R|t]
    P = np.hstack((R, t))  # 3x4
    corners_h = np.hstack((corners_3d, np.ones((4,1))))  # 4x4
    proj = (K @ (P @ corners_h.T)).T  # 4x3
    proj_pixels = (proj[:, :2].T / proj[:, 2]).T  # normalize

    # compute pixel errors
    corners_2d = np.asarray(corners_2d, dtype=np.float32).reshape(-1, 2)
    errors = np.linalg.norm(proj_pixels - corners_2d, axis=1)
    mean_err = errors.mean()

    return mean_err, errors, proj_pixels


def draw_validation(frame, proj_pixels, detected_corners, full_t, K):
    # draw projected corners (green) and detected corners (red)
    for p in proj_pixels:
        cv2.circle(frame, tuple(p.astype(int)), 4, (0,255,0), -1)  # projected
    for p in detected_corners:
        cv2.circle(frame, tuple(np.array(p).astype(int)), 4, (0,0,255), 2)  # detected

    # draw lines connecting projected corners
    cv2.polylines(frame, [proj_pixels.astype(int)], isClosed=True, color=(0,255,0), thickness=2)
    cv2.polylines(frame, [np.asarray(detected_corners).astype(int)], isClosed=True, color=(0,0,255), thickness=1)

    # draw coordinate axes (length in same units as tag_size; choose visually)
    R = full_t[:3,:3]
    t = full_t[:3,3].reshape(3,1)
    rvec, _ = cv2.Rodrigues(R)  # rotation vector for cv2
    tvec = t.flatten()

    axis_len = 0.25  # 5 cm, change to fit scene
    axis_3d = np.array([
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, axis_len]
    ], dtype=np.float32)

    imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, K, distCoeffs=None)
    imgpts = imgpts.reshape(-1,2).astype(int)
    origin = tuple(imgpts[0])
    cv2.line(frame, origin, tuple(imgpts[1]), (255,0,0), 2) # X in blue
    cv2.line(frame, origin, tuple(imgpts[2]), (0,255,0), 2) # Y in green
    cv2.line(frame, origin, tuple(imgpts[3]), (0,0,255), 2) # Z in red

    return frame


if __name__ == "__main__":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    homographies = None
    full_t = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = get_detections(frame)
        if detections is not None:
            for det in detections:
                corners = det.corners
                print(corners)
                h = det.homography
                full_t = get_full_T(h)
                err_mean, err_per_corner, proj_pixels = reprojection_error(full_t, cm2tag/100.0, corners, intrinsics)
                print(f"Tag {det.tag_id}: mean reproj error {err_mean:.2f} px")
                frame = draw_validation(frame, proj_pixels, corners, full_t, intrinsics)
        cv2.imshow("Detected AprilTags", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
 