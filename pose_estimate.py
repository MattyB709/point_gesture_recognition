import numpy as np
import cv2
from find_tag import get_detections

# --- camera intrinsics (make sure these are for the current frame size!) ---
K = np.array([[919.76178, 0,     962.6875],
              [0,        919.8909, 550.9944],
              [0,        0,        1]], dtype=np.float64)

Kinv = np.linalg.inv(K)


# Your physical tag half-side in meters: 8.7 cm per tag unit (half-side) => 0.087 m
half_side_m = 10 / 100.0  # meters per "tag unit" (tag family canonical: half-side = 1 unit)

# useful for if tag order is wrong, the correct order ended up being [3,2,1,0]. No longer necessary
def best_order(corners_proj, corners_det):
    """
    corners_proj: (4,2) projected in pixels
    corners_det : (4,2) detected in pixels (unknown order)
    returns: det_reordered (4,2), best_idx_order (list of 4), best_err
    """
    idx = [0,1,2,3]
    rotations = [idx[i:]+idx[:i] for i in range(4)]
    candidates = rotations + [list(reversed(r)) for r in rotations]

    best = (1e18, candidates[0])
    for order in candidates:
        det_ord = corners_det[order]
        err = np.mean(np.linalg.norm(corners_proj - det_ord, axis=1))
        if err < best[0]:
            best = (err, order)
    return corners_det[best[1]], best[1], best[0]

def decompose_homography(H):
    """
    Returns R (3x3) and t (3,) in *tag units*.
    Assumes the planar model used to generate H has Z=0 and tag corners (±1, ±1).
    """
    A = Kinv @ H  # 3x3

    a1 = A[:, 0]
    a2 = A[:, 1]
    a3 = A[:, 2]

    # common scale (use average of norms; robust under noise)
    lam1 = 1.0 / np.linalg.norm(a1)
    lam2 = 1.0 / np.linalg.norm(a2)
    lam = 0.5 * (lam1 + lam2)

    r1 = lam * a1
    r2 = lam * a2
    r3 = np.cross(r1, r2)

    R = np.column_stack((r1, r2, r3))

    # Orthonormalize via SVD and enforce det=+1
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    t = lam * a3  # translation in "tag units"

    homogenous = np.eye(4)
    homogenous[:3,:3] = R
    homogenous[:3, 3] = t

    return homogenous.astype(np.float64)

def reprojection_error_units(R, t, corners_2d, K):
    """
    R, t in TAG UNITS. Corners in TAG UNITS (±1).
    """
    # Canonical tag corners: half-side = 1 unit
    # corners_3d_units = np.array([[-1, -1, 0.0],
    #                              [ 1, -1, 0.0],
    #                              [ 1,  1, 0.0],
    #                              [-1,  1, 0.0]], dtype=np.float64)
    corners_3d_units = np.array([[-1, 1, 0.0],
                                 [ 1, 1, 0.0],
                                 [ 1,  -1, 0.0],
                                 [-1,  -1, 0.0]], dtype=np.float64)
    corners_3d_units *= half_side_m

    P = np.hstack((R, t.reshape(3,1)))  # 3x4
    corners_h = np.hstack((corners_3d_units, np.ones((4,1))))
    proj = (K @ (P @ corners_h.T)).T  # (4,3)
    proj_pixels = (proj[:, :2].T / proj[:, 2]).T

    corners_2d = np.asarray(corners_2d, dtype=np.float64).reshape(-1, 2)
    errors = np.linalg.norm(proj_pixels - corners_2d, axis=1)
    return errors.mean(), errors, proj_pixels

def draw_validation(frame, proj_pixels, detected_corners, R, t_units, K, half_side_m):
    # projected vs detected
    for p in proj_pixels:
        cv2.circle(frame, tuple(p.astype(int)), 4, (0,255,0), -1)  # projected (green)
    for p in detected_corners:
        cv2.circle(frame, tuple(np.array(p).astype(int)), 4, (0,0,255), 2)  # detected (red)

    cv2.polylines(frame, [proj_pixels.astype(int)], True, (0,255,0), 2)
    cv2.polylines(frame, [np.asarray(detected_corners).astype(int)], True, (0,0,255), 1)

    # axes in TAG UNITS (so origin/axes lengths match the pose units)
    axis_len_units = 0.5  # half a tag-half-side; tweak visually
    axis_3d_units = np.array([[0, 0, 0],
                              [axis_len_units, 0, 0],
                              [0, axis_len_units, 0],
                              [0, 0, axis_len_units]], dtype=np.float64)

    # convert t to meters only for cv2.projectPoints if you also convert the axis to meters
    t_m = t_units 
    axis_3d_m = axis_3d_units * half_side_m

    rvec, _ = cv2.Rodrigues(R)
    imgpts, _ = cv2.projectPoints(axis_3d_m, rvec, t_m, K, distCoeffs=None)
    imgpts = imgpts.reshape(-1,2).astype(int)
    origin = tuple(imgpts[0])
    cv2.line(frame, origin, tuple(imgpts[1]), (255,0,0), 2)   # X (blue)
    cv2.line(frame, origin, tuple(imgpts[2]), (0,255,0), 2)   # Y (green)
    cv2.line(frame, origin, tuple(imgpts[3]), (0,0,255), 2)   # Z (red)
    return frame

# given T_AB and T_CB, return T_AC, i.e. the rigid transformation from C to A's coordinates
def compose_transforms_through_B(T_A_B, T_C_B):
    T_B_C = np.linalg.inv(T_C_B)
    return T_A_B @ T_B_C

if __name__ == "__main__":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cv2.namedWindow("Detected AprilTags", cv2.WINDOW_NORMAL)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = get_detections(frame)
        tag_t = []
        if detections is not None:
            for det in detections:
                corners_px = det.corners  # shape (4,2), order should be TL, TR, BR, BL (verify!)
                H = det.homography.astype(np.float64)

                homogenous = decompose_homography(H, K, Kinv)
                tag_t.append(homogenous.copy())
                t_units *= half_side_m  # convert to meters
                # print("DISTANCE TO TAG: ", np.linalg.norm(t_units))


                # Option A: pure tag units (corners at ±1)
                err_mean, err_per_corner, proj_pixels = reprojection_error_units(R, t_units, corners_px, K)
                corners_px = np.asarray(corners_px, np.float64).reshape(-1,2)

                # Now compute the FINAL error with matched pairs
                # print(f"Tag {det.tag_id}: mean reproj error {err_mean:.2f} px")

                frame = draw_validation(frame, proj_pixels, corners_px, R, t_units, K, half_side_m)

        if len(tag_t) == 2:
            print("Relative pose between tags:")
            T_0_w = tag_t[0]
            T_1_w = tag_t[1]
            T_w_0 = np.linalg.inv(T_0_w)
            T_1_0 = T_w_0 @ T_1_w 
            print(np.linalg.norm(T_1_0[:3,3]) * half_side_m, "units apart")
        cv2.imshow("Detected AprilTags", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
 