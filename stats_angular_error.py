import cv2
from pyk4a import CalibrationType
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS
import os
import numpy as np
import torch
import torchvision
from test_model_live import preprocess_exact
from train.metrics import angular_error
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy import stats

Y_MAX = 1080
X_MAX = 1920

def pixel_to_3d(color_x, color_y, depth_mm):
    """
    Convert color pixel + depth to 3D world coordinates.
    
    Use this when depth is transformed/registered to color camera space (1920x1080).
    
    Args:
        color_x: X pixel coordinate in color image
        color_y: Y pixel coordinate in color image
        depth_mm: Depth value at that pixel (millimeters)
        intrinsics: Dict with fx, fy, cx, cy
    
    Returns:
        (x_m, y_m, z_m) in meters, or None if invalid depth
    """
    # Check for invalid depth
    if depth_mm == 0 or depth_mm is None:
        return None
    
    # Convert depth to meters
    z_m = depth_mm / 1000.0
    
    fx = 919.76178
    fy = 919.8909
    cx = 962.6875
    cy = 550.9944
    # Pinhole camera model
    x_m = (color_x - cx) * z_m / fx
    y_m = (color_y - cy) * z_m / fy
    
    return (x_m, y_m, z_m)

def run_model(model, img):
    with torch.no_grad():
        input_tensor = preprocess_exact(cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)).to('cuda')
        output = model(input_tensor)
        output = output.cpu().squeeze()
        conf = torch.sigmoid(output[0]).item()
        pred_vec = output[1:].numpy()
        pred_vec = pred_vec / np.linalg.norm(pred_vec)
    return conf, pred_vec

def get_2d_points(calib, wrist_coords, vec):

    xmm, ymm, zmm = wrist_coords * 1000
    point_on_ray = np.array((xmm, ymm, zmm)) + (300 * vec) 
    try:
        uv = calib.convert_3d_to_2d(point_on_ray, CalibrationType.COLOR, CalibrationType.COLOR)
        camera_coords_calculated = tuple(map(int, uv))

        uv = calib.convert_3d_to_2d((xmm, ymm, zmm), CalibrationType.COLOR, CalibrationType.COLOR)
        camera_coords_wrist = tuple(map(int, uv))
        return camera_coords_calculated, camera_coords_wrist
    except Exception:
        print("draw line exception")
    return None, None

def get_index_coords(result, rgb):
    landmarks = result.pose_landmarks.landmark
    left_index = landmarks[mp.solutions.pose.PoseLandmark.LEFT_INDEX]
    # Coordinates are normalized (0–1 range)
    x,y = left_index.x, left_index.y
    x *= rgb.shape[1]
    y *= rgb.shape[0]
    x,y = int(x), int(y)
    return x,y

PATH = 'pose_landmarker.task'

if __name__ == "__main__":


    os.makedirs("check_data", exist_ok=True)
    cfg = Config(
        color_resolution=ColorResolution.RES_1080P,       # 1920x1080
        depth_mode=DepthMode.NFOV_UNBINNED,               # 640x576 depth
        synchronized_images_only=True,                     # depth+color in same capture
        camera_fps= FPS.FPS_15
    )

    k4a = PyK4A(cfg)
    k4a.start()
    calib = k4a.calibration                    # pyk4a Calibration object (intrinsics+extrinsics)
    k4a.stop()

    base_options = python.BaseOptions(model_asset_path=PATH)
    options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True
    )

    model = torchvision.models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    state_dict = torch.load("trained_models/ResNet50_augFalse_ampFalse_h_flip_2025-12-12 14:04.pth", map_location="cpu")["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.to("cuda").eval()
    
    total_angular_error = 0.0
    geo_angular_error = 0.0
    num_angular_samples = 0
    angular_errors_pred = []
    angular_errors_geo = []
    DIR = "split_data/val"
    for file in os.listdir(DIR):
        
        if not file.endswith(".txt"):
            continue
        print(f"Processing {file}...")
        txt_path = os.path.join(DIR, file)
        base_name = file[:-4]
        img_path = os.path.join(DIR, base_name + ".jpg")
        depth_path = os.path.join(DIR, base_name + ".npy")
        depth_in_color = np.load(depth_path)

        color_img = cv2.imread(img_path)
        
        # open and read text file to get coordinates/label
        with open(txt_path, "r") as f:
            lines = f.readlines()
            label = int(lines[0].strip())

            if label == 1:
                conf, pred_vec = run_model(model, color_img)

                rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                if not results.pose_landmarks:
                    print(f"No pose landmarks detected for {file}")
                    continue
                index_x, index_y = get_index_coords(results, rgb)
                if index_x > 0 and index_x < X_MAX and index_y > 0 and index_y < Y_MAX:
                    depth_point = depth_in_color[index_y, index_x]
                    if depth_point ==0:
                        print(f"Depth point at ({index_x}, {index_y}) is {depth_point} mm")
                        continue
                    xm,ym, zm = pixel_to_3d(index_x, index_y, depth_point)
                    # try:
                    #     xmm, ymm, zmm = calib.convert_2d_to_3d((index_x, index_y), depth_point, 
                    #                             CalibrationType.COLOR, CalibrationType.COLOR)
                    # except Exception as e:
                    #     print(f"Exception during 2D to 3D conversion 2D to 3D: {e}")
                    #     continue
                    # xm, ym, zm = xmm / 1000, ymm / 1000, zmm / 1000
                    index_xyz_m = np.array((xm, ym, zm))

                num_angular_samples += 1
                nums = [float(x.strip()) for x in lines[1:]]
                start_xyz_m = np.array(nums[:3])
                geo_vec = index_xyz_m - start_xyz_m
                geo_vec = geo_vec / np.linalg.norm(geo_vec)
                geo_point, geo_wrist = get_2d_points(calib, start_xyz_m, geo_vec)

                dir_vec = np.array(nums[3:6])
                cam_point, cam_wrist = get_2d_points(calib, start_xyz_m, dir_vec)

                if cam_point is not None and cam_wrist is not None:
                    cv2.circle(color_img, cam_wrist, 10, (0, 255, 0), -1)  # Green for wrist
                    cv2.line(color_img, cam_wrist, cam_point, (0, 0, 255), 5)  # Red for pointing direction
                cam_point_pred, cam_wrist_pred = get_2d_points(calib, start_xyz_m, pred_vec)

                if cam_point_pred is not None and cam_wrist_pred is not None:
                    cv2.line(color_img, cam_wrist_pred, cam_point_pred, (255, 255, 0), 5)  # Cyan for predicted direction   
                    error = angular_error(torch.from_numpy(dir_vec).unsqueeze(0), torch.from_numpy(pred_vec).unsqueeze(0))
                    angular_errors_pred.append(error)
                    total_angular_error += error
                    cv2.putText(color_img, f"Error: {error:.1f} deg", (650, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

                if geo_point is not None and geo_wrist is not None:
                    error = angular_error(torch.from_numpy(dir_vec).unsqueeze(0), torch.from_numpy(geo_vec).unsqueeze(0))   
                    geo_angular_error += error
                    angular_errors_geo.append(error)
                    cv2.line(color_img, geo_wrist, geo_point, (255, 0, 0), 5)  # Red for pointing direction
                    cv2.putText(color_img, f"Error: {error:.1f} deg", (650, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)


        cv2.imwrite(f"check_data/{base_name}.jpg", color_img)
        key = cv2.waitKey(0)
        if key == 27:  # ESC to exit
            break
    if num_angular_samples > 0:
        avg_angular_error = total_angular_error / num_angular_samples
        avg_geo_angular_error = geo_angular_error / num_angular_samples
        t_stat, p_value = stats.ttest_rel(angular_errors_pred, angular_errors_geo)
        print(f"T-statistic: {t_stat}, P-value: {p_value}")
        print(f"Average Predicted Angular Error over {num_angular_samples} samples: {avg_angular_error:.2f} degrees")
        print(f"Average Geometric Angular Error over {num_angular_samples} samples: {avg_geo_angular_error:.2f} degrees")