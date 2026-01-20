import cv2
from pyk4a import CalibrationType
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS
import os
import numpy as np
import torch
import torchvision
from test_model_live import preprocess_exact
from train.metrics import angular_error

def get_2d_points_from_vec(calib, wrist_coords, vec):

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

if __name__ == "__main__":


    os.makedirs("grid_data", exist_ok=True)
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
    model = torchvision.models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    state_dict = torch.load("trained_models/ResNet50_augFalse_ampFalse_h_flip_2025-12-12 14:04.pth", map_location="cpu")["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    total_angular_error = 0.0
    num_angular_samples = 0
    for file in os.listdir("split_data/val"):
        
        if not file.endswith(".txt"):
            continue
        print(f"Processing {file}...")
        txt_path = os.path.join("split_data/val", file)
        base_name = file[:-4]
        img_path = os.path.join("split_data/val", base_name + ".jpg")
        color_img = cv2.imread(img_path)
        input_tensor = preprocess_exact(cv2.cvtColor(color_img, cv2.COLOR_BGR2BGRA)).to('cpu')
        with torch.no_grad():
            output = model(input_tensor)
            output = output.cpu().squeeze()
            conf = torch.sigmoid(output[0]).item()
            pred_vec = output[1:].numpy()
            pred_vec = pred_vec / np.linalg.norm(pred_vec)



        with open(txt_path, "r") as f:
            lines = f.readlines()
            label = int(lines[0].strip())
            if label == 1:
                num_angular_samples += 1
                nums = [float(x.strip()) for x in lines[1:]]
                start_xyz_m = np.array(nums[:3])
                dir_vec = np.array(nums[3:])
                cam_point, cam_wrist = get_2d_points_from_vec(calib, start_xyz_m, dir_vec)
                if cam_point is not None and cam_wrist is not None:
                    cv2.circle(color_img, cam_wrist, 10, (0, 255, 0), -1)  # Green for wrist
                    cv2.line(color_img, cam_wrist, cam_point, (0, 0, 255), 5)  # Red for pointing direction
                cam_point_pred, cam_wrist_pred = get_2d_points_from_vec(calib, start_xyz_m, pred_vec)
                if cam_point_pred is not None and cam_wrist_pred is not None:
                    cv2.line(color_img, cam_wrist_pred, cam_point_pred, (255, 255, 0), 5)  # Cyan for predicted direction   
                    error = angular_error(torch.from_numpy(dir_vec).unsqueeze(0), torch.from_numpy(pred_vec).unsqueeze(0))
                    total_angular_error += error
                    cv2.putText(color_img, f"Error: {error:.1f} deg", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)


        cv2.imwrite(f"check_data/{base_name}.jpg", color_img)
        key = cv2.waitKey(0)
        if key == 27:  # ESC to exit
            break
    if num_angular_samples > 0:
        avg_angular_error = total_angular_error / num_angular_samples
        print(f"Average Angular Error over {num_angular_samples} samples: {avg_angular_error:.2f} degrees")