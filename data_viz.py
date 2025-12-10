import cv2
from pyk4a import CalibrationType
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS
import os
import numpy as np

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
    for file in os.listdir("data"):
        
        print(f"Processing {file}...")
        if not file.endswith(".txt"):
            continue
        txt_path = os.path.join("data", file)
        base_name = file[:-4]
        img_path = os.path.join("data", base_name + ".jpg")
        color_img = cv2.imread(img_path)

        with open(txt_path, "r") as f:
            lines = f.readlines()
            label = int(lines[0].strip())
            if label == 1:
                nums = [float(x.strip()) for x in lines[1:]]
                start_xyz_m = np.array(nums[:3])
                dir_vec = np.array(nums[3:])
                cam_point, cam_wrist = get_2d_points(calib, start_xyz_m, dir_vec)
                if cam_point is not None and cam_wrist is not None:
                    cv2.circle(color_img, cam_wrist, 10, (0, 255, 0), -1)  # Green for wrist
                    cv2.line(color_img, cam_wrist, cam_point, (0, 0, 255), 5)  # Red for pointing direction

        cv2.imwrite(f"check_data/{base_name}.jpg", color_img)
        key = cv2.waitKey(0)
        if key == 27:  # ESC to exit
            break