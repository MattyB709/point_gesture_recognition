from pyk4a import PyK4A, Config, ColorResolution, DepthMode
from pyk4a import CalibrationType
import numpy as np

cfg = Config(
    color_resolution=ColorResolution.RES_1080P,       # 1920x1080
    depth_mode=DepthMode.NFOV_UNBINNED,               # 640x576 depth
    synchronized_images_only=True                     # depth+color in same capture
)
k4a = PyK4A(cfg)
k4a.start()

while (True):
    cap = k4a.get_capture()          # blocking
    color = cap.color                # numpy uint8, shape (1080,1920,4) BGRA
    depth = cap.depth                # numpy uint16, shape (576,640), units = millimeters

    calib = k4a.calibration                    # pyk4a Calibration object (intrinsics+extrinsics)
    depth_in_color = cap.transformed_depth
    point_x = 640
    point_y = 960
    # returns a (1080,1920) uint16 array (mm) aligned to the color image
    depth_point = depth_in_color[point_x, point_y]
    if depth_point == 0:
        print("0")
        continue
    xmm, ymm, zmm = calib.convert_2d_to_3d((point_x, point_y), 
                                        depth_point, 
                                        CalibrationType.COLOR)

    print(xmm, ymm, zmm)