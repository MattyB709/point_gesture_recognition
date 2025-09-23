from pyk4a import PyK4A, Config, ColorResolution, DepthMode, CalibrationType

k4a = PyK4A(Config(
    color_resolution=ColorResolution.RES_1080P,
    depth_mode=DepthMode.NFOV_UNBINNED
))
k4a.start()

cal = k4a.calibration

# Intrinsics (OpenCV-style) for COLOR and DEPTH
K_color = cal.get_camera_matrix(CalibrationType.COLOR)   # 3x3 (fx,0,cx; 0,fy,cy; 0,0,1)
K_depth = cal.get_camera_matrix(CalibrationType.DEPTH)

# Distortion coefficients (Brown–Conrady, OpenCV-compatible)
dist_color = cal.get_distortion_coefficients(CalibrationType.COLOR)  # [k1,k2,p1,p2,k3,k4,k5,k6]
dist_depth = cal.get_distortion_coefficients(CalibrationType.DEPTH)

# Extrinsics between the two sensors
# R_cd, t_cd = cal.get_extrinsics(CalibrationType.COLOR, CalibrationType.DEPTH)  # R (3x3), t (in mm)
print("K COLOR: ", K_color)
print("dist COLOR: ", dist_color)

k4a.stop()
