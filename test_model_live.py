import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS, CalibrationType
import mediapipe as mp

# -------------------------
# Model (load + CUDA + eval)
# -------------------------
CONF_THRESHOLD = 0.25
device = "cuda" if torch.cuda.is_available() else "cpu"

# model = models.resnet18()
# model.fc = torch.nn.Linear(model.fc.in_features, 4)  # 1 conf + 3 dir
model = models.resnet18()
# model.fc = torch.nn.Linear(model.fc.in_features, 4)  # 1 for confidence + 3 for vector
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(model.fc.in_features, 4)
)
state_dict = torch.load("best_model.pth", map_location="cpu")["model_state_dict"]
model.load_state_dict(state_dict, strict=True)
model.to(device).eval()
torch.backends.cudnn.benchmark = True

# MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
# STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

# def preprocess_exact(bgra_1080p):
#     rgb = cv2.cvtColor(bgra_1080p, cv2.COLOR_BGRA2RGB)
#     rgb = rgb.astype(np.float32) / 255.0
#     chw = np.transpose(rgb, (2, 0, 1))  # (3,H,W)
#     tensor = torch.from_numpy(chw)[None, ...].to(device)  # (1,3,H,W)
    
#     # Apply ImageNet normalization (same as training!)
#     tensor = (tensor - MEAN) / STD
    
#     return tensor

def preprocess_exact(bgra_1080p):
    rgb = cv2.cvtColor(bgra_1080p, cv2.COLOR_BGRA2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return torch.from_numpy(chw)[None, ...].to(device) 

# -------------------------
# Azure Kinect config (match your capture setup)
# -------------------------
cfg = Config(
    color_resolution=ColorResolution.RES_1080P,  # 1920x1080
    depth_mode=DepthMode.NFOV_UNBINNED,         # 640x576 depth (aligned available)
    synchronized_images_only=True,
    camera_fps=FPS.FPS_15,
)
k4a = PyK4A(cfg)
k4a.start()
calib = k4a.calibration

# -------------------------
# MediaPipe Pose (CPU)
# -------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)

print("Press 'q' to quit.")
SCALE_MM = 300.0  # draw 30 cm from wrist along predicted vector

while True:
    cap = k4a.get_capture()
    color_bgra = cap.color                # (1080,1920,4) BGRA uint8
    depth_in_color = cap.transformed_depth  # (1080,1920) uint16

    if color_bgra is None or depth_in_color is None:
        continue

    # 2D wrist with MediaPipe
    rgb_for_pose = cv2.cvtColor(color_bgra, cv2.COLOR_BGRA2RGB)
    res = pose.process(rgb_for_pose)

    disp = cv2.cvtColor(rgb_for_pose, cv2.COLOR_RGB2BGR)  # display image (BGR)
    wrist_px = None

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        x_px = int(lm.x * disp.shape[1])
        y_px = int(lm.y * disp.shape[0])
        if 0 <= x_px < disp.shape[1] and 0 <= y_px < disp.shape[0]:
            wrist_px = (x_px, y_px)
            cv2.circle(disp, wrist_px, 4, (0, 255, 255), -1)

    if wrist_px is not None:
        x, y = wrist_px
        depth_mm = int(depth_in_color[y, x])
        if depth_mm > 0:
            # 2D+depth -> 3D (mm) in COLOR camera frame
            xmm, ymm, zmm = calib.convert_2d_to_3d((x, y), depth_mm, CalibrationType.COLOR)
            wrist_mm = np.array([xmm, ymm, zmm], dtype=np.float32)

            # Model forward (no normalization besides /255; no resize)
            with torch.no_grad():
                inp = preprocess_exact(color_bgra)         # (1,3,H,W) on CUDA
                out = model(inp)                           # (1,4)
                conf = torch.sigmoid(out[:, :1]).item()    # scalar in [0,1]
                vec = F.normalize(out[:, 1:], p=2, dim=1)[0].detach().cpu().numpy()  # (3,)

            if conf > CONF_THRESHOLD:  # optional threshold
                end_mm = wrist_mm + vec * SCALE_MM
                uv_wrist = calib.convert_3d_to_2d(tuple(wrist_mm.tolist()),
                                                  CalibrationType.COLOR, CalibrationType.COLOR)
                uv_end = calib.convert_3d_to_2d(tuple(end_mm.tolist()),
                                                CalibrationType.COLOR, CalibrationType.COLOR)
                if uv_wrist is not None and uv_end is not None:
                    p0 = tuple(map(int, uv_wrist))
                    p1 = tuple(map(int, uv_end))
                    cv2.line(disp, p0, p1, (0, 255, 0), 2)
                    cv2.putText(disp, f"conf={conf:.2f}", (p0[0]+6, p0[1]-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 1, cv2.LINE_AA)

    cv2.imshow("Realtime Pointing (q to quit)", disp)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cv2.destroyAllWindows()
k4a.stop()
