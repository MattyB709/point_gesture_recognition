import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS, CalibrationType
import mediapipe as mp
import sys
import os

# Add path to import joint transformer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.joint_transformer import create_joint_transformer

# -------------------------
# Model (load + CUDA + eval)
# -------------------------
CONF_THRESHOLD = 0.5
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create joint transformer model
model = create_joint_transformer(
    input_dim=3,
    num_joints=33,
    d_model=128,
    nhead=8,
    num_layers=4,
    dropout=0.1
)

# Load checkpoint
checkpoint_path = "joint_transformer_augFalse_ampTrue_2025-12-02 16:20.pth"  # Update this path!
state_dict = torch.load(checkpoint_path, map_location="cpu")["model_state_dict"]
model.load_state_dict(state_dict, strict=True)
model.to(device).eval()
torch.backends.cudnn.benchmark = True

print(f"✓ Loaded joint transformer from {checkpoint_path}")

# -------------------------
# Camera intrinsics (update with your values!)
# -------------------------
K = np.array([[919.76178, 0,        962.6875],
              [0,         919.8909,  550.9944],
              [0,         0,         1]], dtype=np.float64)

fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]

print(f"Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

# -------------------------
# Azure Kinect config
# -------------------------
cfg = Config(
    color_resolution=ColorResolution.RES_1080P,  # 1920x1080
    depth_mode=DepthMode.NFOV_UNBINNED,         # 640x576 depth
    synchronized_images_only=True,
    camera_fps=FPS.FPS_15,
)
k4a = PyK4A(cfg)
k4a.start()
calib = k4a.calibration

# -------------------------
# MediaPipe Pose
# -------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=2,  # Use complexity 2 for better accuracy
    enable_segmentation=False
)

print("Press 'q' to quit.")
SCALE_MM = 300.0  # draw 30 cm from wrist along predicted vector


def extract_joints_kinect(image_rgb, depth_mm, pose_results):
    """
    Extract 3D joint positions using MediaPipe 2D + Kinect depth.
    Matches the dataset's _extract_joints_kinect() method.
    
    Returns:
        np.array of shape (99,) with joint coordinates in meters,
        or None if pose detection failed
    """
    if not pose_results.pose_landmarks:
        return None
    
    H, W = image_rgb.shape[:2]
    landmarks = pose_results.pose_landmarks.landmark
    
    joint_coords = []
    for lm in landmarks:
        # Convert normalized coords to pixels
        x_px = int(lm.x * W)
        y_px = int(lm.y * H)
        
        # Clamp to image bounds
        x_px = max(0, min(W - 1, x_px))
        y_px = max(0, min(H - 1, y_px))
        
        # Get depth at this pixel
        depth_value = depth_mm[y_px, x_px]
        
        if depth_value > 0:
            # Convert 2D + depth to 3D using camera intrinsics
            z_mm = float(depth_value)
            x_mm = (x_px - cx) * z_mm / fx
            y_mm = (y_px - cy) * z_mm / fy
            
            # Convert to meters
            x_m = x_mm / 1000.0
            y_m = y_mm / 1000.0
            z_m = z_mm / 1000.0
        else:
            # No depth data
            x_m, y_m, z_m = 0.0, 0.0, 0.0
        
        joint_coords.extend([x_m, y_m, z_m])
    
    return np.array(joint_coords, dtype=np.float32)


# Main loop
frame_count = 0
while True:
    cap = k4a.get_capture()
    color_bgra = cap.color                # (1080,1920,4) BGRA uint8
    depth_in_color = cap.transformed_depth  # (1080,1920) uint16

    if color_bgra is None or depth_in_color is None:
        continue

    # Convert for MediaPipe
    rgb_for_pose = cv2.cvtColor(color_bgra, cv2.COLOR_BGRA2RGB)
    
    # Get MediaPipe pose
    res = pose.process(rgb_for_pose)

    # Display image
    disp = cv2.cvtColor(rgb_for_pose, cv2.COLOR_RGB2BGR)
    
    # Find left wrist for visualization
    wrist_px = None
    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        x_px = int(lm.x * disp.shape[1])
        y_px = int(lm.y * disp.shape[0])
        if 0 <= x_px < disp.shape[1] and 0 <= y_px < disp.shape[0]:
            wrist_px = (x_px, y_px)
            cv2.circle(disp, wrist_px, 4, (0, 255, 255), -1)
        
        # Draw MediaPipe skeleton (optional)
        # mp.solutions.drawing_utils.draw_landmarks(
        #     disp,
        #     res.pose_landmarks,
        #     mp_pose.POSE_CONNECTIONS,
        #     landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
        #         color=(0, 255, 0), thickness=2, circle_radius=2
        #     ),
        #     connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
        #         color=(0, 255, 0), thickness=1
        #     )
        # )
    
    # Extract joints and run model
    if res.pose_landmarks and wrist_px is not None:
        # Extract 3D joints
        joints = extract_joints_kinect(rgb_for_pose, depth_in_color, res)
        
        if joints is not None and not np.allclose(joints, 0.0):
            # Get wrist 3D position for visualization
            x, y = wrist_px
            depth_mm = int(depth_in_color[y, x])
            
            if depth_mm > 0:
                # Convert to 3D
                xmm, ymm, zmm = calib.convert_2d_to_3d((x, y), depth_mm, CalibrationType.COLOR)
                wrist_mm = np.array([xmm, ymm, zmm], dtype=np.float32)
                
                # Run model on joints
                with torch.no_grad():
                    joints_tensor = torch.from_numpy(joints).float().unsqueeze(0).to(device)
                    
                    # Forward pass
                    out = model(joints_tensor)  # (1, 4)
                    conf = torch.sigmoid(out[:, :1]).item()
                    vec = F.normalize(out[:, 1:], p=2, dim=1)[0].detach().cpu().numpy()
                
                # Visualize if confident
                if conf > CONF_THRESHOLD:
                    end_mm = wrist_mm + vec * SCALE_MM
                    
                    # Project to 2D
                    uv_wrist = calib.convert_3d_to_2d(
                        tuple(wrist_mm.tolist()),
                        CalibrationType.COLOR, 
                        CalibrationType.COLOR
                    )
                    uv_end = calib.convert_3d_to_2d(
                        tuple(end_mm.tolist()),
                        CalibrationType.COLOR, 
                        CalibrationType.COLOR
                    )
                    
                    if uv_wrist is not None and uv_end is not None:
                        p0 = tuple(map(int, uv_wrist))
                        p1 = tuple(map(int, uv_end))
                        
                        # Draw pointing direction
                        cv2.line(disp, p0, p1, (0, 0, 255), 3)  # Red line
                        cv2.circle(disp, p1, 8, (0, 0, 255), -1)  # Red endpoint
                        
                        # Draw confidence
                        cv2.putText(
                            disp, 
                            f"conf={conf:.2f}", 
                            (p0[0] + 6, p0[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, 
                            (0, 0, 255), 
                            2, 
                            cv2.LINE_AA
                        )
                        
                        # Draw vector values
                        cv2.putText(
                            disp,
                            f"dir=[{vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f}]",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA
                        )
                else:
                    # Low confidence - show in corner
                    cv2.putText(
                        disp,
                        f"Not pointing (conf={conf:.2f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA
                    )
        else:
            # Joints are all zeros (depth failed)
            cv2.putText(
                disp,
                "Depth missing",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
    else:
        # No pose detected
        cv2.putText(
            disp,
            "No pose detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
    
    # Show FPS
    frame_count += 1
    if frame_count % 30 == 0:
        cv2.putText(
            disp,
            f"Frame: {frame_count}",
            (10, disp.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    cv2.imshow("Joint Transformer - Realtime Pointing (q to quit)", disp)
    
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cv2.destroyAllWindows()
k4a.stop()
print(f"✓ Processed {frame_count} frames")