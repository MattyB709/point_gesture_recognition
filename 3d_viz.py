import cv2
import json
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS, CalibrationType

# --- 3D Visualization Imports ---
import pyglet
from pyglet.gl import *
from ctypes import *
from find_tag import get_detections
from pose_estimate import decompose_homography

# ===================================================================
# 3D VISUALIZATION SETUP (PYGLET AND OPENGL)
# ===================================================================

# --- Global variables to hold the latest data for drawing ---
# This data is calculated in update() and used in on_draw()
latest_camera_pose = np.identity(4)
latest_wrist_world = None
latest_vector_world = None
is_data_valid = False

# --- PyGLet Window Setup ---
config = Config(sample_buffers=1, samples=8) # For anti-aliasing
window = pyglet.window.Window(width=1280, height=720, caption='3D Pipeline Verification', config=config, resizable=True)
glEnable(GL_DEPTH_TEST)
glEnable(GL_MULTISAMPLE_ARB)

def draw_axes(size=1.0):
    """Draws Red, Green, Blue lines for X, Y, Z axes."""
    glLineWidth(3.0)
    glBegin(GL_LINES)
    # X Axis (Red)
    glColor3f(1, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(size, 0, 0)
    # Y Axis (Green)
    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, size, 0)
    # Z Axis (Blue)
    glColor3f(0, 0, 1)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, size)
    glEnd()

def draw_posed_thing(pose_matrix, draw_function, *args, **kwargs):
    """Helper function to apply a pose and draw something."""
    glPushMatrix()
    # OpenGL expects column-major matrices, numpy provides row-major. We must transpose.
    glMultMatrixf(pose_matrix.T.flatten().ctypes.data_as(POINTER(c_float)))
    draw_function(*args, **kwargs)
    glPopMatrix()

def draw_quad(size=0.1, color=(1,1,1)):
    """Draws a simple square quad."""
    s = size / 2.0
    glColor3f(*color)
    glBegin(GL_QUADS)
    glVertex3f(-s, -s, 0)
    glVertex3f( s, -s, 0)
    glVertex3f( s,  s, 0)
    glVertex3f(-s,  s, 0)
    glEnd()

def draw_vector(start_point, vector, length=1.0, color=(1,1,0)):
    """Draws a line segment from a start point along a vector."""
    glLineWidth(2.0)
    glColor3f(*color)
    end_point = start_point + vector * length
    glBegin(GL_LINES)
    glVertex3f(*start_point)
    glVertex3f(*end_point)
    glEnd()

@window.event
def on_draw():
    """This function is called by PyGLet to render the scene."""
    window.clear()
    
    # --- Setup 3D Perspective ---
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, window.width / float(window.height), 0.1, 100.0)

    # --- Setup Viewer Camera ---
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    # Position the viewer's camera to look at the scene
    # Params: (eyeX, eyeY, eyeZ, centerX, centerY, centerZ, upX, upY, upZ)
    gluLookAt(1.5, 1.5, 1.5, 0, 0, 0, 0, 1, 0)

    # --- Draw the World and its contents ---
    draw_axes(size=0.5) # World origin axes

    # Draw all the pre-calibrated AprilTags
    for tag_id, tag_pose in transformation_map.items():
        draw_posed_thing(tag_pose, draw_quad, size=0.2, color=(0.8, 0.8, 0.8))

    # If the last frame's data was valid, draw the live elements
    if is_data_valid:
        # Draw the camera's current pose
        draw_posed_thing(latest_camera_pose, draw_axes, size=0.1)
        
        # Draw the wrist and the pointing vector
        if latest_wrist_world is not None:
            # Draw the vector first
            draw_vector(latest_wrist_world, latest_vector_world, length=1.5, color=(1, 1, 0))
            # Draw a small quad at the wrist position
            # Construct a pose matrix for the wrist sphere
            wrist_pose = np.identity(4)
            wrist_pose[:3, 3] = latest_wrist_world
            draw_posed_thing(wrist_pose, draw_quad, size=0.05, color=(0,1,0))


# ===================================================================
# MAIN APPLICATION LOGIC (YOUR ORIGINAL CODE, RESTRUCTURED)
# ===================================================================

# --- Global App State and Setup ---
PATH = 'pose_landmarker.task'
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose()

# --- K4A Camera Setup ---
k4a = PyK4A(Config(
    color_resolution=ColorResolution.RES_1080P,
    depth_mode=DepthMode.NFOV_UNBINNED,
    synchronized_images_only=True,
    camera_fps=FPS.FPS_15
))
k4a.start()
pointed_to_id = 19 # Default target

with open("transformation_map.json", "r") as f:
    loaded = json.load(f)
transformation_map = {int(k): np.array(v) for k, v in loaded.items()}


def update(dt):
    """
    This function contains your original processing loop.
    It's called repeatedly by pyglet.clock.
    """
    global is_data_valid, latest_camera_pose, latest_wrist_world, latest_vector_world

    # --- Get Frames ---
    cap = k4a.get_capture()
    if not cap.color is None and not cap.depth is None:
        color = cap.color
        depth_in_color = cap.transformed_depth
        calib = k4a.calibration
    else:
        return # Skip frame if capture failed

    rgb = cv2.cvtColor(color, cv2.COLOR_BGRA2RGB)
    
    # --- Run Detections ---
    pose_result = pose_detector.process(rgb)
    tag_detections = get_detections(rgb) # Your AprilTag detector
    
    # --- Main Logic ---
    is_data_valid = False # Assume failure until success
    if tag_detections and pose_result.pose_landmarks:
        # For simplicity, use the first detected tag for localization
        ref_tag = tag_detections[0]
        
        # Check if reference tag and target tag are in our map
        if ref_tag.tag_id in transformation_map and pointed_to_id in transformation_map:
            # --- Perform Transformations (as in your original code) ---
            T_world_from_ref_tag = transformation_map[ref_tag.tag_id]
            H = ref_tag.homography.astype(np.float64)
            T_cam_from_ref_tag = decompose_homography(H) # You need to implement this
            
            # This is the pose of the camera in the world frame
            T_world_from_cam = T_world_from_ref_tag @ np.linalg.inv(T_cam_from_ref_tag)
            
            # --- Get Wrist Position ---
            landmarks = pose_result.pose_landmarks.landmark
            left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
            
            h, w, _ = rgb.shape
            cx, cy = int(left_wrist.x * w), int(left_wrist.y * h)

            if 0 < cx < w and 0 < cy < h:
                depth_mm = depth_in_color[cy, cx]
                if depth_mm > 0: # Valid depth reading
                    # --- Get Wrist and Vector in Camera Frame ---
                    wrist_pos_cam = calib.convert_2d_to_3d((cx, cy), depth_mm, CalibrationType.COLOR)
                    wrist_pos_cam = np.array(wrist_pos_cam) / 1000.0 # Convert to meters

                    target_pos_world = transformation_map[pointed_to_id][:3, 3]
                    
                    # Transform target from world to camera frame to calculate vector
                    T_cam_from_world = np.linalg.inv(T_world_from_cam)
                    target_pos_world_h = np.append(target_pos_world, 1) # Homogeneous coords
                    target_pos_cam = (T_cam_from_world @ target_pos_world_h)[:3]

                    vector_cam = target_pos_cam - wrist_pos_cam
                    if np.linalg.norm(vector_cam) > 0:
                        vector_cam = vector_cam / np.linalg.norm(vector_cam)
                        
                        # --- CONVERT TO WORLD FRAME FOR VISUALIZATION ---
                        # Transform wrist point from camera to world
                        wrist_world = (T_world_from_cam @ np.append(wrist_pos_cam, 1))[:3]
                        
                        # Transform vector from camera to world (only rotation)
                        vector_world = (T_world_from_cam[:3, :3] @ vector_cam)
                        
                        # --- Update global state for the renderer ---
                        latest_camera_pose = T_world_from_cam
                        latest_wrist_world = wrist_world
                        latest_vector_world = vector_world
                        is_data_valid = True # Success!

    # --- 2D Visualization (Optional) ---
    cv2.imshow("2D Camera View", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pyglet.app.exit() # Properly close the app


if __name__ == "__main__":
    # Schedule the update function to be called at 15 FPS
    pyglet.clock.schedule_interval(update, 1.0/15.0)
    
    # Run the PyGLet application
    pyglet.app.run()
    
    # --- Cleanup ---
    k4a.stop()
    cv2.destroyAllWindows()
    print("Application closed.")