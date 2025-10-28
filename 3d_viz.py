import cv2
import json
import numpy as np
import mediapipe as mp

from pyk4a import PyK4A, Config as K4AConfig, ColorResolution, DepthMode, FPS, CalibrationType

# --- Window/context via pyglet ---
import pyglet
from pyglet import gl as _pg_gl  # only for Config
from ctypes import c_float, POINTER

# --- All OpenGL calls via PyOpenGL (fixes your error) ---
from OpenGL.GL import *            # glMatrixMode, glLoadMatrixf, etc.
from OpenGL.GLU import *           # optional (we don't rely on it)

from find_tag import get_detections
from pose_estimate import decompose_homography

# ===================================================================
# Helpers (no GLU dependency)
# ===================================================================

def _to_gl_mat(m: np.ndarray):
    """ctypes pointer to float32 column-major 4x4 matrix for fixed pipeline."""
    m = np.asarray(m, dtype=np.float32)
    col_major = m.T.copy().reshape(-1)
    return (c_float * 16)(*col_major)

def perspective(fovy_deg: float, aspect: float, znear: float, zfar: float) -> np.ndarray:
    f = 1.0 / np.tan(np.deg2rad(fovy_deg) / 2.0)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (zfar + znear) / (znear - zfar)
    M[2, 3] = (2 * zfar * znear) / (znear - zfar)
    M[3, 2] = -1.0
    return M

def _normalize(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def look_at(eye, center, up) -> np.ndarray:
    eye = np.array(eye, dtype=np.float32)
    center = np.array(center, dtype=np.float32)
    up = _normalize(np.array(up, dtype=np.float32))

    f = _normalize(center - eye)
    s = _normalize(np.cross(f, up))
    u = np.cross(s, f)

    M = np.identity(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    T = np.identity(4, dtype=np.float32)
    T[:3, 3] = -eye
    return M @ T

# ===================================================================
# Global state
# ===================================================================

# ---------- Arcball / FPS hybrid camera ----------
class ArcballCamera:
    def __init__(self, target=(0,0,0), distance=6.0, yaw_deg=-45.0, pitch_deg=25.0,
                 min_dist=0.2, max_dist=200.0):
        self.target = np.array(target, dtype=np.float32)
        self.distance = float(distance)
        self.yaw = np.deg2rad(yaw_deg)
        self.pitch = np.deg2rad(pitch_deg)
        self.min_dist = float(min_dist)
        self.max_dist = float(max_dist)
        self._eps = 1e-3

    def view_matrix(self):
        # eye from spherical around target
        cp = np.cos(self.pitch); sp = np.sin(self.pitch)
        cy = np.cos(self.yaw);   sy = np.sin(self.yaw)
        dir_cam = np.array([cp*cy, sp, cp*sy], dtype=np.float32)   # forward vector from target to eye (but we want eye)
        eye = self.target + dir_cam * self.distance
        return look_at(tuple(eye), tuple(self.target), (0,1,0))

    # ----- interactions -----
    def orbit(self, dx_px, dy_px, sensitivity=0.005):
        self.yaw   -= dx_px * sensitivity
        self.pitch -= dy_px * sensitivity
        # clamp pitch to avoid gimbal flip
        self.pitch = float(np.clip(self.pitch, -np.pi/2 + 0.01, np.pi/2 - 0.01))

    def zoom(self, scroll_y, zoom_factor=0.12):
        # scroll_y > 0 => zoom in
        scale = np.exp(-scroll_y * zoom_factor)
        self.distance = float(np.clip(self.distance * scale, self.min_dist, self.max_dist))

    def pan(self, dx_px, dy_px, viewport_h, pan_speed=1.2):
        # pan amount proportional to distance and screen size
        amount = (self.distance * pan_speed) / max(1, viewport_h)
        # camera basis
        cp = np.cos(self.pitch); sp = np.sin(self.pitch)
        cy = np.cos(self.yaw);   sy = np.sin(self.yaw)
        forward = np.array([cp*cy, sp, cp*sy], dtype=np.float32)
        right   = np.array([sy, 0.0, -cy], dtype=np.float32)
        up_cam  = np.cross(right, forward)
        # drag right (+dx) moves target left in world; drag up (-dy) moves target up
        self.target += (-dx_px * amount) * right + (dy_px * amount) * up_cam

    # keyboard strafing (WASD + QE)
    def strafe(self, dx, dy, dz, speed=0.1):
        # dx: right(-/+)  dy: up(-/+)  dz: forward(-/+)
        cy = np.cos(self.yaw); sy = np.sin(self.yaw)
        right   = np.array([sy, 0.0, -cy], dtype=np.float32)
        forward = np.array([ -cy, 0.0, -sy], dtype=np.float32)  # screen-forward keeping horizon level
        up      = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        move = (dx*right + dy*up + dz*forward) * (self.distance * speed)
        self.target += move.astype(np.float32)


latest_camera_pose = np.identity(4, dtype=np.float32)
latest_wrist_world = None
latest_vector_world = None
is_data_valid = False

with open("transformation_map.json", "r") as f:
    loaded = json.load(f)

half_side_m = 0.10  # same as your other file
transformation_map = {}
for k, v in loaded.items():
    M = np.array(v, dtype=np.float32).reshape(4, 4)
    M[:3, 3] *= half_side_m   # convert tag-units -> meters
    transformation_map[int(k)] = M

# Pyglet window (use pyglet.gl.Config ONLY for context setup)
pg_config = _pg_gl.Config(sample_buffers=1, samples=8, depth_size=24, double_buffer=True)
window = pyglet.window.Window(
    width=1280, height=720, caption="3D Pipeline Verification",
    config=pg_config, resizable=True
)
cam = ArcballCamera(target=(0,0,0), distance=6.0, yaw_deg=-45, pitch_deg=25)

# Enable GL state (via PyOpenGL)
glEnable(GL_DEPTH_TEST)
try:
    glEnable(GL_MULTISAMPLE)
except Exception:
    pass  # not available on all drivers

def draw_axes(size=1.0):
    glLineWidth(3.0)
    glBegin(GL_LINES)
    # X (red)
    glColor3f(1.0, 0.0, 0.0); glVertex3f(0.0, 0.0, 0.0); glVertex3f(size, 0.0, 0.0)
    # Y (green)
    glColor3f(0.0, 1.0, 0.0); glVertex3f(0.0, 0.0, 0.0); glVertex3f(0.0, size, 0.0)
    # Z (blue)
    glColor3f(0.0, 0.0, 1.0); glVertex3f(0.0, 0.0, 0.0); glVertex3f(0.0, 0.0, size)
    glEnd()


def draw_posed_thing(pose_matrix, draw_function, *args, **kwargs):
    glPushMatrix()
    glMultMatrixf(_to_gl_mat(pose_matrix))
    draw_function(*args, **kwargs)
    glPopMatrix()

def draw_quad(size=0.1, color=(1, 1, 1)):
    s = size / 2.0
    glColor3f(*color)
    glBegin(GL_QUADS)
    glVertex3f(-s, -s, 0.0)
    glVertex3f( s, -s, 0.0)
    glVertex3f( s,  s, 0.0)
    glVertex3f(-s,  s, 0.0)
    glEnd()

def draw_vector(start_point, vector, length=1.5, color=(1, 1, 0)):
    end_point = start_point + vector * length
    glLineWidth(2.0)
    glColor3f(*color)
    glBegin(GL_LINES)
    glVertex3f(*start_point.astype(np.float32))
    glVertex3f(*end_point.astype(np.float32))
    glEnd()

@window.event
def on_draw():
    window.clear()

    # Projection
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    P = perspective(60.0, window.width / float(window.height), 0.1, 100.0)
    glLoadMatrixf(_to_gl_mat(P))

    # View
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    V = cam.view_matrix() 
    glLoadMatrixf(_to_gl_mat(V))

    # World axes
    draw_axes(size=0.5)

    # Tags in world
    for tag_id, tag_pose in transformation_map.items():
        draw_posed_thing(tag_pose, draw_quad, size=0.2, color=(0.8, 0.8, 0.8))

    # Live camera pose / wrist / vector
 
    draw_posed_thing(latest_camera_pose, draw_axes, size=0.12)
    if is_data_valid:
        # draw_posed_thing(latest_camera_pose, draw_axes, size=0.12)
        if latest_wrist_world is not None and latest_vector_world is not None:
            draw_vector(latest_wrist_world, latest_vector_world, length=1.5, color=(1, 1, 0))
            wrist_pose = np.identity(4, dtype=np.float32)
            wrist_pose[:3, 3] = latest_wrist_world.astype(np.float32)
            draw_posed_thing(wrist_pose, draw_quad, size=0.05, color=(0, 1, 0))

_last_buttons = 0
_last_x = 0
_last_y = 0

@window.event
def on_mouse_press(x, y, button, modifiers):
    global _last_buttons, _last_x, _last_y
    _last_buttons = button
    _last_x, _last_y = x, y

@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    global _last_x, _last_y, _last_buttons
    _last_x, _last_y = x, y
    # Left-drag = orbit; Right-drag (or Middle) = pan
    if buttons & pyglet.window.mouse.LEFT:
        cam.orbit(dx, dy, sensitivity=0.0045)
    if buttons & (pyglet.window.mouse.RIGHT | pyglet.window.mouse.MIDDLE):
        cam.pan(dx, dy, viewport_h=window.height, pan_speed=1.3)

@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    cam.zoom(scroll_y, zoom_factor=0.18)

# Simple WASD + QE + R reset
_keys = pyglet.window.key.KeyStateHandler()
window.push_handlers(_keys)

def _update_camera_keys(dt):
    speed = 0.06
    if _keys[pyglet.window.key.LSHIFT] or _keys[pyglet.window.key.RSHIFT]:
        speed *= 3.0
    dx = dy = dz = 0.0
    if _keys[pyglet.window.key.A]: dx -= 1
    if _keys[pyglet.window.key.D]: dx += 1
    if _keys[pyglet.window.key.W]: dz += 1
    if _keys[pyglet.window.key.S]: dz -= 1
    if _keys[pyglet.window.key.Q]: dy -= 1
    if _keys[pyglet.window.key.E]: dy += 1
    if dx or dy or dz:
        cam.strafe(dx, dy, dz, speed=speed)

@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.R:
        # reset camera
        cam.__init__(target=(0,0,0), distance=6.0, yaw_deg=-45, pitch_deg=25)

# run the camera key-update at ~60Hz
pyglet.clock.schedule_interval(_update_camera_keys, 1.0/60.0)

# ===================================================================
# Capture / processing
# ===================================================================

mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(model_complexity=1, enable_segmentation=False)

k4a = PyK4A(K4AConfig(
    color_resolution=ColorResolution.RES_1080P,
    depth_mode=DepthMode.NFOV_UNBINNED,
    synchronized_images_only=True,
    camera_fps=FPS.FPS_15
))
k4a.start()
print("Input a tag: ")
pointed_to_id = int(input())  # change as needed

def update(dt):
    global is_data_valid, latest_camera_pose, latest_wrist_world, latest_vector_world

    cap = k4a.get_capture()
    if cap is None or cap.color is None or cap.transformed_depth is None:
        return

    color = cap.color  # BGRA
    depth_in_color = cap.transformed_depth
    calib = k4a.calibration

    rgb = cv2.cvtColor(color, cv2.COLOR_BGRA2RGB)

    pose_result = pose_detector.process(rgb)
    tag_detections = get_detections(rgb)

    is_data_valid = False
    if not tag_detections:
        _preview(rgb)
        return

    ref_tag = tag_detections[0]
    if (getattr(ref_tag, "tag_id", None) not in transformation_map) or (pointed_to_id not in transformation_map):
        _preview(rgb)
        return

    T_world_from_ref_tag = transformation_map[ref_tag.tag_id]
    H = np.asarray(ref_tag.homography, dtype=np.float64)

    T_cam_from_ref_tag = np.asarray(decompose_homography(H), dtype=np.float32)  # 4x4
    T_cam_from_ref_tag[:3,3] *= half_side_m
    T_world_from_cam = (T_world_from_ref_tag @ np.linalg.inv(T_cam_from_ref_tag)).astype(np.float32)
    latest_camera_pose = T_world_from_cam

    # Wrist pixel
    if pose_result is None or pose_result.pose_landmarks is None:
        _preview(rgb)
        return
    landmarks = pose_result.pose_landmarks.landmark
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    h, w, _ = rgb.shape
    cx, cy = int(left_wrist.x * w), int(left_wrist.y * h)
    if not (0 <= cx < w and 0 <= cy < h):
        _preview(rgb); return

    depth_mm = int(depth_in_color[cy, cx])
    if depth_mm <= 0:
        _preview(rgb); return

    # Wrist 3D in color camera frame (m)
    wrist_pos_cam = calib.convert_2d_to_3d((cx, cy), depth_mm, CalibrationType.COLOR)
    wrist_pos_cam = np.array(wrist_pos_cam, dtype=np.float32) / 1000.0

    # Target world -> camera
    target_pos_world = transformation_map[pointed_to_id][:3, 3].astype(np.float32)
    T_cam_from_world = np.linalg.inv(T_world_from_cam)
    target_cam = (T_cam_from_world @ np.append(target_pos_world, 1.0).astype(np.float32))[:3]

    vec_cam = target_cam - wrist_pos_cam
    n = np.linalg.norm(vec_cam)
    if n < 1e-8:
        _preview(rgb); return
    vec_cam = (vec_cam / n).astype(np.float32)

    # To world
    wrist_world = (T_world_from_cam @ np.append(wrist_pos_cam, 1.0).astype(np.float32))[:3]
    dir_world = (T_world_from_cam[:3, :3] @ vec_cam).astype(np.float32)

    latest_wrist_world = wrist_world
    latest_vector_world = dir_world
    is_data_valid = True

    # 2D preview
    cv2.circle(rgb, (cx, cy), 6, (255, 0, 0), 2)
    _preview(rgb)

def _preview(rgb):
    cv2.imshow("2D Camera View", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pyglet.app.exit()

@window.event
def on_close():
    try:
        k4a.stop()
    except Exception:
        pass
    cv2.destroyAllWindows()

if __name__ == "__main__":
    pyglet.clock.schedule_interval(update, 1.0 / 15.0)
    try:
        pyglet.app.run()
    finally:
        try:
            k4a.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Application closed.")
