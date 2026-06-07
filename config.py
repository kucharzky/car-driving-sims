"""Centralized hyperparameters and physical constants for the simulation."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
DEFAULT_TRACK_PATH = ASSETS_DIR / "track.png"

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
FPS = 60
DT = 1.0 / FPS

# ---------------------------------------------------------------------------
# Vehicle (kinematic bicycle model)
# ---------------------------------------------------------------------------
WHEELBASE = 40.0          # pixels — distance between axles
CAR_LENGTH = 30.0
CAR_WIDTH = 18.0
MAX_SPEED = 6.0           # pixels / frame
MAX_REVERSE_SPEED = 2.0
MAX_ACCELERATION = 0.15
MAX_BRAKING = 0.25
MAX_STEER_ANGLE = 0.6     # radians (~34°)
STEER_RATE = 0.08         # radians per frame while key held

# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------
TRACK_WIDTH = 80.0        # pixels — procedural track half-width (both sides)
TRACK_COLOR_ROAD = (255, 255, 255)
TRACK_COLOR_WALL = (0, 0, 0)
TRACK_COLOR_START_FINISH = (0, 255, 0)

# Tolerance when matching PNG pixel colors (anti-aliasing)
COLOR_TOLERANCE = 30

# Procedural generation
PROCEDURAL_CONTROL_POINTS = 8
PROCEDURAL_MARGIN = 120   # inset from screen edges for control points

# ---------------------------------------------------------------------------
# Vision (Phase 2+)
# ---------------------------------------------------------------------------
VISION_SIZE = 84
FRAME_STACK = 4
VISION_CROP_SIZE = 200

# ---------------------------------------------------------------------------
# Raycasting (Phase 2+)
# ---------------------------------------------------------------------------
RAY_COUNT = 7
RAY_MAX_DISTANCE = 300.0
RAY_ANGLES_DEG = (-90, -60, -30, 0, 30, 60, 90)

# ---------------------------------------------------------------------------
# Rewards (Phase 3)
# ---------------------------------------------------------------------------
REWARD_STEP = 0.1
REWARD_CRASH = -100.0
REWARD_LAP = 500.0

# ---------------------------------------------------------------------------
# RL / Agent (Phase 3)
# ---------------------------------------------------------------------------
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 50_000
TARGET_UPDATE_EVERY = 10

# Discrete actions: 0=left, 1=right, 2=accelerate, 3=brake
ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_ACCELERATE = 2
ACTION_BRAKE = 3
NUM_ACTIONS = 4
