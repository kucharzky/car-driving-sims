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
WHEELBASE = 40.0
CAR_LENGTH = 30.0
CAR_WIDTH = 18.0
MAX_SPEED = 6.0
MAX_REVERSE_SPEED = 2.0
MAX_ACCELERATION = 0.15
MAX_BRAKING = 0.25
MAX_STEER_ANGLE = 0.6
STEER_RATE = 0.08

# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------
TRACK_WIDTH = 80.0
TRACK_COLOR_ROAD = (255, 255, 255)
TRACK_COLOR_WALL = (0, 0, 0)
TRACK_COLOR_START_FINISH = (0, 255, 0)
COLOR_TOLERANCE = 30
PROCEDURAL_CONTROL_POINTS = 8
PROCEDURAL_MARGIN = 120
START_FINISH_LINE_WIDTH = 6

# Lap detection
SPAWN_OFFSET_BEFORE_LINE = 55.0
MIN_LAP_DISTANCE_FLOOR = 200.0
LAP_DISTANCE_FRACTION = 0.42
MIN_LAP_ANGLE_PROGRESS = 3.5
LAP_FAR_DISTANCE_FRACTION = 0.72
LAP_HEADING_TOLERANCE = 0.75
FINISH_ZONE_RADIUS = 38.0
MIN_CLEAR_START_DISTANCE = 25.0
MIN_FORWARD_DOT = 0.35

# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------
VISION_SIZE = 84
FRAME_STACK = 4
VISION_CROP_SIZE = 200
RAY_COLOR_NEAR = (255, 80, 80)
RAY_COLOR_FAR = (80, 220, 120)

# ---------------------------------------------------------------------------
# Raycasting
# ---------------------------------------------------------------------------
RAY_COUNT = 7
RAY_MAX_DISTANCE = 300.0
RAY_ANGLES_DEG = (-90, -60, -30, 0, 30, 60, 90)

# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------
REWARD_PATH_SCALE = 0.2           # per pixel traveled on road with speed > 0
REWARD_WALL_PROXIMITY = 2.0       # scales with (1 - min_ray)^2 when close to wall
REWARD_PROGRESS_DELTA = 0.15      # per % lap progress gained this step
REWARD_SPEED_CLEAR = 0.08         # bonus for speed when center ray is clear
REWARD_IDLE = 0.08                # penalty when stopped with clear path ahead
REWARD_OFF_ROAD = 3.0             # per step off drivable surface
REWARD_CRASH = -80.0
REWARD_LAP = 500.0
WALL_PROXIMITY_THRESHOLD = 0.45   # rays below this trigger proximity penalty
CLEAR_RAY_THRESHOLD = 0.5         # center-ish clearance for speed / idle logic

# ---------------------------------------------------------------------------
# RL / Agent
# ---------------------------------------------------------------------------
LEARNING_RATE = 3e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.08
EPSILON_DECAY = 0.995             # per episode
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 80_000
TARGET_UPDATE_EVERY = 500
TRAIN_UPDATES_PER_STEP = 2
MIN_REPLAY_TO_TRAIN = 500

# Combined actions: steer + throttle in one decision
ACTION_ACCEL_STRAIGHT = 0
ACTION_ACCEL_LEFT = 1
ACTION_ACCEL_RIGHT = 2
ACTION_COAST_LEFT = 3
ACTION_COAST_RIGHT = 4
ACTION_BRAKE = 5
NUM_ACTIONS = 6

# Legacy aliases (manual keyboard mapping unchanged)
ACTION_LEFT = ACTION_COAST_LEFT
ACTION_RIGHT = ACTION_COAST_RIGHT
ACTION_ACCELERATE = ACTION_ACCEL_STRAIGHT
ACTION_BRAKE = ACTION_BRAKE

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "best_agent.pth"
MAX_EPISODE_STEPS = 4000
TRAIN_LOG_EVERY = 10
SAVE_EVERY_EPISODES = 25
