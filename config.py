# ============================================================
# Screen / Game Window Size
# ============================================================

GAME_WIDTH = 1920
GAME_HEIGHT = 1080

# Compatibility alias
SCREEN_WIDTH = GAME_WIDTH
SCREEN_HEIGHT = GAME_HEIGHT


# ============================================================
# Camera
# ============================================================

#CAMERA_URL = "http://172.30.2.62:8080/video"
# CAMERA_URL = "http://192.168.1.5:8080/video"
CAMERA_URL = "http://192.168.1.194:4747/video"

CAMERA_CONNECT_RETRY_SECONDS = 2.0


# ============================================================
# Detection intervals
# ============================================================

MONITOR_DETECTION_INTERVAL = 1.0
OBSTACLE_DETECTION_INTERVAL = 0.25


# ============================================================
# Calibration
# ============================================================

CALIBRATION_PATTERN_OUTER_COLOR = (40, 255, 60)
CALIBRATION_PATTERN_INNER_COLOR = (255, 255, 255)

CALIBRATION_OUTER_BORDER_SIZE = 40
CALIBRATION_INNER_BORDER_SIZE = 26

CALIBRATION_TARGET_HUE_DEGREES = 120
CALIBRATION_TARGET_HUE_TOLERANCE_DEGREES = 45

CALIBRATION_MIN_TARGET_SATURATION = 50
CALIBRATION_MIN_TARGET_VALUE = 35

CALIBRATION_ADAPTIVE_HUE_ENABLED = True
CALIBRATION_ADAPTIVE_HUE_BLEND = 0.08
CALIBRATION_ADAPTIVE_MAX_DRIFT_DEGREES = 30

CALIBRATION_WHITE_S_MAX = 90
CALIBRATION_WHITE_V_MIN = 150

CALIBRATION_MIN_SCREEN_AREA_RATIO = 0.10
CALIBRATION_MAX_SCREEN_AREA_RATIO = 0.90

CALIBRATION_MIN_OUTER_TARGET_RATIO = 0.55
CALIBRATION_MIN_INNER_WHITE_RATIO = 0.40

CALIBRATION_MIN_QUALITY = 0.50
CALIBRATION_MIN_FULL_TARGET_RATIO = 0.85

CALIBRATION_REQUIRED_STABLE_FRAMES = 5
CALIBRATION_MAX_CORNER_JUMP_RATIO = 0.035

CALIBRATION_SMOOTHING_ALPHA = 0.25

CALIBRATION_MASK_CLOSE_KERNEL = 17


# ============================================================
# Camera feed health
# ============================================================

CAMERA_FROZEN_DIFF_THRESHOLD = 0.6
CAMERA_FROZEN_SECONDS = 2.0


# ============================================================
# Rendering
# ============================================================

BACKGROUND_COLOR = (0, 0, 0)

TARGET_FPS = 144

USE_DIRTY_RECTS = True
FULL_REPAINT_SECONDS = 10.0


# ============================================================
# Visual effects
# ============================================================

VISUAL_EFFECTS_ENABLED = True

BALL_HUE_MIN_DEGREES = 15
BALL_HUE_MAX_DEGREES = 255

BALL_HUE_SHIFT_ON_BOUNCE = 22

BALL_TRAIL_LENGTH = 16
BALL_TRAIL_SAMPLE_SECONDS = 1.0 / 60.0

BALL_GLOW_ENABLED = True

BALL_SQUASH_STRETCH_ENABLED = True
BALL_SQUASH_DURATION_SECONDS = 0.16

IMPACT_EFFECT_MIN_SPEED = 160.0
MAX_PARTICLES = 260


# ============================================================
# Detected obstacles
# ============================================================

SHOW_DETECTED_OBSTACLES = True

DETECTED_OBSTACLE_COLOR = (255, 255, 255)
DETECTED_OBSTACLE_BORDER_WIDTH = 0

DETECTED_OBSTACLE_PAINT_INSET_RATIO = 0.22


# ============================================================
# Ball physics
# ============================================================

BALL_RADIUS = 16
BALL_RADIUS_VARIATION = 0.25

GRAVITY = 900
BALL_GRAVITY = GRAVITY

BALL_RESTITUTION = 1.0
BALL_REST_SPEED = 30.0

BALL_FRICTION = 0.0

BALL_MAX_SPEED = 2400.0

BALL_BALL_COLLISION_ENABLED = True
BALL_BALL_RESTITUTION = 0.95

PHYSICS_MAX_STEP_SECONDS = 1.0 / 240.0


# ============================================================
# Ball spawning and stuck balls
# ============================================================

BALL_SPAWN_INTERVAL = 3.0
BALL_SPAWN_INTERVAL_SECONDS = BALL_SPAWN_INTERVAL

MAX_SIMULTANEOUS_BALLS = 12

BALL_SPAWN_RANDOM_X = True
BALL_SPAWN_MARGIN = 120
BALL_SPAWN_VELOCITY_JITTER = 60.0

BALL_STUCK_DISTANCE = 30.0
BALL_STUCK_TIMEOUT_SECONDS = 8.0

BALL_LIFETIME = 10.0
BALL_MAX_LIFETIME_SECONDS = BALL_LIFETIME



# ============================================================
# White obstacle detection
# ============================================================

DETECTION_SCALE = 0.5

MIN_OBSTACLE_AREA = 1500
MAX_OBSTACLE_AREA = 200000

OBSTACLE_COARSE_BRIGHTNESS = 90
OBSTACLE_COARSE_SATURATION = 160

OBSTACLE_EDGE_LEVEL = 0.5

MIN_OBSTACLE_BRIGHTNESS = 130
MAX_OBSTACLE_SATURATION = 110

MIN_OBSTACLE_RECTANGULARITY = 0.86
MIN_OBSTACLE_SOLIDITY = 0.90

MAX_OBSTACLE_ASPECT_RATIO = 4.0

MAX_OBSTACLE_BRIGHTNESS_STD = 30.0

MIN_OBSTACLE_DARK_SURROUND_RATIO = 0.55

OBSTACLE_BORDER_MASK_MARGIN = 4
OBSTACLE_EDGE_MARGIN = 5

OBSTACLE_SIZE_TOLERANCE = 0.22
OBSTACLE_EXPECTED_SIZE = None

OBSTACLE_ALLOW_MERGED_STICKERS = True

OBSTACLE_SNAP_TO_EXPECTED_SIZE = True

OBSTACLE_SIZE_FORGET_SECONDS = 30.0

OBSTACLE_RELAXED_FACTOR = 0.90

OBSTACLE_REQUIRED_STABLE_FRAMES = 2

OBSTACLE_MAX_MISSING_FRAMES = 12

OBSTACLE_STALE_MISSES = 3

OBSTACLE_PENDING_MAX_MISSING_FRAMES = 1

OBSTACLE_TEMPORAL_MATCH_DISTANCE = 80

OBSTACLE_MASK_OPEN_KERNEL = 3
OBSTACLE_MASK_CLOSE_KERNEL = 5
