"""
Centralized configuration: all rule thresholds and dataset constants.

Pose backbone  : YOLO11n-pose  (17 COCO keypoints)
All detectors  : Rule-based  (no ML model)
"""

# ─── Paths ─────────────────────────────────────────────────────────────────────
# Pre-processed X.npy / y.npy / meta.csv  (YOLO keypoints, already extracted)
NPY_DATA_DIR = r"f:\Project_F\Company_Abnormal_Project\iccas\subject1_2_3_4\cv_dataset"

# YOLO pose model weights
YOLO_POSE_MODEL = "yolo11s-pose.pt"   # auto-downloaded by ultralytics on first run

# Raw video / PNG dataset (UP-Fall, Subject 1-4)
UPFALL_DATASET_PATH = r"F:\Project_F\ICCAS_2026\fall_iccas\dataset"

# ─── Dataset ───────────────────────────────────────────────────────────────────
STGCN_FPS    = 19.0   # FPS used during keypoint extraction → evaluation FPS
STGCN_WINDOW = 30     # frames per sliding window in X.npy
STGCN_STRIDE = 15     # stride between windows

# UP-Fall activity labels (Activity01..Activity11)  — CORRECTED mapping
# Source: Martinez-Velasco et al., "UP-Fall Detection Dataset", Data 2019
UPFALL_ACTIVITIES = {
    1:  "falling_forward_hands",   # 10s
    2:  "falling_forward_knees",   # 10s
    3:  "falling_backward",        # 10s
    4:  "falling_sideward",        # 10s
    5:  "falling_sitting",         # 10s  (falling into empty chair)
    6:  "walking",                 # 60s
    7:  "standing",               # 60s
    8:  "sitting",                # 60s  (NOTE: NOT running — UP-Fall has no running)
    9:  "picking_up_object",      # 10s
    10: "jumping",                # 30s
    11: "laying",                 # 60s
}

FALL_ACTIVITY_IDS       = {1, 2, 3, 4, 5}
SITTING_ACTIVITY_IDS    = {8}               # sitting (not running)
JUMPING_ACTIVITY_IDS    = {10}              # jumping
INACTIVITY_ACTIVITY_IDS = {7, 8, 11}       # standing + sitting + laying = inactive

# NOTE: UP-Fall dataset has NO running activity.
# Running detector is evaluated on KTH Action Dataset (separate).
RUNNING_ACTIVITY_IDS    = set()   # empty — no running in UP-Fall

# ─── YOLO Pose — COCO 17-joint indices ──────────────────────────────────────────
KP = {
    "nose":       0,
    "l_eye":      1,  "r_eye":      2,
    "l_ear":      3,  "r_ear":      4,
    "l_shoulder": 5,  "r_shoulder": 6,
    "l_elbow":    7,  "r_elbow":    8,
    "l_wrist":    9,  "r_wrist":    10,
    "l_hip":      11, "r_hip":      12,
    "l_knee":     13, "r_knee":     14,
    "l_ankle":    15, "r_ankle":    16,
}
NUM_JOINTS = 17

# Minimum YOLO keypoint confidence to treat a joint as visible
KP_VISIBILITY_THRESHOLD = 0.2

# ─── YOLO detector settings ─────────────────────────────────────────────────────
YOLO_CONF    = 0.10
YOLO_VERBOSE = False

# ─── Fall Detection (rule-based) ─────────────────────────────────────────────────
#
#   Two complementary rule sets (OR logic — either fires → fall):
#
#   A) Geometry rules
#      • body_tilt_angle > FALL["angle_threshold"]        (spine ≥55° from vertical)
#      • hip_height_norm > FALL["hip_height_threshold"]   (hips in lower frame)
#        OR aspect_ratio  < FALL["aspect_ratio_threshold"] (wide/horizontal bbox)
#
#   B) Kinematics rules  (Butterworth-filtered hip Y)
#      • downward_velocity > FALL["vel_threshold"]        (fast descent)
#      • abs_acceleration  > FALL["acc_threshold"]        (sudden deceleration)
#
FALL = {
    # ── A: geometry ──
    "angle_threshold":        70.0,   # degrees from vertical (fallen >= 70°)
    "aspect_ratio_threshold": 0.60,   # bbox height/width (horizontal < 0.60)

    # ── B: angle rate gate (KEY discriminator — falls are fast, lying is slow) ──
    # angle_rate = |body_angle_t - body_angle_{t-stride}| / stride_sec
    # Falls: 70-140 deg/sec | Lying down: 2-5 deg/sec | Walking: 8-12 deg/sec
    "angle_rate_threshold":   65.0,   # deg/sec — falls are rapid
    "angle_rate_window":      15,     # frames to look back (1 stride = ~0.79 sec)

    # ── C: kinematics (Butterworth hip velocity — secondary confirmation) ──
    "vel_threshold":          0.30,   # normalized units / second (downward positive)
    "pos_cutoff_hz":          4.0,
    "vel_cutoff_hz":          8.0,

    # ── confirmation ──
    "min_fall_frames":        4,      # consecutive triggered frames to confirm
    "cooldown_seconds":       3.0,    # suppress re-trigger after confirmed fall

    # ── inactivity midpoint threshold (for clip-level eval) ──
    # Found from LOOCV: midpoint between standing centroid and walking centroid
    "inactivity_midpoint_thr": 0.008,
}

# ─── Running Detection (rule-based) ──────────────────────────────────────────────
RUNNING = {
    "step_freq_threshold":             2.0,    # Hz — walk ≈1-1.8, run ≥2
    "horizontal_speed_threshold":      0.020,  # normalized px/frame
    "vertical_oscillation_threshold":  0.010,  # std-dev of CoM Y
    "knee_angle_variance_threshold":   150.0,  # deg² (combined L+R)
    "window_frames":                   30,
    "min_run_frames":                  10,
    "cooldown_seconds":                1.5,
    "min_conditions":                  3,      # min conditions met (out of 4)
    "score_threshold":                 0.55,   # composite weighted score
}

# ─── Inactivity Detection (rule-based) ───────────────────────────────────────────
INACTIVITY = {
    # ── AND-rule thresholds (midpoint between standing and each negative class) ──
    # All three must be satisfied simultaneously → INACTIVE
    #
    # Feature 1: mean keypoint displacement per frame
    #   Standing: ~0.002  |  Walking: ~0.020  |  Midpoint: ~0.011
    "kp_disp_threshold":        0.010,

    # Feature 2: CoM vertical oscillation (std across clip windows)
    #   Standing: ~0.003  |  Jumping:  ~0.038  |  Midpoint: ~0.035
    "com_y_std_threshold":      0.030,

    # Feature 3: body angle stability (std across clip windows, degrees)
    #   Standing: ~1.0°   |  Jumping: ~23.5°, Picking: ~5.9°  |  Midpoint: ~5.5
    "body_angle_std_threshold": 5.5,

    # ── Timing ──
    "inactivity_timeout_seconds": 300.0,   # production: 5 minutes
    "eval_timeout_seconds":       5.0,

    # ── Streaming window ──
    "window_frames":              30,

    # ── Cascade: these activities handled by other detectors ──
    # Act8 (running)  → running detector (93.75%)
    # Act10 (lying)   → fall detector    (92.4%)
}

# ─── Evaluation ──────────────────────────────────────────────────────────────────
EVAL = {
    "test_subjects_fraction": 0.35,
    "random_seed":            42,
}
