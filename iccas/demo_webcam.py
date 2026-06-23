"""
Real-time fall detection demo — webcam.

Usage:
  cd fall_iccas
  python demo_webcam.py
  python demo_webcam.py --exp experiments/subject1_2_3_4
  python demo_webcam.py --camera 1      # external camera index

Keys:
  Q  — quit
  R  — reset fall alert manually
  S  — screenshot
"""

import os
import sys
import json
import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(__file__))
from stgcn import STGCN, PhysicsFilter, TwoStageDetector

# ── COCO 17-joint skeleton edges ──────────────────────────────────────────────
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6),                                     # shoulders
    (5, 7), (7, 9),                             # left arm
    (6, 8), (8, 10),                            # right arm
    (5, 11), (6, 12), (11, 12),                 # torso
    (11, 13), (13, 15),                         # left leg
    (12, 14), (14, 16),                         # right leg
]
KP_COLOR   = (0, 255, 0)
BONE_COLOR = (255, 200, 0)
FALL_COLOR   = (0, 0, 255)
NFALL_COLOR  = (0, 220, 0)
UNSURE_COLOR = (0, 200, 255)

N_JOINTS   = 17
WINDOW     = 30
STRIDE     = 15


def load_detector(exp_dir: str, device: str):
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    cfg_path = os.path.join(ckpt_dir, "two_stage_config.json")
    pth_path = os.path.join(ckpt_dir, "best_stgcn.pth")

    with open(cfg_path) as f:
        cfg = json.load(f)

    fps = cfg.get("fps", 19.0)

    model = STGCN(in_channels=3, num_classes=2, dropout=0.0).to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()

    physics = PhysicsFilter(
        fps=fps,
        vel_threshold=cfg["vel_threshold"],
        acc_threshold=cfg["acc_threshold"],
    )

    detector = TwoStageDetector(
        model, physics,
        stage1_threshold=cfg["stage1_threshold"],
        rescue_threshold=cfg["rescue_threshold"],
        device=device,
    )
    return detector, fps


def extract_kp(result, h, w):
    """Extract normalized (17,3) keypoints from a YOLO result. Zeros if no person."""
    kp = np.zeros((N_JOINTS, 3), dtype=np.float32)
    if result.keypoints is None or len(result.keypoints.xy) == 0:
        return kp
    if result.keypoints.conf is not None:
        idx = int(result.keypoints.conf.sum(dim=1).argmax())
    else:
        idx = 0
    xy   = result.keypoints.xy[idx].cpu().numpy()   # (17,2) pixels
    conf = result.keypoints.conf[idx].cpu().numpy()  # (17,)
    kp[:, 0] = xy[:, 0] / w
    kp[:, 1] = xy[:, 1] / h
    kp[:, 2] = conf
    return kp


def draw_skeleton(frame, kp_norm, h, w, color_kp, color_bone):
    """Draw keypoints and bones on frame from normalized coords."""
    pts = (kp_norm[:, :2] * np.array([w, h])).astype(int)
    vis = kp_norm[:, 2] > 0.2
    for a, b in SKELETON:
        if vis[a] and vis[b]:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color_bone, 2)
    for i in range(N_JOINTS):
        if vis[i]:
            cv2.circle(frame, tuple(pts[i]), 4, color_kp, -1)


def to_stgcn_tensor(seq_np: np.ndarray, device: str) -> torch.Tensor:
    """(T, V, C) → (1, C, T, V, 1) tensor."""
    x = torch.from_numpy(seq_np.transpose(2, 0, 1)).float()  # (C, T, V)
    return x.unsqueeze(0).unsqueeze(-1).to(device)            # (1, C, T, V, 1)


def is_lying(kp: np.ndarray, delta: float = 0.10) -> bool:
    """
    True if body is horizontal — shoulder Y ≈ hip Y (person lying on floor).
    Standing/sitting upright: hip_y - shoulder_y > 0.12
    Lying: hip_y - shoulder_y < delta (shoulders dropped to hip level)
    """
    shoulder_vis = kp[[5, 6], 2].max() > 0.2
    hip_vis      = kp[[11, 12], 2].max() > 0.2
    if not (shoulder_vis and hip_vis):
        return False
    shoulder_y = kp[[5, 6], 1].mean()
    hip_y      = kp[[11, 12], 1].mean()
    return (hip_y - shoulder_y) < delta


def is_standing(kp: np.ndarray, delta: float = 0.12) -> bool:
    """
    Return True if the skeleton looks upright.
    shoulder_y must be significantly above hip_y (in image coords: smaller Y value).
    kp: (17, 3) normalized keypoints
    """
    # joints: shoulders=5,6  hips=11,12
    shoulder_vis = kp[[5, 6], 2].max() > 0.2
    hip_vis      = kp[[11, 12], 2].max() > 0.2
    if not (shoulder_vis and hip_vis):
        return False  # not enough info
    shoulder_y = kp[[5, 6], 1].mean()
    hip_y      = kp[[11, 12], 1].mean()
    # standing: shoulder_y < hip_y (shoulder is higher in the image = smaller Y)
    return (hip_y - shoulder_y) > delta


def ffill(buf: np.ndarray) -> np.ndarray:
    """Forward/backward fill zero frames in (T, V, C) buffer."""
    out = buf.copy()
    T = len(out)
    is_zero = out[:, :, :2].sum(axis=(1, 2)) == 0
    last = None
    for i in range(T):
        if not is_zero[i]:
            last = out[i].copy()
        elif last is not None:
            out[i] = last
    first = None
    for i in range(T):
        if out[i, :, :2].sum() > 0:
            first = out[i].copy()
            break
    if first is not None:
        for i in range(T):
            if out[i, :, :2].sum() == 0:
                out[i] = first
            else:
                break
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",    default=None,
                        help="Experiment directory. Default: experiments/subject1_2_3_4")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default 0)")
    parser.add_argument("--confirm", type=int, default=3,
                        help="Consecutive FALL windows needed to trigger alert (default 3 ≈ 1.5s)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override stage1 fall probability threshold (e.g. 0.90)")
    parser.add_argument("--physics-only", action="store_true",
                        help="Bypass ST-GCN; use only hip velocity/acceleration rule")
    parser.add_argument("--vel-thresh", type=float, default=0.30,
                        help="Physics-only: hip downward velocity threshold (default 0.30 /s)")
    parser.add_argument("--acc-thresh", type=float, default=1.50,
                        help="Physics-only: hip acceleration threshold (default 1.50 /s²)")
    parser.add_argument("--stand-streak", type=int, default=2,
                        help="Consecutive standing windows to auto-reset fall alert (default 2)")
    parser.add_argument("--stand-delta", type=float, default=0.12,
                        help="Min shoulder-hip Y gap to consider standing (default 0.12)")
    parser.add_argument("--drop-thresh", type=float, default=0.10,
                        help="Min net hip drop per window for fast fall (default 0.10)")
    parser.add_argument("--slow-windows", type=int, default=4,
                        help="Windows to check for downward trend in slow fall (default 4 ≈ 3s)")
    parser.add_argument("--slow-drop", type=float, default=0.12,
                        help="Min hip drop FROM STANDING BASELINE to confirm slow fall (default 0.12)")
    parser.add_argument("--fast-fall", action="store_true",
                        help="Also enable fast-fall detection (velocity+acc based). Default OFF.")
    parser.add_argument("--lying-delta", type=float, default=0.10,
                        help="Max shoulder-hip Y gap to consider lying down (default 0.10). Lower = stricter.")
    parser.add_argument("--lying-confirm", type=int, default=2,
                        help="Consecutive windows of lying detection to confirm slow fall (default 2 ≈ 1.5s)")
    parser.add_argument("--min-lock", type=float, default=5.0,
                        help="Minimum seconds to hold FALL DETECTED alert before posture check (default 5)")
    parser.add_argument("--use-model", action="store_true",
                        help="Physics-only mode + ST-GCN confirmation: both must agree (AND logic)")
    parser.add_argument("--stgcn-thresh", type=float, default=0.30,
                        help="ST-GCN prob threshold for model confirmation (default 0.30). Lower = model confirms easier.")
    args = parser.parse_args()

    base = os.path.dirname(__file__)
    exp_dir = args.exp or os.path.join(base, "experiments", "subject1_2_3_4")
    device  = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading detector from: {exp_dir}")
    detector, train_fps = load_detector(exp_dir, device)
    if args.threshold is not None:
        detector.stage1_threshold = args.threshold
        detector.rescue_threshold = args.threshold - 0.05
    confirm_needed = args.confirm
    physics_only   = args.physics_only
    print(f"Detector loaded  (device={device}, train_fps={train_fps})")
    if physics_only:
        mode_str = "PHYSICS+MODEL" if args.use_model else "PHYSICS-ONLY"
        fall_types = "slow+fast" if args.fast_fall else "slow-only"
        print(f"Mode: {mode_str}  [{fall_types}]  slow_drop>{args.slow_drop:.2f}  windows={args.slow_windows}  confirm={confirm_needed}")
    else:
        print(f"Mode: ST-GCN+Physics  stage1_threshold={detector.stage1_threshold:.2f}  confirm={confirm_needed}")

    yolo = YOLO("yolo11n-pose.pt")
    print("YOLO loaded")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.camera}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    buf             = deque(maxlen=WINDOW)
    frame_no        = 0
    prob            = 0.0
    fps_disp        = 0.0
    t_prev          = time.time()
    screenshot_n    = 0
    net_drop        = 0.0
    final_hip_y     = 0.0
    hip_y_now       = 0.0
    drop_from_base  = 0.0
    baseline_hip_y  = 0.0      # EMA of hip Y while standing — personal reference
    lying_streak    = 0        # consecutive windows where person is lying on floor
    model_prob      = -1.0     # last ST-GCN probability (-1 = not run yet)
    feats           = {"max_velocity": 0.0, "max_abs_acc": 0.0, "hip_y_filtered": np.zeros(WINDOW)}
    hip_y_history   = deque(maxlen=10)
    # state machine
    fall_streak     = 0          # consecutive FALL windows (pre-confirm)
    fall_active     = False      # confirmed FALL alert is ON
    fall_type       = ""         # "FAST FALL" or "SLOW FALL"
    fall_lock_until = 0.0        # timestamp: posture reset disabled until this time
    stand_streak    = 0

    print("\nPress Q to quit, S for screenshot\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        frame_no += 1

        # FPS measurement
        now = time.time()
        fps_disp = 0.9 * fps_disp + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now

        # YOLO pose
        results = yolo(frame, verbose=False, conf=0.1)
        kp = extract_kp(results[0], h, w)
        buf.append(kp)

        # Draw skeleton on frame
        draw_skeleton(frame, kp, h, w, KP_COLOR, BONE_COLOR)

        # Inference every STRIDE frames once buffer is full
        if len(buf) == WINDOW and frame_no % STRIDE == 0:
            seq = np.stack(buf, axis=0)   # (T, V, C)
            seq = ffill(seq)

            if physics_only:
                phys = PhysicsFilter(
                    fps=fps_disp if fps_disp > 5 else 25.0,
                    vel_threshold=args.vel_thresh,
                    acc_threshold=args.acc_thresh,
                )
                feats       = phys.extract_features(seq)
                hip_y_f     = feats["hip_y_filtered"]
                net_drop    = float(hip_y_f[-1] - hip_y_f[0])
                final_hip_y = float(hip_y_f[-1])
                hip_y_now   = float(hip_y_f.mean())
                hip_y_history.append(hip_y_now)

                # ── personal standing baseline (EMA, only while upright) ──────
                if not fall_active and is_standing(kp, delta=args.stand_delta):
                    if baseline_hip_y == 0.0:
                        baseline_hip_y = hip_y_now
                    else:
                        baseline_hip_y = 0.88 * baseline_hip_y + 0.12 * hip_y_now

                drop_from_base = max(0.0, hip_y_now - baseline_hip_y) if baseline_hip_y > 0 else 0.0

                # ── fast fall: velocity+acc spike (optional, --fast-fall) ─────
                fast_fall = False
                if args.fast_fall:
                    fast_fall = (
                        feats["max_velocity"] > args.vel_thresh
                        and feats["max_abs_acc"] > args.acc_thresh
                        and net_drop > args.drop_thresh
                    )

                # ── slow fall: hip below baseline AND body gone horizontal ────
                # Floor sitting: hip low BUT shoulders still above → NOT fall
                # Lying after fall: hip low AND shoulders at same level → FALL
                on_floor = baseline_hip_y > 0 and drop_from_base > args.slow_drop
                lying_now = is_lying(kp, delta=args.lying_delta)

                if not fast_fall:
                    if on_floor and lying_now:
                        lying_streak += 1
                    else:
                        lying_streak = 0

                slow_fall = (not fast_fall) and lying_streak >= args.lying_confirm

                current_fall_type = "FAST FALL" if fast_fall else ("SLOW FALL" if slow_fall else "")
                phys_pred = int(fast_fall or slow_fall)

                if args.use_model and phys_pred == 1:
                    # run model for display — physics still controls detection
                    x_t = to_stgcn_tensor(seq, device)
                    with torch.no_grad():
                        logits = detector.model(x_t)
                        model_prob = float(torch.softmax(logits, dim=-1)[0, 1].item())
                    prob = model_prob
                    pred = phys_pred  # physics decides; model shown on screen only
                elif args.use_model and phys_pred == 0:
                    model_prob = -1.0  # physics clear — no need to run model
                    pred = 0
                    prob = min(drop_from_base / (args.slow_drop * 2 + 1e-6), 1.0)
                else:
                    pred = phys_pred
                    prob = min(drop_from_base / (args.slow_drop * 2 + 1e-6), 1.0)
            else:
                current_fall_type = ""
                x_t  = to_stgcn_tensor(seq, device)
                with torch.no_grad():
                    logits = detector.model(x_t)
                    prob   = float(torch.softmax(logits, dim=-1)[0, 1].item())
                pred = detector.predict_one(x_t.squeeze(0), seq)
                if pred == 1:
                    current_fall_type = "FALL"

            if pred == 1:
                fall_streak += 1
                if not fall_active:
                    fall_type = current_fall_type
            else:
                if not fall_active:
                    fall_streak = 0

            # confirm → activate alert with minimum lock
            if fall_streak >= confirm_needed and not fall_active:
                fall_active     = True
                fall_lock_until = time.time() + args.min_lock
                stand_streak    = 0
                print(f"FALL DETECTED [{fall_type}]")

        # ── posture-based auto-reset (only after min-lock expires) ────────────
        now = time.time()
        if fall_active and now > fall_lock_until:
            if is_standing(kp, delta=args.stand_delta):
                stand_streak += 1
                if stand_streak >= args.stand_streak:
                    fall_active  = False
                    fall_streak  = 0
                    stand_streak = 0
                    lying_streak = 0
                    fall_type    = ""
                    print("Auto-reset: person standing again")
            else:
                stand_streak = 0

        # ── display label ─────────────────────────────────────────────────────
        now = time.time()
        if fall_active:
            locked = now < fall_lock_until
            lock_str = f"  locked {int(fall_lock_until - now)}s" if locked else f"  stand {stand_streak}/{args.stand_streak} to reset"
            label = f"FALL DETECTED! [{fall_type}]{lock_str}"
            color = FALL_COLOR
        elif fall_streak > 0:
            label = f"FALL? ({fall_streak}/{confirm_needed})"
            color = (0, 165, 255)   # orange — not alarming
        elif len(buf) == WINDOW:
            label = "NO FALL"
            color = NFALL_COLOR
        else:
            label = "Waiting..."
            color = UNSURE_COLOR

        # ── overlay ──────────────────────────────────────────────────────────
        # top panel — main status
        cv2.rectangle(frame, (0, 0), (w, 65), (20, 20, 20), -1)
        cv2.putText(frame, label, (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

        # top-right — mode badge
        if physics_only and args.use_model:
            badge      = "PHYS + MODEL"
            badge_col  = (100, 220, 100)
        elif physics_only:
            badge      = "PHYSICS ONLY"
            badge_col  = (140, 140, 140)
        else:
            badge      = "ST-GCN + PHYS"
            badge_col  = (100, 180, 255)
        (bw, _), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.putText(frame, badge, (w - bw - 10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, badge_col, 1, cv2.LINE_AA)

        # top-right second line — model prob (only when model ran)
        if args.use_model and model_prob >= 0:
            ml_col  = (0, 200, 80) if model_prob > args.stgcn_thresh else (180, 180, 180)
            ml_text = f"ML: {model_prob:.2f}  thr:{args.stgcn_thresh:.2f}"
            (mw, _), _ = cv2.getTextSize(ml_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.putText(frame, ml_text, (w - mw - 10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, ml_col, 1, cv2.LINE_AA)

        # bottom panel — debug info (2 lines, dark background)
        INFO_H = 52
        cv2.rectangle(frame, (0, h - INFO_H), (w, h), (20, 20, 20), -1)
        if physics_only:
            sh_hip_gap = (kp[[11,12],1].mean() - kp[[5,6],1].mean()
                          if kp[[5,6],2].max() > 0.2 else 0.0)
            posture    = f"LYING {lying_streak}/{args.lying_confirm}" if lying_streak > 0 else "upright"
            line1 = (f"drop={drop_from_base:.3f}/{args.slow_drop:.2f}  "
                     f"gap={sh_hip_gap:.3f}/{args.lying_delta:.2f}  {posture}")
            line2 = f"base={baseline_hip_y:.3f}  streak={fall_streak}/{confirm_needed}  fps={fps_disp:.0f}"
        else:
            line1 = f"prob={prob:.3f}  streak={fall_streak}/{confirm_needed}  fps={fps_disp:.0f}"
            line2 = ""
        cv2.putText(frame, line1, (10, h - INFO_H + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 230), 1, cv2.LINE_AA)
        if line2:
            cv2.putText(frame, line2, (10, h - INFO_H + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160), 1, cv2.LINE_AA)

        # progress bar (buffer fill) — 4px at very bottom
        bar_w = int(w * len(buf) / WINDOW)
        cv2.rectangle(frame, (0, h - 4), (bar_w, h), (60, 180, 60), -1)

        cv2.imshow("MobiCare Fall Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r') or key == ord('R'):
            fall_active  = False
            fall_streak  = 0
            stand_streak = 0
            lying_streak = 0
            print("Fall alert manually reset")
        if key == ord('s'):
            fname = f"screenshot_{screenshot_n:03d}.jpg"
            cv2.imwrite(fname, frame)
            print(f"Saved {fname}")
            screenshot_n += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
