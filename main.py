"""
Real-time Worker Abnormal Behavior Detection — Multi-person with Tracking.

Usage:
    python main.py                        # webcam
    python main.py --source video.mp4     # video file
    python main.py --source rtsp://...    # RTSP stream
    python main.py --no-tracking          # disable tracking (single person)
    python main.py --source video.mp4 --save out.mp4

Keys:
    Q — quit
    R — reset all detectors
"""

import argparse
import sys
import time

import cv2

from src.behavior_monitor import BehaviorMonitor
from src.config import STGCN_FPS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source",       default=0)
    p.add_argument("--fps",          type=float, default=0.0)
    p.add_argument("--no-display",   action="store_true")
    p.add_argument("--no-tracking",  action="store_true",
                   help="Disable ByteTracker (single person mode)")
    p.add_argument("--eval-mode",    action="store_true",
                   help="Short inactivity timeout for demo")
    p.add_argument("--save",         default="")
    return p.parse_args()


def main():
    args = parse_args()

    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open '{source}'")
        sys.exit(1)

    native_fps = cap.get(cv2.CAP_PROP_FPS) or STGCN_FPS
    fps = args.fps if args.fps > 0 else native_fps
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (W, H))

    use_tracking = not args.no_tracking
    monitor = BehaviorMonitor(fps=fps, eval_mode=args.eval_mode,
                              use_tracking=use_tracking)

    mode = "Multi-person tracking" if use_tracking else "Single-person"
    print(f"Source  : {source}  |  FPS: {fps:.1f}  |  Mode: {mode}")
    print("Press Q to quit, R to reset.\n")

    frame_idx   = 0
    all_alerts  = []
    t0 = time.time()

    with monitor:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Stream ended.")
                break

            new_alerts = monitor.update(frame)
            for a in new_alerts:
                print(f"  [{a.timestamp_sec:7.2f}s] {a.message}")
                all_alerts.append(a)

            if not args.no_display or writer:
                annotated = monitor.annotate(frame, all_alerts[-5:])

            if writer:
                writer.write(annotated)

            if not args.no_display:
                cv2.imshow("Worker Behavior Monitor", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    monitor.reset()
                    all_alerts.clear()
                    print("Reset.")

            frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t0
    alerts  = monitor.all_alerts
    print(f"\n--- Session summary ---")
    print(f"  Frames : {frame_idx}  |  Time: {elapsed:.1f}s")
    print(f"  Alerts : {len(alerts)}")
    for t in ("FALL", "RUNNING", "INACTIVITY"):
        n = sum(1 for a in alerts if a.alert_type == t)
        print(f"    {t:<12}: {n}")


if __name__ == "__main__":
    main()
