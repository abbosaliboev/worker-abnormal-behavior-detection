"""
Full video test — multi-person running detection with false positive filtering.

Fixes:
  1. Multi-person: detects ALL persons in frame, each gets own detector
  2. False positive filter:
       - bbox must be taller than wide (human shape)
       - minimum 8 keypoints visible
       - bounding box height > 10% of frame (not tiny object)

Usage:
    python test_video.py
    python test_video.py --source results/fire1.mp4
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from src.pose_extractor import PoseFrame
from running_detection.detector import RunningDetector
from src.config import YOLO_POSE_MODEL

SKEL = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,11),(6,12),(11,12),
        (5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16)]

TRACK_COLORS = [
    (0,255,0),(0,200,255),(255,100,0),(180,0,255),(0,255,180)
]


def is_human(bb, kp, h, w):
    """
    Strict human validation — rejects fire/smoke/objects that look human-shaped.
    """
    x1, y1, x2, y2 = bb
    bh = y2 - y1
    bw = x2 - x1

    # 1. Shape: person must be taller than wide
    if bw > bh * 1.3:
        return False

    # 2. Size: at least 10% of frame height
    if bh < h * 0.10:
        return False

    # 3. At least 10 keypoints visible
    vis_mask = kp[:, 2] > 0.25
    if vis_mask.sum() < 10:
        return False

    # 4. Average keypoint confidence must be decent
    if kp[vis_mask, 2].mean() < 0.35:
        return False

    # 5. Anatomical check: head must be ABOVE hips (in image coords: smaller y)
    #    Nose (0), Left hip (11), Right hip (12)
    nose_vis  = kp[0, 2]  > 0.2
    lhip_vis  = kp[11, 2] > 0.2
    rhip_vis  = kp[12, 2] > 0.2
    if nose_vis and (lhip_vis or rhip_vis):
        hip_y  = kp[11, 1] if lhip_vis else kp[12, 1]
        nose_y = kp[0, 1]
        # head must be at least 15% of frame height above hips
        if nose_y >= hip_y - 0.15:
            return False

    # 6. Core body parts must be present:
    #    at least 1 shoulder visible AND at least 1 hip visible
    shoulder_vis = (kp[5, 2] > 0.25 or kp[6, 2] > 0.25)
    hip_vis      = (kp[11, 2] > 0.25 or kp[12, 2] > 0.25)
    if not (shoulder_vis and hip_vis):
        return False

    return True


def extract_all(result, h, w, conf_thr=0.25):
    """Extract ALL valid persons from YOLO result."""
    persons = []
    if not result.boxes or not result.keypoints:
        return persons
    n = len(result.boxes)
    for i in range(n):
        # Box confidence threshold
        box_conf = float(result.boxes.conf[i].cpu())
        if box_conf < conf_thr:
            continue
        bb  = result.boxes.xyxy[i].cpu().numpy().astype(int)
        xy  = result.keypoints.xy[i].cpu().numpy()
        cf  = result.keypoints.conf[i].cpu().numpy() if result.keypoints.conf is not None \
              else np.ones(17)
        kp  = np.zeros((17,3), np.float32)
        kp[:,0] = xy[:,0]/max(w,1)
        kp[:,1] = xy[:,1]/max(h,1)
        kp[:,2] = cf
        if is_human(bb, kp, h, w):
            persons.append((bb, kp))
    return persons


def draw_person(out, bb, kp, color, label, w, h):
    x1,y1,x2,y2 = bb
    cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
    cv2.rectangle(out,(x1,max(0,y1-28)),(x1+len(label)*11,y1),color,-1)
    cv2.putText(out,label,(x1+4,y1-7),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,0),2)
    pts = (kp[:,:2]*np.array([w,h])).astype(int)
    vis = kp[:,2] > 0.2
    for a,b in SKEL:
        if vis[a] and vis[b]:
            cv2.line(out,tuple(pts[a]),tuple(pts[b]),(255,200,0),2)
    for i in range(17):
        if vis[i]:
            cv2.circle(out,tuple(pts[i]),4,color,-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="results/fire1.mp4")
    parser.add_argument("--output", default="results/fire1_annotated.mp4")
    args = parser.parse_args()

    print("=" * 60)
    print("  Full Video Test — Multi-person Running Detection")
    print(f"  Input : {args.source}")
    print(f"  Output: {args.output}")
    print("=" * 60)

    cap   = cv2.VideoCapture(args.source)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {W}x{H}  |  {fps:.0f}fps  |  {total/fps:.1f}s  ({total} frames)\n")

    writer = cv2.VideoWriter(args.output,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))

    yolo = YOLO(YOLO_POSE_MODEL)

    detectors:  dict[int, RunningDetector] = {}
    statuses:   dict[int, str] = {}
    all_events = []
    i = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        res     = yolo(frame, conf=0.40, verbose=False)[0]
        persons = extract_all(res, H, W, conf_thr=0.40)

        out = frame.copy()
        ts  = i / fps

        # Sort persons left-to-right for consistent labeling
        persons.sort(key=lambda p: p[0][0])

        any_running    = False
        running_labels = []
        detected_pids = set()

        for idx, (bb, kp) in enumerate(persons):
            pid = idx
            detected_pids.add(pid)
            if pid not in detectors:
                detectors[pid] = RunningDetector(fps=fps)
                statuses[pid]  = "NORMAL"

            pf = PoseFrame(kp, i, ts, True, pid)
            ev = detectors[pid].update(pf)

            HOLD_SEC = 5.0   # keep RUNNING alert for 5 seconds after last event

            if ev:
                statuses[pid] = "RUNNING"
                all_events.append((ts, f"Worker #{pid+1}"))
                print(f"  [{ts:6.1f}s] UNSAFE RUNNING — Worker #{pid+1}  (frame {i})")
                running_labels.append(f"Worker #{pid+1}")
                any_running = True
            elif statuses.get(pid) == "RUNNING":
                # Only reset after HOLD_SEC seconds since last event
                last = max((e[0] for e in all_events
                            if e[1] == f"Worker #{pid+1}"), default=0)
                if ts - last > HOLD_SEC:
                    statuses[pid] = "NORMAL"
                else:
                    # Still in hold period — keep showing RUNNING
                    running_labels.append(f"Worker #{pid+1}")
                    any_running = True

            status = statuses.get(pid, "NORMAL")
            color  = (0,140,255) if status == "RUNNING" else TRACK_COLORS[pid % len(TRACK_COLORS)]
            label  = f"W#{pid+1} {'RUN' if status=='RUNNING' else 'OK'}"
            draw_person(out, bb, kp, color, label, W, H)

        # Alert banner
        cv2.rectangle(out,(0,0),(W,60),(20,20,20),-1)
        if any_running or running_labels:
            who = ", ".join(running_labels) if running_labels else ""
            banner = f"UNSAFE RUNNING DETECTED  |  {who}  [{ts:.1f}s]"
            cv2.putText(out, banner, (12,44),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,140,255), 3, cv2.LINE_AA)
        else:
            pcount = len(persons)
            cv2.putText(out, f"Normal  |  {pcount} worker(s) detected  [{ts:.1f}s]",
                        (12,44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,200,200), 2, cv2.LINE_AA)

        # Bottom bar
        cv2.rectangle(out,(0,H-30),(W,H),(30,30,30),-1)
        cv2.putText(out, f"Frame {i}/{total}  |  Persons: {len(persons)}  |  Events: {len(all_events)}",
                    (8,H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)

        writer.write(out)
        if i % 150 == 0:
            print(f"  Progress: {i}/{total}  ({ts:.1f}s)  persons={len(persons)}")
        i += 1

    cap.release()
    writer.release()

    print(f"\n{'='*60}")
    print(f"  Done!  Output: {args.output}")
    print(f"\n  Running events detected ({len(all_events)} total):")
    for ts, who in all_events:
        print(f"    [{ts:6.1f}s]  {who}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
