"""
Capture demo screenshots: fall, running (KTH full-body), inactivity.
Running: scans middle 50% of clips, picks frame where head+shoulders visible.

Usage:
    python capture_demo.py
"""

import os, glob
import cv2
import numpy as np
from ultralytics import YOLO
from src.pose_extractor import PoseFrame
from fall_detection.detector import FallDetector
from running_detection.detector import RunningDetector
from inactivity_detection.detector import InactivityDetector
from src.config import YOLO_POSE_MODEL, UPFALL_DATASET_PATH, KTH_DATA_DIR

UPFALL = UPFALL_DATASET_PATH
KTH    = KTH_DATA_DIR
OUT    = "results"
os.makedirs(OUT, exist_ok=True)

COLORS = {"FALL":(0,0,255), "RUNNING":(0,140,255), "INACTIVITY":(255,165,0)}
SKEL   = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,11),(6,12),(11,12),
          (5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16)]


def extract(result, h, w):
    if not result.boxes or not result.keypoints: return None, None
    n = len(result.keypoints.xy)
    if n == 0: return None, None
    ci = int(np.argmax([result.keypoints.conf[i].sum().item() for i in range(n)]))
    xy = result.keypoints.xy[ci].cpu().numpy()
    cf = result.keypoints.conf[ci].cpu().numpy()
    kp = np.zeros((17,3), np.float32)
    kp[:,0] = xy[:,0]/max(w,1); kp[:,1] = xy[:,1]/max(h,1); kp[:,2] = cf
    bb = result.boxes.xyxy[ci].cpu().numpy().astype(int)
    return bb, kp


def vis_count(kp): return int((kp[:,2]>0.2).sum()) if kp is not None else 0


def draw(frame, bbox, kp, alert=None, color=(0,0,255), still=None):
    out = frame.copy(); h,w = out.shape[:2]
    if bbox is not None:
        x1,y1,x2,y2 = bbox
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(out,(x1,y1-24),(x1+120,y1),(0,255,0),-1)
        cv2.putText(out,"Worker #1",(x1+4,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,0),2)
    if kp is not None:
        pts = (kp[:,:2]*np.array([w,h])).astype(int); vis=kp[:,2]>0.2
        for a,b in SKEL:
            if vis[a] and vis[b]: cv2.line(out,tuple(pts[a]),tuple(pts[b]),(255,200,0),2)
        for i in range(17):
            if vis[i]: cv2.circle(out,tuple(pts[i]),4,(0,255,0),-1)
    if alert:
        cv2.rectangle(out,(0,0),(w,60),(20,20,20),-1)
        cv2.putText(out,alert,(12,46),cv2.FONT_HERSHEY_SIMPLEX,1.1,color,3,cv2.LINE_AA)
    if still is not None:
        cv2.rectangle(out,(0,h-28),(w,h),(30,30,30),-1)
        cv2.putText(out,f"Worker #1  Still: {still:.1f}s",(8,h-8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.65,(200,200,200),1)
    return out


def upscale(img, min_w=600):
    h,w = img.shape[:2]
    if w < min_w:
        s = min_w // w + 1
        img = cv2.resize(img,(w*s,h*s),interpolation=cv2.INTER_LINEAR)
    return img


def get_png(subject, activity, trial=1, cam=1):
    return sorted(glob.glob(os.path.join(
        UPFALL,f"Subject{subject}",f"Activity{activity}",f"Trial{trial}",
        f"Subject{subject}Activity{activity}Trial{trial}Camera{cam}","*.png")))


# ── Fall ─────────────────────────────────────────────────────────────────────
def cap_fall(yolo):
    print("Fall (Subject1 Activity1)...")
    files = get_png(1,1,1)
    det = FallDetector(fps=17.)
    out = None
    for i,p in enumerate(files):
        img = cv2.imread(p)
        if img is None: continue
        h,w = img.shape[:2]
        r = yolo(img,conf=0.1,verbose=False)[0]
        bb,kp = extract(r,h,w)
        pf = PoseFrame(kp,i,i/17.,True,-1) if kp is not None \
             else PoseFrame(np.zeros((17,3),np.float32),i,i/17.,False,-1)
        ev = det.update(pf)
        out = draw(img,bb,kp)
        if ev:
            out = draw(img,bb,kp,
                f"FALL DETECTED  |  angle={ev.body_angle:.0f}  rate={ev.angle_rate:.0f} deg/s",
                COLORS["FALL"])
            cv2.imwrite(f"{OUT}/demo_fall.png", upscale(out))
            print(f"  Saved frame {i}"); return
    if out is not None:
        cv2.imwrite(f"{OUT}/demo_fall.png", upscale(out))


# ── Running ──────────────────────────────────────────────────────────────────
def cap_running(yolo):
    """Scan fire1.mp4 for the best UNSAFE RUNNING frame (full body visible)."""
    src = os.path.join(OUT, "fire1.mp4")
    if not os.path.exists(src):
        print(f"  fire1.mp4 not found at {src}"); return

    print(f"Running (fire1.mp4 — scanning for UNSAFE RUNNING with full body)...")
    cap = cv2.VideoCapture(src)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    det   = RunningDetector(fps=fps)

    best_score, best_frame, best_bb, best_kp = 0, None, None, None
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        h, w = frame.shape[:2]
        r = yolo(frame, conf=0.40, verbose=False)[0]
        if not r.boxes or not r.keypoints:
            i += 1; continue

        # Pick person with most visible keypoints
        n = len(r.boxes)
        best_p = max(range(n), key=lambda x: r.keypoints.conf[x].sum().item())
        xy = r.keypoints.xy[best_p].cpu().numpy()
        cf = r.keypoints.conf[best_p].cpu().numpy() if r.keypoints.conf is not None \
             else np.ones(17)
        kp = np.zeros((17, 3), np.float32)
        kp[:, 0] = xy[:, 0] / max(w, 1)
        kp[:, 1] = xy[:, 1] / max(h, 1)
        kp[:, 2] = cf
        bb = r.boxes.xyxy[best_p].cpu().numpy().astype(int)

        pf = PoseFrame(kp, i, i/fps, True, -1)
        ev = det.update(pf)

        head_ok = kp[0, 2] > 0.3
        sh_ok   = kp[5, 2] > 0.3 and kp[6, 2] > 0.3
        hip_ok  = kp[11, 2] > 0.3 and kp[12, 2] > 0.3
        vis     = int((kp[:, 2] > 0.2).sum())
        score   = vis + (8 if head_ok else 0) + (6 if sh_ok else 0) + (4 if hip_ok else 0)

        if ev and score > best_score:
            best_score = score
            best_frame = frame.copy()
            best_bb, best_kp = bb.copy(), kp.copy()
            print(f"  Best so far: f={i} ({i/fps:.1f}s) kp={vis} "
                  f"head={head_ok} sh={sh_ok} score={score}")
        i += 1
    cap.release()

    if best_frame is not None:
        out_img = draw(best_frame, best_bb, best_kp,
                       "UNSAFE RUNNING  |  high speed detected",
                       COLORS["RUNNING"])
        cv2.imwrite(f"{OUT}/demo_running.png", out_img)
        print(f"  Saved: {OUT}/demo_running.png  ({best_frame.shape[1]}x{best_frame.shape[0]})")
    else:
        print("  No RUNNING event detected in fire1.mp4")


# ── Inactivity ────────────────────────────────────────────────────────────────
def cap_inactivity(yolo):
    print("Inactivity (Subject1 Activity7 standing)...")
    files = get_png(1,7,1)
    det = InactivityDetector(fps=17.,eval_mode=True)
    out = None
    for i,p in enumerate(files[30:],30):
        img = cv2.imread(p)
        if img is None: continue
        h,w = img.shape[:2]
        r = yolo(img,conf=0.1,verbose=False)[0]
        bb,kp = extract(r,h,w)
        pf = PoseFrame(kp,i,i/17.,True,-1) if kp is not None \
             else PoseFrame(np.zeros((17,3),np.float32),i,i/17.,False,-1)
        ev = det.update(pf)
        still = float(det.still_duration_sec())
        out = draw(img,bb,kp,still=still)
        if ev and vis_count(kp) >= 12:
            out = draw(img,bb,kp,
                f"INACTIVITY DETECTED  |  still={ev.duration_sec:.1f}s",
                COLORS["INACTIVITY"], still=still)
            cv2.imwrite(f"{OUT}/demo_inactivity.png", upscale(out))
            print(f"  Saved frame {i}"); return
    if out is not None:
        cv2.imwrite(f"{OUT}/demo_inactivity.png", upscale(out))


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading YOLO11s-pose...")
    yolo = YOLO(YOLO_POSE_MODEL)
    cap_fall(yolo)
    cap_running(yolo)
    cap_inactivity(yolo)
    print("\nDone: results/demo_fall.png  demo_running.png  demo_inactivity.png")
