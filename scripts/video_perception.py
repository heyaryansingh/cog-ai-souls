"""Run the human-like perception pipeline on a gameplay video (e.g. an Elden Ring boss fight from OBS).

    python scripts/video_perception.py fight.mp4 [--every 2] [--max 6000] [--out results/video/fight]

Assumes a souls camera with lock-on: the target is near the screen centre (slightly above), the player's
own character is in the lower-middle. Outputs:
  <out>_pose.npz   per processed frame: time, 33x(x,y,vis) image landmarks, 33x3 metric landmarks, motion energy, onset
  <out>_overlay.mp4  the video with detected boss landmarks drawn (for checking by eye)
  <out>_summary.json detection rate, frames, fps
Then `scripts/video_syllables.py` (next step) segments the boss's motion into syllables.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
from arena3d.perception import Observer, PoseEstimator

EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (0, 11), (0, 12)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--max", type=int, default=100000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--lock", default="0.5,0.42", help="lock-on point (x,y) as screen fractions")
    ap.add_argument("--self_box", default="0.38,0.55,0.62,1.0", help="own character region x0,y0,x1,y1")
    ap.add_argument("--width", type=int, default=640, help="resize width for processing")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join("results", "video", os.path.splitext(os.path.basename(a.video))[0])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    lock = tuple(float(x) for x in a.lock.split(","))
    sbox = tuple(float(x) for x in a.self_box.split(","))
    pe = PoseEstimator(model=a.model) if a.model else PoseEstimator()
    obs = Observer(pe)
    T, K2, K3, ME, ON, OK = [], [], [], [], [], []
    vw, i, t0 = None, 0, time.time()
    while i < a.max:
        ret, frame = cap.read()
        if not ret:
            break
        i += 1
        if i % a.every:
            continue
        h, w = frame.shape[:2]
        if w != a.width:
            frame = cv2.resize(frame, (a.width, int(h * a.width / w)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # follow the boss: once found, aim the fovea at its last centre (lock-on keeps it near centre anyway)
        lk = lock
        if obs.last2 is not None and obs.miss < 5:
            v = obs.last2[:, 2] > 0.3
            if v.sum() >= 6:
                lk = (float(obs.last2[v, 0].mean()), float(obs.last2[v, 1].mean()))
        o = obs(rgb, lk, sbox)
        T.append(i / fps)
        OK.append(o["ok"])
        K2.append(o["kp2"] if o["ok"] else np.full((33, 3), np.nan, np.float32))
        K3.append(o["kp3"] if o["ok"] else np.full((33, 3), np.nan, np.float32))
        ME.append(o["motion"]); ON.append(o["onset"])
        if vw is None:
            vw = cv2.VideoWriter(out + "_overlay.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps / a.every, (frame.shape[1], frame.shape[0]))
        if o["ok"]:
            H, W = frame.shape[:2]
            p = o["kp2"]
            for u, v in EDGES:
                if p[u, 2] > 0.3 and p[v, 2] > 0.3:
                    cv2.line(frame, (int(p[u, 0] * W), int(p[u, 1] * H)), (int(p[v, 0] * W), int(p[v, 1] * H)), (0, 255, 0), 2)
        cv2.putText(frame, f"t={i / fps:6.2f}s {'BOSS' if o['ok'] else '----'} motion={o['motion']:.1f}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        vw.write(frame)
    if vw is not None:
        vw.release()
    np.savez_compressed(out + "_pose.npz", t=np.array(T), kp2=np.array(K2), kp3=np.array(K3), motion=np.array(ME),
                        onset=np.array(ON), ok=np.array(OK))
    s = {"video": a.video, "frames_processed": len(T), "detect_rate": float(np.mean(OK)) if OK else 0.0,
         "proc_fps": len(T) / (time.time() - t0)}
    json.dump(s, open(out + "_summary.json", "w"), indent=1)
    print(s)


if __name__ == "__main__":
    main()
