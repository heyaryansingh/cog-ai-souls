"""Human-like perception from pixels (works on Arena3D renders and on real game video alike).

Two streams, as in primate vision:
  * magnocellular / peripheral: low-resolution frame differencing -> motion energy, its centroid and onset
    ("something is moving fast over there"), fast and cheap;
  * parvocellular / foveal: an off-the-shelf human pose estimator (MediaPipe Pose, 33 landmarks + metric 3D
    "world" landmarks) run on a crop around the attended target, with the crop steered by the last pose
    and by peripheral motion (attention).
Outputs per frame: 2D landmarks (image coords), 3D body-centred landmarks, visibility, motion energy and
onset. A constant-velocity Kalman smoother fills dropouts. Nothing here reads the simulator state.
"""
from __future__ import annotations

import os

import numpy as np

POSE_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "pose",
                          "pose_landmarker_full.task")
# MediaPipe indices used downstream (17, COCO-like)
MP17 = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 19, 20, 31, 32]
MP17_NAMES = ["nose", "lsho", "rsho", "lelb", "relb", "lwri", "rwri", "lhip", "rhip", "lkne", "rkne", "lank",
              "rank", "lidx", "ridx", "lfoot", "rfoot"]


class PoseEstimator:
    def __init__(self, model=POSE_MODEL, num_poses=2, video=True):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        self.mp = mp
        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        opts = vision.PoseLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=model), running_mode=mode,
                                            num_poses=num_poses, min_pose_detection_confidence=0.3,
                                            min_pose_presence_confidence=0.3, min_tracking_confidence=0.3)
        self.det = vision.PoseLandmarker.create_from_options(opts)
        self.video, self.ts = video, 0

    def __call__(self, rgb):
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self.video:
            self.ts += 33
            r = self.det.detect_for_video(img, self.ts)
        else:
            r = self.det.detect(img)
        out = []
        for p2, p3 in zip(r.pose_landmarks, r.pose_world_landmarks):
            a2 = np.array([[l.x, l.y, l.visibility] for l in p2], np.float32)
            a3 = np.array([[l.x, l.y, l.z] for l in p3], np.float32)
            out.append((a2, a3))
        return out


class Magno:
    """Peripheral motion channel: frame difference at 1/4 resolution."""

    def __init__(self, ds=4, onset_k=3.0):
        self.prev, self.ds, self.k = None, ds, onset_k
        self.mu, self.var = 0.0, 1.0

    def __call__(self, rgb):
        g = rgb[::self.ds, ::self.ds].astype(np.float32).mean(2)
        if self.prev is None:
            self.prev = g
            return 0.0, np.array([0.5, 0.5]), False
        diff = np.abs(g - self.prev)
        self.prev = g
        e = float(diff.mean())
        ys, xs = np.nonzero(diff > 12)
        c = np.array([xs.mean() / g.shape[1], ys.mean() / g.shape[0]]) if len(xs) else np.array([0.5, 0.5])
        onset = e > self.mu + self.k * np.sqrt(self.var) and e > 1.0
        self.mu = 0.97 * self.mu + 0.03 * e
        self.var = 0.97 * self.var + 0.03 * (e - self.mu) ** 2
        return e, c, onset


class Observer:
    """Attention + pose + smoothing.

    Two cues any souls player has steer attention: the lock-on reticle (where the target is on screen) and
    knowledge of one's own body (efference copy: the player never mistakes their own character for the
    boss). The fovea is a crop around the reticle, sized by the last boss detection; detections that sit on
    the self region are discarded and the one nearest the reticle is the boss.
    """

    def __init__(self, estimator=None, crop=True, min_half=0.12, mask_self=True, grow=1.6):
        self.pe = estimator or PoseEstimator()
        self.mag = Magno()
        self.crop, self.min_half, self.mask_self, self.grow = crop, min_half, mask_self, grow
        self.reset()

    def reset(self):
        self.half, self.last2, self.last3, self.vel3, self.miss = 0.3, None, None, None, 0
        self.mag = Magno()

    def __call__(self, rgb, lock_xy=None, self_box=None):
        H, W = rgb.shape[:2]
        e, c, onset = self.mag(rgb)
        lock = np.array(lock_xy if lock_xy is not None else (0.5, 0.45), float)
        x0 = y0 = 0
        img = rgb
        if self.crop:
            hp = int(np.clip(self.half * self.grow, self.min_half, 0.5) * H)
            cx, cy = int(lock[0] * W), int(lock[1] * H)
            x0, y0 = max(cx - hp, 0), max(cy - hp, 0)
            x1, y1 = min(cx + hp, W), min(cy + hp, H)
            img = rgb[y0:y1, x0:x1]
            if self_box is not None and self.mask_self:      # paint out the self region inside the fovea (self-suppression)
                img = img.copy()
                sx0, sy0, sx1, sy1 = self_box
                a0, b0 = int(sx0 * W) - x0, int(sy0 * H) - y0
                a1, b1 = int(sx1 * W) - x0, int(sy1 * H) - y0
                a0, b0, a1, b1 = max(a0, 0), max(b0, 0), min(a1, img.shape[1]), min(b1, img.shape[0])
                if a1 > a0 and b1 > b0:
                    img[b0:b1, a0:a1] = img.reshape(-1, 3).mean(0).astype(img.dtype)
            if img.shape[0] < 160:   # upsample small foveal crops (more pixels for the pose network)
                k = int(np.ceil(160 / img.shape[0]))
                img = np.repeat(np.repeat(img, k, 0), k, 1)
            sx, sy = (x1 - x0) / img.shape[1], (y1 - y0) / img.shape[0]
        else:
            sx = sy = 1.0
        dets = self.pe(img)
        best, bd = None, 1e9
        for a2, a3 in dets:
            a2 = a2.copy()
            a2[:, 0] = (a2[:, 0] * img.shape[1] * sx + x0) / W
            a2[:, 1] = (a2[:, 1] * img.shape[0] * sy + y0) / H
            vis = a2[:, 2] > 0.3
            if vis.sum() < 6:
                continue
            ctr = np.array([a2[vis, 0].mean(), a2[vis, 1].mean()])
            if self_box is not None and self_box[0] < ctr[0] < self_box[2] and self_box[1] < ctr[1] < self_box[3]:
                continue
            dd = np.linalg.norm(ctr - lock)
            if dd < bd:
                best, bd = (a2, a3, vis), dd
        if best is not None:
            a2, a3, vis = best
            ys = a2[vis, 1]
            self.half = float(0.75 * np.ptp(ys) + 0.04)
            if self.last3 is not None:
                self.vel3 = 0.5 * (self.vel3 if self.vel3 is not None else 0) + 0.5 * (a3 - self.last3)
            self.last2, self.last3, self.miss = a2, a3, 0
            ok = True
        else:
            self.miss += 1
            ok = False
            self.half = min(0.5, self.half * 1.15)       # widen the fovea while searching
            if self.last3 is not None and self.vel3 is not None:
                self.last3 = self.last3 + 0.5 * self.vel3   # coast
        return {"ok": ok, "kp2": self.last2, "kp3": self.last3, "motion": e, "motion_c": c, "onset": onset}
