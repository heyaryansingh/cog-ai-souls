"""Open move library: every motion-capture clip is cut into *motion units*; any unit can become a boss move.

No hand-made templates. A unit is the stretch between two lulls of whole-body motion. For each unit we
store its root-relative, yaw-normalized pose track (for motion matching) and what the weapon does in it
(tip speed profile, peak frame). Strikes are not labelled by hand: a unit "can hurt" when the weapon tip
or a limb moves fast, and whether it actually hurts is decided at run time by 3D contact.
Held-out mocap subjects give test bosses whose motions the agent has never seen.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import mocap
from .world import build_world

FEAT_BODIES = ["root", "head", "lhand", "rhand", "lfoot", "rfoot", "lradius", "rradius", "ltibia", "rtibia"]


@dataclass
class Unit:
    clip: str
    start: int
    end: int                      # exclusive
    subject: str
    tip_speed: np.ndarray         # (T,) m/s, unit-scale boss
    peak: int                     # frame of max tip speed (relative)
    peak_speed: float
    reach: float                  # max horizontal tip distance from root (m)
    travel: float                 # root horizontal displacement (m)
    pose0: np.ndarray             # matching feature at start
    pose1: np.ndarray             # matching feature at end
    upright: float                # min head-above-root height (m)
    kind: str = "move"            # "strike" | "move" (data-derived, not hand-labelled)
    tip_rel: np.ndarray = None    # (T,3) weapon tip in the unit's start frame (origin = start root on floor,
                                  #  x = start facing), unit-scale boss
    tags: dict = field(default_factory=dict)

    @property
    def length(self):
        return self.end - self.start


def _yaw(q):
    w, x, y, z = q
    # heading of the body's local -y?  use the projected x-axis of the root frame
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z)]])
    return np.arctan2(R[1, 0], R[0, 0])


def clip_features(m, d, clip):
    T = len(clip)
    bid = [m.body("b_" + n).id for n in FEAT_BODIES]
    tip = m.site("b_tip").id
    P, TIP, ROOT, YAW = np.zeros((T, len(bid), 3)), np.zeros((T, 3)), np.zeros((T, 3)), np.zeros(T)
    for t in range(T):
        d.qpos[:63] = clip.qpos(t)
        mujoco.mj_kinematics(m, d)
        ROOT[t] = d.xpos[bid[0]]
        YAW[t] = _yaw(clip.quat[t])
        c, s = np.cos(-YAW[t]), np.sin(-YAW[t])
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        P[t] = (d.xpos[bid] - ROOT[t] * np.array([1, 1, 0])) @ Rz.T
        TIP[t] = d.site_xpos[tip]
    return P, TIP, ROOT, YAW


def segment(P, min_len=12, max_len=90):
    """Cut at local minima of whole-body speed (lulls)."""
    v = np.linalg.norm(np.diff(P, axis=0), axis=2).sum(1)
    v = np.convolve(v, np.ones(5) / 5, mode="same")
    cuts = [0]
    for t in range(2, len(v) - 2):
        if v[t] <= v[t - 1] and v[t] <= v[t + 1] and v[t] < np.percentile(v, 40) and t - cuts[-1] >= min_len:
            cuts.append(t)
        elif t - cuts[-1] >= max_len:
            cuts.append(t)
    if len(P) - cuts[-1] >= min_len:
        cuts.append(len(P) - 1)
    return list(zip(cuts[:-1], cuts[1:]))


def match_feat(Pt, Pt1):
    """Pose + velocity, for motion matching (per-frame, root-relative)."""
    return np.concatenate([Pt[1:].ravel(), (Pt1[1:] - Pt[1:]).ravel() * 5.0]).astype(np.float32)


def build_library(clips=None, strike_speed=6.0, max_tip=30.0):
    m, _ = build_world(1.0, "greatsword")
    d = mujoco.MjData(m)
    clips = clips or mocap.load()
    units = []
    for name, c in clips.items():
        if len(c) < 20:
            continue
        P, TIP, ROOT, YAW = clip_features(m, d, c)
        for a, b in segment(P):
            tip_rel = TIP[a:b] - ROOT[a:b] * np.array([1, 1, 0])
            sp = np.r_[0, np.linalg.norm(np.diff(TIP[a:b], axis=0), axis=1) / mocap.DT]
            if sp.max() > max_tip:          # retarget glitches / flips
                continue
            up = (P[a:b, 1, 2] - P[a:b, 0, 2]).min()
            if up < 0.15:                   # inverted / lying: keep out of boss moves for now
                continue
            reach = np.linalg.norm((TIP[a:b] - ROOT[a:b])[:, :2], axis=1).max()
            pk = int(sp.argmax())
            f0 = _yaw(c.quat[a]) + np.pi / 2                 # facing at unit start
            cz, sz = np.cos(-f0), np.sin(-f0)
            Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            tip_rel = ((TIP[a:b] - ROOT[a] * np.array([1, 1, 0])) @ Rz.T).astype(np.float32)
            units.append(Unit(name, a, b, name.split("_")[1], sp.astype(np.float32), pk, float(sp.max()), float(reach),
                              float(np.linalg.norm(ROOT[b - 1, :2] - ROOT[a, :2])),
                              match_feat(P[a], P[a + 1]), match_feat(P[b - 2], P[b - 1]), float(up),
                              "strike" if sp.max() >= strike_speed else "move", tip_rel=tip_rel))
    return units


def split_subjects(units, frac_test=0.2, seed=0):
    subs = sorted({u.subject for u in units})
    rng = np.random.default_rng(seed)
    test = set(rng.choice(subs, max(1, int(len(subs) * frac_test)), replace=False))
    return [u for u in units if u.subject not in test], [u for u in units if u.subject in test], sorted(test)


def save(units, path):
    with open(path, "wb") as f:
        pickle.dump(units, f)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)
