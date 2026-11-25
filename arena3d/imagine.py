"""Anticipation by motion matching: the agent's memory of seen motions predicts the boss's weapon future.

Game engines use motion matching to *synthesise* animation: find the stored frame whose recent pose best
matches, then play what followed. The agent runs it in reverse to *anticipate*: the perceived pose window
is matched against its memory of motions (training-subject mocap only), and the matched continuations give
a sample-based forecast of where the weapon tip will be k frames ahead, in the boss's own frame, in boss
heights. A move it has never seen is predicted by analogy to the most similar motions it knows.
Geometry then decides danger: P(hit at t+k | my planned position) = share of forecasts whose tip passes
within my body's reach at a hittable height.
"""
from __future__ import annotations

import pickle

import mujoco
import numpy as np

from .agent import J13, centric
from .world import build_world

LAGS = (0, 3, 6)
HORIZ = tuple(range(1, 31))


def build_bank(units, clips, allowed, stride=2):
    m, _ = build_world(1.0, "greatsword")
    d = mujoco.MjData(m)
    ids = [m.body("b_" + n).id for n in J13]
    tip = m.site("b_tip").id
    K, V = [], []
    for ui in allowed:
        u = units[ui]
        c = clips[u.clip]
        T = u.length
        P = np.zeros((T, 13, 3)); TIP = np.zeros((T, 3))
        for t in range(T):
            d.qpos[:63] = c.qpos(u.start + t)
            mujoco.mj_kinematics(m, d)
            P[t] = d.xpos[ids]; TIP[t] = d.site_xpos[tip]
        for f in range(max(LAGS), T, stride):
            x0, hip, fwd, h = centric(P[f])
            key = np.concatenate([centric(P[f - l])[0] if l else x0 for l in LAGS])
            cs, sn = np.cos(-fwd), np.sin(-fwd)
            R = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]])
            fut = np.zeros((len(HORIZ), 3), np.float32)
            for j, k in enumerate(HORIZ):
                g = min(f + k, T - 1)
                fut[j] = (TIP[g] - np.array([hip[0], hip[1], 0.0])) @ R.T / h
            K.append(key.astype(np.float32)); V.append(fut)
    return np.array(K), np.array(V)


class Anticipator:
    def __init__(self, K, V, ncomp=24, k=12):
        from sklearn.decomposition import PCA
        from sklearn.neighbors import KDTree
        self.pca = PCA(n_components=ncomp).fit(K[::5])
        self.Z = self.pca.transform(K).astype(np.float32)
        self.tree = KDTree(self.Z)
        self.V, self.k = V, k

    def forecast(self, key):
        z = self.pca.transform(key[None])
        dist, idx = self.tree.query(z, k=self.k)
        w = np.exp(-0.5 * (dist[0] / (dist[0].min() + 1e-3)) ** 2)
        return self.V[idx[0]], w / w.sum()          # (k, len(HORIZ), 3), weights

    def save(self, path):
        pickle.dump(self, open(path, "wb"))

    @staticmethod
    def load(path):
        return pickle.load(open(path, "rb"))


def p_hit(fut, w, local_xy, my_h, body_r=0.17):
    """fut: (k, H, 3) tip forecasts (boss heights); local_xy: (H, 2) my planned positions (boss heights);
    my_h: my height in boss heights. Returns (H,) hit probability."""
    dxy = np.linalg.norm(fut[:, :, :2] - local_xy[None], axis=2)
    z = fut[:, :, 2]
    near = (dxy < body_r) & (z > 0.02) & (z < my_h * 1.05)
    return (near * w[:, None]).sum(0)
