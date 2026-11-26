"""Learned threat model: P(I am hit k frames from now | what I see of the boss, where I will be).

A fast appraisal network (the "intuition" of chapter 1, here amortised over many training bosses), with a
Bayesian last layer adapted per boss across attempts (learning between deaths).

Input features at decision time t (all from perception, delayed by the sensor latency):
  boss-centric, height-normalised pose now and 3, 6, 9 frames earlier (4 x 39), pose velocity, boss height,
  and the query: my position in the boss's frame at t+k (in boss heights) and k (one-hot over 8 bins).
Label: an unprotected hit lands in [t+k, t+k+2]. Frames where I was protected are excluded (unknown).
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn

LAGS = (0, 3, 6, 9)
KS = (1, 4, 7, 10, 13, 16, 20, 24)          # query horizons (frames)
import os
POSE_D = 42 if os.environ.get("SENSE_TIP") == "1" else 39
FEAT_D = len(LAGS) * POSE_D + POSE_D + 1     # poses + velocity + height
CUE_J = (5, 6, 3, 4, 11, 12)                 # wrists, elbows, feet (J13 order)
N_CUE = len(CUE_J) * 6 + 2
RAW = os.environ.get("THREAT_RAW", "0") == "1"
IN_D = (FEAT_D if RAW else 1) + 2 + len(KS) + 2 + N_CUE


def cue_feats(f, local):
    """Boss-invariant cues a human reads: for each hand/elbow/foot, its height, distance to (my planned)
    position, speed, closing velocity toward me, earlier speed, rise; plus whole-body speed now / before.
    f: (n, FEAT_D) or (FEAT_D,), local: (n, 2) or (2,)."""
    f = np.atleast_2d(f); local = np.atleast_2d(local)
    P0 = f[:, 0:POSE_D].reshape(len(f), -1, 3)[:, :13]
    P3 = f[:, POSE_D:2 * POSE_D].reshape(len(f), -1, 3)[:, :13]
    P6 = f[:, 2 * POSE_D:3 * POSE_D].reshape(len(f), -1, 3)[:, :13]
    out = []
    for j in CUE_J:
        v = (P0[:, j] - P3[:, j]) / 3
        v6 = (P3[:, j] - P6[:, j]) / 3
        rel = local - P0[:, j, :2]
        r = np.linalg.norm(rel, axis=1) + 1e-6
        out += [P0[:, j, 2], r, np.linalg.norm(v, axis=1), (v[:, :2] * rel).sum(1) / r, np.linalg.norm(v6, axis=1),
                P0[:, j, 2] - P3[:, j, 2]]
    out += [np.abs(P0 - P3).sum((1, 2)), np.abs(P3 - P6).sum((1, 2))]
    return np.stack(out, 1)


def query_feats(local, ki, f=None):
    r = np.hypot(local[..., 0], local[..., 1])
    q = np.concatenate([local, np.eye(len(KS))[ki], np.stack([r, np.minimum(r, 3.0) ** 2], -1)], -1)
    if f is None:
        return q
    f2 = np.atleast_2d(f)
    if len(f2) == 1 and np.ndim(local) > 1:
        f2 = np.repeat(f2, len(local), 0)
    base = f2 if RAW else f2[:, -1:]
    return np.concatenate([base, np.atleast_2d(q), cue_feats(f2, np.atleast_2d(local))], 1)


class PoseHistory:
    def __init__(self):
        self.h = deque(maxlen=max(LAGS) + 1)
        self.height = 1.75

    def push(self, x, height):
        if x is None:
            x = self.h[-1] if self.h else np.zeros(POSE_D)
        self.h.append(x)
        if height:
            self.height = 0.9 * self.height + 0.1 * height

    def feat(self):
        if not self.h:
            return np.zeros(FEAT_D, np.float32)
        hs = list(self.h)
        get = lambda lag: hs[max(len(hs) - 1 - lag, 0)]
        P = [get(l) for l in LAGS]
        return np.concatenate(P + [P[0] - P[1], [self.height / 3.0]]).astype(np.float32)


class ThreatNet(nn.Module):
    def __init__(self, hidden=192):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(IN_D, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 64), nn.ReLU())
        self.head = nn.Linear(64, 1)

    def embed(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)


class BayesHead:
    """Per-boss Bayesian logistic last layer (diagonal Laplace / online Newton), prior = trained head."""

    def __init__(self, net: ThreatNet, prior_prec=50.0):
        w = net.head.weight.detach().numpy()[0].astype(np.float64)
        b = float(net.head.bias.detach().numpy()[0])
        self.m = np.concatenate([w, [b]])
        self.m0 = self.m.copy()
        self.prec = np.full_like(self.m, prior_prec)

    def logits(self, E):
        return E @ self.m[:-1] + self.m[-1]

    def update(self, E, y, steps=3):
        """E: (n,64) embeddings, y: (n,) 0/1 outcomes."""
        if len(y) == 0:
            return
        X = np.hstack([E, np.ones((len(E), 1))])
        prec0 = self.prec.copy()
        m = self.m.copy()
        for _ in range(steps):
            p = 1 / (1 + np.exp(-np.clip(X @ m, -30, 30)))
            g = X.T @ (p - y) + prec0 * (m - self.m)
            # diagonal Newton is overconfident with correlated features: damp by the feature count
            H = (X * X).T @ (p * (1 - p)) * X.shape[1] ** 0.5 + prec0
            m = m - np.clip(g / H, -0.5, 0.5)
        self.m = m
        p = 1 / (1 + np.exp(-np.clip(X @ m, -30, 30)))
        self.prec = prec0 + (X * X).T @ (p * (1 - p))
