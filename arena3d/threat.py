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
IN_D = FEAT_D + 2 + len(KS) + 2              # + local xy + k one-hot + (r, cos, sin features) -> r and angle


def query_feats(local, ki):
    r = np.hypot(local[..., 0], local[..., 1])
    return np.concatenate([local, np.eye(len(KS))[ki], np.stack([r, np.minimum(r, 3.0) ** 2], -1)], -1)


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
            p = 1 / (1 + np.exp(-(X @ m)))
            g = X.T @ (p - y) + prec0 * (m - self.m)
            H = (X * X).T @ (p * (1 - p)) + prec0
            m = m - g / H
        self.m = m
        p = 1 / (1 + np.exp(-(X @ m)))
        self.prec = prec0 + (X * X).T @ (p * (1 - p))
