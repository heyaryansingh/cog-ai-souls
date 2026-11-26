"""Recurrent PPO on Arena3D from perceived features (sensor model), the deep-RL baseline for chapter 3.

Observation per decision (all perceived, delayed 180 ms, noisy, with dropout): boss-centric pose now and
3 frames earlier (2 x 39), invariant cues relative to me (38), my position in the boss's frame, boss height,
sounds (3), my body (hp, stamina, flasks, action, phase, i-frames). Actions: the planner's 14 candidates.
Bosses are drawn fresh from the training subjects each episode; held-out subjects for testing.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import threat as T
from .agent import BayesAgent3D, Sensor, centric

CANDS = BayesAgent3D.CANDS
OBS_D = 2 * 39 + T.N_CUE + 2 + 3 + 3 + 6


class Featurizer:
    def __init__(self, env, seed):
        self.env = env
        self.sensor = Sensor(env, 6, seed=seed)
        self.hist = []
        self.frame = (env.boss.root.copy(), 0.0, 2.5)
        self.seen = 0.0

    def tick(self):
        P, _ = self.sensor()
        if P is not None:
            x, hip, fwd, h = centric(P)
            self.frame = (hip[:2].copy(), fwd, h)
            self.hist.append(x)
            self.seen = 1.0
        else:
            self.seen = 0.0
            if self.hist:
                self.hist.append(self.hist[-1])
        self.hist = self.hist[-8:]

    def obs(self, audio, body):
        if not self.hist:
            return np.zeros(OBS_D, np.float32)
        bxy, fwd, h = self.frame
        rel = self.env.p_xy - bxy
        c, s = np.cos(-fwd), np.sin(-fwd)
        me = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]]) / max(h, 0.5)
        P0 = self.hist[-1]
        P3 = self.hist[max(-4, -len(self.hist))]
        P6 = self.hist[max(-7, -len(self.hist))]
        f = np.concatenate([P0, P3, P6, np.zeros(39), P0 - P3, [h / 3]])
        cue = T.cue_feats(f[None], me[None])[0]
        return np.concatenate([P0, P3, cue, me, [np.linalg.norm(me), h / 3, self.seen], audio, body]).astype(np.float32)


class Policy(nn.Module):
    def __init__(self, hidden=256, din=OBS_D):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(din, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.pi = nn.Linear(hidden, len(CANDS))
        self.v = nn.Linear(hidden, 1)
        self.hidden = hidden

    def forward(self, x, h):
        z, h = self.gru(self.inp(x), h)
        return self.pi(z), self.v(z).squeeze(-1), h


class Worker:
    """One env; steps until the next free decision, accumulating reward."""

    def __init__(self, make_env, seed):
        self.make_env, self.rng = make_env, np.random.default_rng(seed)
        self.new()

    def new(self):
        self.env = self.make_env(self.rng)
        self.env.reset()
        self.feat = Featurizer(self.env, int(self.rng.integers(1 << 30)))
        self.audio = np.zeros(3, np.float32)
        self.feat.tick()
        return self.obs()

    def obs(self):
        e = self.env
        body = np.array([e.hp / 100, e.stamina / 100, e.flasks / 3, e.act / 4, min(e.act_t / 30, 1), e.in_iframes()], np.float32)
        return self.feat.obs(self.audio, body)

    def step(self, ai):
        a = CANDS[ai]
        tot, n = 0.0, 0
        while True:
            o, r, done, info = self.env.step(a if n == 0 else 0)
            self.audio = np.maximum(self.audio * 0.5, o["audio"])
            self.feat.tick()
            tot += r
            n += 1
            if done or (self.env.player_free() and n >= 2):
                break
        out = info.get("outcome") if done else None
        return tot / 100.0, done, out, n
