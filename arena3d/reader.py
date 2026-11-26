"""Episodic boss reader: learns *this* boss's timing between deaths, the way a player does.

No cross-boss training. During a fight the agent sees only what a player sees: noisy, delayed, dropping-out
3D keypoints of the boss (Sensor, calibrated to the pixel pipeline), the sounds (grunt at a move's start in
60% of moves, whoosh of a fast swing, impact), its own body (hp, flasks, current action).

Memory: every frame it stores a context vector (recent boss-centric pose, its change, time since the last
whoosh and grunt, distance, whether it is healing, boss phase) and, once the future is known, the label
"frames until the next danger" (danger = whoosh, impact, projectile or shockwave hit), censored at 40.
Decision: k nearest contexts in memory give a predictive distribution over time-to-danger:
  * roll when P(danger lands inside the roll's i-frames) is high (roll direction: Thompson bandit);
  * attack when P(danger within the attack's commitment) is low;
  * heal when far and P(danger within the heal) is low (and it learns heals get punished, from labels).
Memory persists across attempts (episodic learning between deaths); `forget=True` is the control.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .agent import Sensor, centric
from .env import HEAL, LIGHT, N_BUTTONS, NONE, ROLL

CAP = 40
ARM = np.r_[9:21]                      # radius + wrist joints in the 39-d centric pose


class EpisodicReader:
    def __init__(self, k=15, roll_thr=0.35, atk_thr=0.25, heal_thr=0.2, forget=False, seed=0, cap_mem=40000,
                 i_lo=3, i_hi=10):
        self.k, self.roll_thr, self.atk_thr, self.heal_thr = k, roll_thr, atk_thr, heal_thr
        self.forget, self.rng, self.cap_mem = forget, np.random.default_rng(seed), cap_mem
        self.i_lo, self.i_hi = i_lo, i_hi
        self.X = np.zeros((0, 0), np.float32)
        self.Y = np.zeros(0, np.float32)
        self.dir_ab = {d: [1.0, 1.0] for d in (3, 7, 5, 1)}      # roll direction bandit: (no-hit, hit)
        self.attempt = 0

    # -- per fight ---------------------------------------------------------------------------------
    def reset(self, env):
        if self.forget:
            self.X, self.Y = np.zeros((0, 0), np.float32), np.zeros(0, np.float32)
        self.sen = Sensor(env, seed=int(self.rng.integers(1 << 30)))
        self.poses = deque(maxlen=16)
        self.pending = deque()               # (t, feature)
        self.t, self.last_whoosh, self.last_grunt = 0, -999, -999
        self.last_x, self.hip, self.h = None, None, 3.0
        self.roll_dir, self.roll_t = None, -999
        self.attempt += 1
        self._mu = self.X.mean(0) if len(self.X) else None
        self._sd = self.X.std(0) + 1e-3 if len(self.X) else None

    def _feature(self, env):
        P, _ = self.sen()
        if P is not None:
            x, hip, fwd, h = centric(P)
            self.last_x, self.hip, self.h = x, hip, h
        self.poses.append(self.last_x)
        ps = [p for p in self.poses if p is not None]
        if not ps:
            return None
        a = np.array(ps)
        m4 = a[-4:].mean(0)
        old = a[:max(1, len(a) - 8)][-4:].mean(0)
        dist = np.linalg.norm(self.hip[:2] - env.p_xy) / self.h if self.hip is not None else 1.0
        tw = min(self.t - self.last_whoosh, 60) / 20.0
        tg = min(self.t - self.last_grunt, 60) / 20.0
        heal = float(env.act == HEAL)
        return np.concatenate([m4, 2.0 * (m4 - old), [3 * tw, 3 * tg, 2 * dist, 3 * heal, 2 * float(env.phase)]]
                              ).astype(np.float32)

    def _predict(self, f):
        """Empirical distribution of time-to-danger among the k nearest remembered contexts (None if no memory)."""
        if len(self.Y) < 200:
            return None
        if self._mu is None:
            self._mu, self._sd = self.X.mean(0), self.X.std(0) + 1e-3
        d = (((self.X - f) / self._sd) ** 2).sum(1)
        nn = np.argpartition(d, self.k)[:self.k]
        return self.Y[nn]

    def act(self, obs, env):
        f = self.f
        if f is None or not env.player_free():
            return 0
        y = self._predict(f)
        dist = np.linalg.norm(self.hip[:2] - env.p_xy) if self.hip is not None else 3.0   # perceived
        reach = 0.8 + 0.25 * env.spec.scale          # own body knowledge: how close the sword needs to be
        if y is None:                                  # first attempt, no memory yet: naive aggression
            return LIGHT if dist <= reach else 1 * N_BUTTONS + NONE
        p_roll = np.mean((y >= self.i_lo) & (y <= self.i_hi))
        if p_roll > self.roll_thr:
            ds = list(self.dir_ab)
            draws = [self.rng.beta(*self.dir_ab[d]) for d in ds]
            self.roll_dir, self.roll_t = ds[int(np.argmax(draws))], self.t
            return self.roll_dir * N_BUTTONS + ROLL
        p30 = np.mean(y < 30)
        if env.hp < 45 and env.flasks and dist > 3.0 and np.mean(y < 50) < self.heal_thr:
            return HEAL
        if dist <= reach:
            return LIGHT if p30 < self.atk_thr else 5 * N_BUTTONS + NONE
        if np.mean(y < 15) > 0.5:
            return 5 * N_BUTTONS + NONE                # danger soon and out of reach: keep away
        return 1 * N_BUTTONS + NONE

    def observe(self, obs, env, info):
        au = obs["audio"]
        danger = bool(au[1] or au[2] or info.get("hit") is not None)
        if au[1]:
            self.last_whoosh = self.t
        if au[0]:
            self.last_grunt = self.t
        if self.roll_dir is not None and self.t - self.roll_t == 20:
            self.dir_ab[self.roll_dir][0] += 1
            self.roll_dir = None
        if self.roll_dir is not None and info.get("hit") is not None:
            self.dir_ab[self.roll_dir][1] += 1
            self.roll_dir = None
        if danger:                                   # label everything still pending
            for (t0, f0) in self.pending:
                self._store(f0, self.t - t0)
            self.pending.clear()
        while self.pending and self.t - self.pending[0][0] >= CAP:
            t0, f0 = self.pending.popleft()
            self._store(f0, CAP)
        self.t += 1
        self.f = self._feature(env)
        if self.f is not None:
            self.pending.append((self.t, self.f))

    def _store(self, f, y):
        if self.X.size == 0:
            self.X = f[None].copy()
            self.Y = np.array([y], np.float32)
            return
        if len(self.Y) >= self.cap_mem:
            i = int(self.rng.integers(len(self.Y)))
            self.X[i], self.Y[i] = f, y
        else:
            self._buf = getattr(self, "_buf", [])
            self._buf.append((f, y))
            if len(self._buf) >= 256:
                self.X = np.vstack([self.X, np.array([b[0] for b in self._buf])])
                self.Y = np.concatenate([self.Y, np.array([b[1] for b in self._buf], np.float32)])
                self._buf = []
                self._mu = None


def run_attempt(env, agent):
    obs = env.reset()
    agent.reset(env)
    agent.f = agent._feature(env)
    done, info = False, {}
    c = {"roll": 0, "light": 0, "heal": 0, "hit": 0, "dodged": 0, "blocked": 0}
    while not done:
        a = agent.act(obs, env)
        b = a % N_BUTTONS
        c["roll"] += b == ROLL; c["light"] += b == LIGHT; c["heal"] += b == HEAL
        obs, r, done, info = env.step(a)
        c["hit"] += info.get("hit") is not None; c["dodged"] += bool(info.get("dodged")); c["blocked"] += bool(info.get("blocked"))
        agent.observe(obs, env, info)
    agent.counts = {k: int(v) for k, v in c.items()}
    return info["outcome"], env.t, env.boss_hp / env.spec.hp, env.hp
