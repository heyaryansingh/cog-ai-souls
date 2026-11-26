"""Reference players for Arena3D: random, aggressive, and an oracle dodger that peeks into the true future."""
from __future__ import annotations

import numpy as np

from .env import GUARD, HEAL, LIGHT, N_ACTIONS, N_BUTTONS, NONE, ROLL


def random_policy(env, rng):
    return int(rng.integers(N_ACTIONS))


def aggro(env, rng):
    dist = np.linalg.norm(env.boss.root - env.p_xy)
    if dist > 0.8 + 0.25 * env.spec.scale:
        return 1 * N_BUTTONS + NONE
    return LIGHT


def _hit_soon(env, a, horizon, seed):
    sim = env.clone(seed)
    for k in range(horizon):
        _, _, done, info = sim.step(a if k == 0 else (0 if a % N_BUTTONS != NONE else a))
        if info["hit"] is not None or (done and sim.hp <= 0):
            return True
        if done:
            return False
    return False


def oracle_dodger(env, rng, horizon=14):
    """Cheats: simulates the true boss future (same RNG) to decide when and where to roll."""
    if not env.player_free():
        return 0
    seed = None   # same RNG stream as the real boss -> exact future
    if _hit_soon(env, NONE, horizon, seed):
        for d in rng.permutation([5, 3, 7, 4, 6, 2, 8]):
            a = int(d) * N_BUTTONS + ROLL
            if not _hit_soon(env, a, horizon, seed):
                return a
        return 5 * N_BUTTONS + ROLL
    if env.hp < 45 and env.flasks and np.linalg.norm(env.boss.root - env.p_xy) > 4:
        return HEAL
    dist = np.linalg.norm(env.boss.root - env.p_xy)
    if dist > 0.8 + 0.25 * env.spec.scale:
        return 1 * N_BUTTONS + NONE
    if not _hit_soon(env, LIGHT, 24, seed):
        return LIGHT
    return 5 * N_BUTTONS + NONE


class CueReflex:
    """Human-like reactive player on *perceived* cues only (sensor model: noise, dropout, 180 ms delay):
    roll sideways when a hand or foot closes on me fast and near; attack when the boss's whole body is
    still (recovery window) and in reach; otherwise close distance or wait. Thresholds in boss heights."""

    def __init__(self, v_th=0.035, r_th=0.75, still_th=0.25, seed=0):
        self.v_th, self.r_th, self.still_th = v_th, r_th, still_th
        self.rng = np.random.default_rng(seed)

    def reset(self, env):
        from .agent import Sensor
        self.env = env
        self.sensor = Sensor(env, 6, seed=int(self.rng.integers(1 << 30)))
        self.hist = []
        self.frame = (env.boss.root.copy(), 0.0, 1.75 * 1.5)
        self.side = 3

    def observe(self, obs, env, info=None):
        from .agent import centric
        P, _ = self.sensor()
        if P is not None:
            x, hip, fwd, h = centric(P)
            self.frame = (hip[:2].copy(), fwd, h)
            self.hist.append(x.reshape(-1, 3))
        elif self.hist:
            self.hist.append(self.hist[-1])
        self.hist = self.hist[-8:]

    def act(self, obs, env):
        if len(self.hist) < 4:
            return 1 * N_BUTTONS + NONE
        bxy, fwd, h = self.frame
        rel = env.p_xy - bxy
        c, s = np.cos(-fwd), np.sin(-fwd)
        me = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]]) / max(h, 0.5)
        P0, P3 = self.hist[-1], self.hist[-4]
        threat = 0.0
        for j in (5, 6, 3, 4, 11, 12):
            v = (P0[j] - P3[j]) / 3
            d = me - P0[j, :2]
            r = np.linalg.norm(d) + 1e-6
            closing = v[:2] @ d / r
            if r < self.r_th and (np.linalg.norm(v) > self.v_th or closing > 0.6 * self.v_th):
                threat = max(threat, np.linalg.norm(v))
        body = np.abs(P0 - P3).sum()
        dist = np.linalg.norm(me)
        if threat > 0 and env.stamina >= 18:
            self.side = 7 if self.side == 3 else 3
            return int(self.rng.choice([self.side, 4 if self.side == 3 else 6])) * N_BUTTONS + ROLL
        if env.hp < 45 and env.flasks and dist > 1.2:
            return HEAL
        if dist < 0.45 and body < self.still_th and env.stamina > 30:
            return LIGHT
        if dist > 0.4:
            return 1 * N_BUTTONS + NONE
        return 5 * N_BUTTONS + GUARD if env.stamina < 40 else 0 * N_BUTTONS + GUARD


def cue_reflex(env, rng, _state={}):
    """Stateless wrapper for scripts/arena3d_baselines.py (keeps one CueReflex per env)."""
    key = id(env)
    if _state.get("key") != key or env.t <= 1:
        _state["key"] = key
        _state["pol"] = CueReflex(seed=int(rng.integers(1 << 30)))
        _state["pol"].reset(env)
        _state["t"] = -1
    pol = _state["pol"]
    if env.t != _state["t"]:
        pol.observe(None, env)
        _state["t"] = env.t
    return pol.act(env, env) if env.player_free() else 0


def state_oracle(env, rng, horizon=14, n=5):
    """Perfect perception of the boss's *current* state, but no clairvoyance: imagines n futures with fresh
    randomness (hold lengths, next-move choices) and rolls if most of them hit. The information ceiling
    for any perception-based agent."""
    if not env.player_free():
        return 0
    seeds = [int(s) for s in rng.integers(1 << 30, size=n)]
    risk = lambda a: np.mean([_hit_soon(env, a, horizon, s) for s in seeds])
    if risk(NONE) >= 0.4:
        best, bv = 5 * N_BUTTONS + ROLL, 2.0
        for d in rng.permutation([5, 3, 7, 4, 6, 2, 8]):
            a = int(d) * N_BUTTONS + ROLL
            r = risk(a)
            if r < bv:
                best, bv = a, r
            if r == 0:
                break
        return best
    if env.hp < 45 and env.flasks and np.linalg.norm(env.boss.root - env.p_xy) > 4:
        return HEAL
    dist = np.linalg.norm(env.boss.root - env.p_xy)
    if dist > 0.8 + 0.25 * env.spec.scale:
        return 1 * N_BUTTONS + NONE
    if risk(LIGHT) < 0.34:
        return LIGHT
    return 5 * N_BUTTONS + NONE
