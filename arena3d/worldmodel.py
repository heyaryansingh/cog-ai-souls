"""R3: a learned world model of the boss's moves, used instead of the true simulator for planning.

For every move the agent has identified, it keeps a spatio-temporal danger map learned from what it felt:
    p(contact | move m, frames since the move started tau, my position in the boss's frame at move onset)
binned over tau (3-frame bins up to 90), distance (0.5 m rings to 6 m) and bearing (8 sectors). Exposures are
counted every frame the player was within 6 m; contacts are hits, blocked hits and attacks felt passing
through i-frames. Planning imagines the player's own body (the self-model: walk, roll with i-frames, attack
commitment, heal) against this map, with the move identity drawn from the learned strings until the move is
readable (12 frames, 70% accurate: same perception limits as before). No boss simulator is used to plan.
"""
from __future__ import annotations

import numpy as np

from .env import HEAL, LIGHT, N_BUTTONS, NONE, ROLL

TB, RB, AB = 30, 12, 8           # 3-frame tau bins (90 frames), 0.5 m rings, 8 bearing sectors
R_MAX = 6.0


def _bins(tau, rel):
    r = np.hypot(rel[..., 0], rel[..., 1])
    a = np.arctan2(rel[..., 1], rel[..., 0])
    ti = np.clip(np.asarray(tau) // 3, 0, TB - 1).astype(int)
    ri = np.clip((r / 0.5).astype(int), 0, RB - 1)
    ai = ((a + np.pi) / (2 * np.pi) * AB).astype(int) % AB
    return ti, ri, ai, r < R_MAX


class DangerMaps:
    def __init__(self, n_moves, prior_hit=0.005, prior_n=2.0, seed=0, read_k=12, acc=0.7):
        self.N = np.zeros((n_moves, TB, RB, AB)); self.H = np.zeros((n_moves, TB, RB, AB))
        self.a, self.b = prior_hit * prior_n, prior_n
        self.rng = np.random.default_rng(seed); self.read_k, self.acc, self.n = read_k, acc, n_moves
        self.update = True

    # --- learning from felt contacts -----------------------------------------------------------------
    def reset(self, env):
        self.serial, self.origin, self.fwd, self.label, self.onset = -1, None, 0.0, None, 0
        self.pend = []                      # (tau, rel) for the current move before it is identified

    def rel(self, env, p=None):
        p = env.p_xy if p is None else p
        d = p - self.origin; c, s = np.cos(-self.fwd), np.sin(-self.fwd)
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])

    def observe(self, env, info):
        b = env.boss
        s = getattr(env, "boss_serial", -1)
        if b.kind == "strike" and s != self.serial:
            self.serial, self.onset, self.label, self.pend = s, env.t, None, []
            to_p = env.p_xy - b.root
            self.origin = b.root + self.rng.normal(0, 0.1, 2)
            self.fwd = float(np.arctan2(to_p[1], to_p[0])) + self.rng.normal(0, 0.1)   # bosses face you at onset
        if self.origin is None or b.kind != "strike" or s != self.serial:
            return
        tau = env.t - self.onset
        if tau >= self.read_k and self.label is None:
            self.label = b.move_id if self.rng.random() < self.acc else int(self.rng.integers(self.n))
        hit = bool(info.get("hit") is not None or info.get("blocked") or info.get("dodged"))
        self.pend.append((tau, self.rel(env), hit))
        if self.label is not None and self.update:
            for (t0, r0, h0) in self.pend:
                ti, ri, ai, ok = _bins(t0, r0)
                if ok:
                    self.N[self.label, ti, ri, ai] += 1; self.H[self.label, ti, ri, ai] += h0
            self.pend = []

    def hazard(self, m, tau, rel):
        ti, ri, ai, ok = _bins(tau, rel)
        p = (self.H[m, ti, ri, ai] + self.a) / (self.N[m, ti, ri, ai] + self.b)
        return np.where(ok & (np.asarray(tau) < 3 * TB), p, 0.0)


def self_model(env, a, H):
    """The player's own body over H frames for action a (positions, i-frame mask, commitment) — known controls."""
    d_i, btn = divmod(int(a), N_BUTTONS)
    to_b = env.boss.root - env.p_xy
    face = np.arctan2(to_b[1], to_b[0])
    pos = np.zeros((H, 2)); inv = np.zeros(H, bool); p = env.p_xy.copy()
    if btn == ROLL:
        dd = d_i if d_i else 5; ang = face + (dd - 1) * np.pi / 4; v = np.array([np.cos(ang), np.sin(ang)])
    elif d_i and btn == NONE:
        ang = face + (d_i - 1) * np.pi / 4; v = np.array([np.cos(ang), np.sin(ang)])
    else:
        v = np.zeros(2)
    for t in range(H):
        if btn == ROLL:
            p = p + v * (6.5 * 0.03 if t < 14 else 0.0); inv[t] = 2 <= t <= 11
        elif btn == NONE and t < 6:
            p = p + v * 4.0 * 0.03
        elif btn == HEAL:
            p = p + 0.0
        pos[t] = p
    return pos, inv


class LearnedModelPlanner:
    """MPC over own macro-actions using DangerMaps (learned) + learned strings, no boss simulator."""

    def __init__(self, maps, grammar, H=30, hit_cost=125.0, dmg=40.0, seed=0, n=8):
        self.maps, self.g, self.H, self.hc, self.dmg, self.n = maps, grammar, H, hit_cost, dmg, n
        self.rng = np.random.default_rng(seed)
        self.hold, self.hold_n = None, 0

    def reset(self, env):
        self.maps.reset(env); self.g.reset(env); self.hold, self.hold_n = None, 0

    def _move_samples(self, env):
        """(move id, tau) samples for the current boss move; None if the boss isn't attacking."""
        b = env.boss; mp = self.maps
        if b.kind != "strike" or mp.origin is None:
            return None, 0
        tau = env.t - mp.onset
        if mp.label is not None:
            return [mp.label] * self.n, tau
        return [self.g.sample(env, self.rng) for _ in range(self.n)], tau

    def act(self, env):
        self.g.observe(env)
        if not env.player_free():
            return 0
        if self.hold_n > 0:
            self.hold_n -= 1
            return self.hold
        acts = [NONE, 1 * N_BUTTONS + NONE, 5 * N_BUTTONS + NONE, 3 * N_BUTTONS + NONE, 7 * N_BUTTONS + NONE,
                LIGHT, 3 * N_BUTTONS + ROLL, 7 * N_BUTTONS + ROLL, 5 * N_BUTTONS + ROLL]
        if env.flasks and env.hp < 60:
            acts.append(HEAL)
        ms, tau0 = self._move_samples(env)
        dist = np.linalg.norm(env.boss.root - env.p_xy); reach = 0.8 + 0.25 * env.spec.scale
        best, bv = NONE, -1e9
        for a in acts:
            pos, inv = self_model(env, a, self.H)
            if ms is None:
                risk = 0.0
            else:
                rel = np.array([self.maps.rel(env, q) for q in pos])
                taus = tau0 + np.arange(1, self.H + 1)
                rs = []
                for m in ms:
                    h = self.maps.hazard(m, taus, rel) * (~inv)
                    rs.append(1 - np.prod(1 - h))
                risk = float(np.mean(rs))
            btn = a % N_BUTTONS
            gain = 0.0
            if btn == LIGHT and dist <= reach:
                gain = self.dmg
            elif btn == HEAL:
                gain = 0.6 * min(60, 100 - env.hp)
            # shaping: prefer ending near striking range when it is safe
            dend = np.linalg.norm(env.boss.root - pos[-1])
            v = gain - self.hc * risk - 2.0 * max(0.0, dend - reach)
            if btn in (LIGHT, HEAL):
                v -= self.hc * risk * 0.5        # commitment: cannot roll out
            if v > bv:
                best, bv = a, v
        if best % N_BUTTONS == NONE and best != NONE:
            self.hold, self.hold_n = best, 5
        return best
