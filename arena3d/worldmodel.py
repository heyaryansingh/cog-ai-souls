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

TB, RB, AB = 30, 16, 8           # 3-frame tau bins (90 frames), 0.75 m rings (12 m), 8 bearing sectors
R_MAX = 12.0


def _bins(tau, rel):
    r = np.hypot(rel[..., 0], rel[..., 1])
    a = np.arctan2(rel[..., 1], rel[..., 0])
    ti = np.clip(np.asarray(tau) // 3, 0, TB - 1).astype(int)
    ri = np.clip((r / 0.75).astype(int), 0, RB - 1)
    ai = ((a + np.pi) / (2 * np.pi) * AB).astype(int) % AB
    return ti, ri, ai, r < R_MAX


class DangerMaps:
    def __init__(self, n_moves, prior_hit=0.005, prior_n=2.0, seed=0, read_k=12, acc=0.7):
        self.N = np.zeros((n_moves, TB, RB, AB)); self.H = np.zeros((n_moves, TB, RB, AB))
        self.D = np.zeros((n_moves, 3 * TB, 2)); self.Dn = np.zeros((n_moves, 3 * TB))   # boss travel (onset frame)
        import os
        self.cur = os.environ.get('WM_FRAME', 'onset') == 'current'
        self.geom = os.environ.get('WM_GEOM') == '1'; self.trajs = {}; self.cur_traj = []; self.prev_label = None
        self.a, self.b = prior_hit * prior_n, prior_n
        self.rng = np.random.default_rng(seed); self.read_k, self.acc, self.n = read_k, acc, n_moves
        self.update = True

    # --- learning from felt contacts -----------------------------------------------------------------
    def reset(self, env):
        import os
        # body-size units: positions in units of the boss's (perceived) size, Margit-like scale 1.9 = 1, so hit
        # volumes transfer across bosses of different size
        self.unit = env.spec.scale / 1.9 if os.environ.get('WM_NORM') == '1' else 1.0
        self.serial, self.origin, self.fwd, self.label, self.onset = -1, None, 0.0, None, 0
        self.pend = []                      # (tau, rel) for the current move before it is identified

    def rel(self, env, p=None, origin=None, fwd=None):
        p = env.p_xy if p is None else p
        origin = self.origin if origin is None else origin; fwd = self.fwd if fwd is None else fwd
        d = p - origin; c, s = np.cos(-fwd), np.sin(-fwd)
        return (np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])) / getattr(self, 'unit', 1.0)

    def observe(self, env, info):
        b = env.boss
        s = getattr(env, "boss_serial", -1)
        if b.kind == "strike" and s != self.serial:
            self.serial, self.onset, self.label, self.pend = s, env.t, None, []
            if getattr(self, 'geom', False) and getattr(self, 'cur_traj', None) and getattr(self, 'prev_label', None) is not None:
                self.trajs.setdefault(self.prev_label, []).append(self.cur_traj[:])
                self.trajs[self.prev_label] = self.trajs[self.prev_label][-40:]
            self.cur_traj = []
            to_p = env.p_xy - b.root
            self.origin = b.root + self.rng.normal(0, 0.1, 2)
            self.fwd = float(np.arctan2(to_p[1], to_p[0])) + self.rng.normal(0, 0.1)   # bosses face you at onset
        if self.origin is None or b.kind != "strike" or s != self.serial:
            return
        tau = env.t - self.onset
        if tau >= self.read_k and self.label is None:
            self.label = b.move_id if self.rng.random() < self.acc else int(self.rng.integers(self.n))
            self.prev_label = self.label
        hit = bool(info.get("hit") is not None or info.get("blocked") or info.get("dodged"))
        bdisp = self.rel(env, b.root + self.rng.normal(0, 0.1, 2))      # boss travel since onset (perceived)
        if getattr(self, 'geom', False):        # seen weapon tip (noisy keypoint), relative to the boss now
            sd = 0.06 * 1.75 * env.spec.scale
            tip = env.d.site_xpos[env.tip][:2] + self.rng.normal(0, sd, 2)
            hand = env.d.xpos[env.m.body('b_rhand').id][:2] + self.rng.normal(0, sd, 2)
            self.cur_traj.append((tau, np.r_[self.rel(env, tip) - bdisp, self.rel(env, hand) - bdisp]))
        r = self.rel(env) - bdisp if self.cur else self.rel(env)       # my position relative to the boss now
        self.pend.append((tau, r, hit, bdisp))
        if self.label is not None and self.update:
            for (t0, r0, h0, d0) in self.pend:
                ti, ri, ai, ok = _bins(t0, r0)
                if ok:
                    self.N[self.label, ti, ri, ai] += 1; self.H[self.label, ti, ri, ai] += h0
                if t0 < 3 * TB:
                    self.D[self.label, t0] += d0; self.Dn[self.label, t0] += 1
            self.pend = []

    def _smoothed(self):
        """Kernel-smoothed counts (generalise across neighbouring times, distances and bearings), cached."""
        import os
        sig = float(os.environ.get("WM_SMOOTH", 1.0))
        key = (int(self.N.sum()), sig)
        if getattr(self, "_key", None) != key:
            if sig > 0:
                from scipy.ndimage import gaussian_filter
                f = lambda X: gaussian_filter(X, sigma=(0, sig, sig, sig), mode=("nearest", "nearest", "nearest", "wrap"))
                self._Ns, self._Hs = f(self.N), f(self.H)
            else:
                self._Ns, self._Hs = self.N, self.H
            self._key = key
        return self._Ns, self._Hs

    def geom_hazard(self, m, tau, rel, reach=0.4):
        """Fraction of remembered tip trajectories of move m passing within `reach` of rel at tau (+-1)."""
        T = self.trajs.get(m)
        if not T:
            return np.zeros(len(tau))
        key = (m, len(T), id(T[-1]))
        if getattr(self, "_gk", {}).get(m) != key:
            A = np.full((len(T), 3 * TB + 2, 4), np.nan)
            for j, tr in enumerate(T):
                for t, p in tr:
                    if 0 <= t < 3 * TB:
                        A[j, t + 1] = p
            self._ga = getattr(self, "_ga", {}); self._ga[m] = A
            self._gk = getattr(self, "_gk", {}); self._gk[m] = key
        A = self._ga[m]
        ti = np.clip(np.asarray(tau), 0, 3 * TB - 1) + 1
        best = np.full((A.shape[0], len(ti)), np.inf)
        for o in (-1, 0, 1):
            P = A[:, ti + o]                                   # (traj, tau, 4): tip xy, hand xy
            tip, hand = P[..., :2], P[..., 2:]
            seg = tip - hand; w = rel[None] - hand
            u = np.clip((seg * w).sum(-1) / np.maximum((seg * seg).sum(-1), 1e-6), 0, 1)
            d = np.hypot(*(w - u[..., None] * seg).transpose(2, 0, 1))   # distance to the blade segment
            best = np.fmin(best, d)
        return (best < reach).mean(0) * 0.5

    def hazard(self, m, tau, rel):
        ti, ri, ai, ok = _bins(tau, rel)
        Ns, Hs = self._smoothed()
        p = (Hs[m, ti, ri, ai] + self.a) / (Ns[m, ti, ri, ai] + self.b)
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

    def __init__(self, maps, grammar, H=30, hit_cost=None, dmg=40.0, seed=0, n=8):
        import os
        hit_cost = float(os.environ.get("WM_HITCOST", 200.0)) if hit_cost is None else hit_cost
        self.maps, self.g, self.H, self.hc, self.dmg, self.n = maps, grammar, H, hit_cost, dmg, n
        self.rng = np.random.default_rng(seed)
        self.hold, self.hold_n = None, 0

    def reset(self, env):
        self.maps.reset(env); self.g.reset(env); self.hold, self.hold_n = None, 0

    def _move_samples(self, env):
        """(move id, tau) samples for the current boss move; None if the boss isn't attacking."""
        b = env.boss; mp = self.maps
        if b.kind != "strike" or mp.origin is None or env.t - mp.onset > 3 * TB:
            # boss between moves: imagine the next move starting soon (opener anticipation), frame of
            # reference = now, start delay 0-15 frames, identity from the learned strings
            import os
            if os.environ.get("WM_OPENER", "1") == "1":
                to_p = env.p_xy - b.root
                self.ref = (b.root.copy(), float(np.arctan2(to_p[1], to_p[0])))
                return [self.g.sample(env, self.rng, next_move=True) for _ in range(self.n)], -int(self.rng.integers(0, 16))
            return None, 0
        self.ref = (None, None)
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
                rel = np.array([self.maps.rel(env, q, *self.ref) for q in pos])
                taus = tau0 + np.arange(1, self.H + 1)
                rs = []
                for m in ms:
                    rr = rel
                    if self.maps.cur:     # predicted boss travel for this move (learned mean)
                        ti = np.clip(taus, 0, 3 * TB - 1)
                        disp = self.maps.D[m, ti] / np.maximum(self.maps.Dn[m, ti], 1)[:, None]
                        rr = rel - disp
                    h = self.maps.hazard(m, np.maximum(taus, 0), rr)
                    if getattr(self.maps, 'geom', False):
                        h = np.maximum(h, self.maps.geom_hazard(m, np.maximum(taus, 0), rr))
                    h = h * (~inv) * (taus >= 0)
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
