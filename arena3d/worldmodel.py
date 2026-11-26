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

import os
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
        # B33 aim model: the boss keeps turning toward you until its swing commits. Frame = onset origin, rotated
        # by the boss's perceived turn since onset; the planner imagines that turn toward its imagined position.
        self.track = os.environ.get('WM_TRACK') == '1'
        if self.track:
            self.cur = False
        self.turn, self.dturn = 0.0, []          # perceived turn since onset; per-frame turn magnitudes seen
        self.peak_obs = {}                       # move -> taus at which turning was last seen (commit time)
        # B43 motion-phase clock: index danger by how far the swing has progressed (pose progress), not by time
        # since onset, so a delayed windup doesn't smear timing; learn each move's hold (delay) lengths
        self.phase_idx = os.environ.get('WM_PHASEIDX') == '1'
        self.hold_obs, self.hold_ph = {}, {}
        self.ph, self.hold_el, self.cur_hold, self.cur_hold_ph, self.holding = 0, 0, 0, None, False
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
            if self.track and getattr(self, 'prev_label', None) is not None and getattr(self, 'last_turn_tau', 0) > 0:
                self.peak_obs.setdefault(self.prev_label, []).append(self.last_turn_tau)
            to_p = env.p_xy - b.root
            self.origin = b.root + self.rng.normal(0, 0.1, 2)
            self.fwd = float(np.arctan2(to_p[1], to_p[0])) + self.rng.normal(0, 0.1)   # bosses face you at onset
            self.yaw0, self.turn, self.last_turn_tau = float(b.yaw_off), 0.0, 0
            if self.phase_idx and getattr(self, 'prev_label', None) is not None:
                self.hold_obs.setdefault(self.prev_label, []).append(self.cur_hold)
                if self.cur_hold_ph is not None:
                    self.hold_ph.setdefault(self.prev_label, []).append(self.cur_hold_ph)
            self.ph, self.hold_el, self.cur_hold, self.cur_hold_ph, self.holding = 0, 0, 0, None, False
        import os
        persist = os.environ.get("WM_PERSIST", "0") == "1"   # B30: keep attending to a move's effects after the swing
        if self.origin is None or s != self.serial or (b.kind != "strike" and not persist):
            return
        tau = env.t - self.onset
        if b.kind != "strike" and tau >= 3 * TB:
            return
        if tau >= self.read_k and self.label is None:
            self.label = b.move_id if self.rng.random() < self.acc else int(self.rng.integers(self.n))
            self.prev_label = self.label
        if self.phase_idx:
            if b.kind == "strike":
                self.holding = bool(b.hold > 0 and int(b.f) >= b.hold_at)     # seen as the body freezing mid-windup
                if self.holding:
                    if self.cur_hold == 0: self.cur_hold_ph = self.ph
                    self.cur_hold += 1; self.hold_el = self.cur_hold
                else:
                    self.ph = int(round(b.f / max(b.speed, 1e-3)))
            else:
                self.holding = False; self.ph += 1
            tau_m = self.ph
        else:
            tau_m = tau
        fwd_now = None
        if self.track:
            if b.kind == "strike":
                new = float(b.yaw_off) - self.yaw0 + self.rng.normal(0, 0.003)   # seen body rotation since onset
                d = abs(new - self.turn)
                if d > 0.004:
                    self.dturn.append(d); self.dturn = self.dturn[-2000:]; self.last_turn_tau = tau_m
                self.turn = new
            fwd_now = self.fwd + self.turn
        hit = bool(info.get("hit") is not None or info.get("blocked") or info.get("dodged"))
        bdisp = self.rel(env, b.root + self.rng.normal(0, 0.1, 2))      # boss travel since onset (perceived)
        if getattr(self, 'geom', False) and b.kind == "strike":        # seen weapon tip (noisy keypoint), relative to the boss now
            sd = 0.06 * 1.75 * env.spec.scale
            tip = env.d.site_xpos[env.tip][:2] + self.rng.normal(0, sd, 2)
            hand = env.d.xpos[env.m.body('b_rhand').id][:2] + self.rng.normal(0, sd, 2)
            if self.track:
                self.cur_traj.append((tau_m, np.r_[self.rel(env, tip, fwd=fwd_now), self.rel(env, hand, fwd=fwd_now)]))
            else:
                self.cur_traj.append((tau, np.r_[self.rel(env, tip) - bdisp, self.rel(env, hand) - bdisp]))
        r = self.rel(env) - bdisp if self.cur else self.rel(env, fwd=fwd_now)       # my position relative to the boss now
        self.pend.append((tau_m, r, hit, bdisp))
        if self.label is not None and self.update:
            for (t0, r0, h0, d0) in self.pend:
                ti, ri, ai, ok = _bins(t0, r0)
                if ok:
                    self.N[self.label, ti, ri, ai] += 1; self.H[self.label, ti, ri, ai] += h0
                if t0 < 3 * TB:
                    self.D[self.label, t0] += d0; self.Dn[self.label, t0] += 1
            self.pend = []

    def future_phases(self, m, H, rng):
        """Imagined motion-phase clock for the next H frames of move m: progress pauses for a sampled remaining
        windup hold (learned hold lengths of this move; conditioned on the hold already seen)."""
        L = self.hold_obs.get(m, [])
        hp = float(np.median(self.hold_ph[m])) if self.hold_ph.get(m) else 1e9
        if self.holding:
            longer = [x for x in L if x > self.hold_el]
            rem = (int(rng.choice(longer)) - self.hold_el) if longer else 3
            hp = self.ph
        elif L and self.ph < hp:
            rem = int(rng.choice(L))
        else:
            rem = 0
        out = np.zeros(H, int); p = self.ph
        for i in range(H):
            if rem > 0 and p >= hp:
                rem -= 1
            else:
                p += 1
            out[i] = p
        return out

    def track_rate(self):
        return float(np.percentile(self.dturn, 90)) if len(self.dturn) > 20 else 0.03

    def commit_tau(self, m):
        v = self.peak_obs.get(m)
        return float(np.median(v)) if v else 30.0

    def rel_tracked(self, env, pos, taus, tau_now, m):
        """Imagined positions relative to the boss as it keeps turning toward each of them until it commits."""
        c = self.commit_tau(m); w = self.track_rate()
        out = np.zeros((len(pos), 2)); turn = self.turn
        for i, (q, t) in enumerate(zip(pos, taus)):
            if t <= c:
                to_q = q - self.origin; err = (np.arctan2(to_q[1], to_q[0]) - (self.fwd + turn) + np.pi) % (2 * np.pi) - np.pi
                turn += float(np.clip(err, -w, w))
            out[i] = self.rel(env, q, fwd=self.fwd + turn)
        return out

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

    def geom_hazard(self, m, tau, rel, reach=None):
        """Fraction of remembered tip trajectories of move m passing within `reach` of rel at tau (+-1)."""
        T = self.trajs.get(m)
        if not T:
            return np.zeros(len(tau))
        key = (m, len(T), id(T[-1]))
        if getattr(self, "_gk", {}).get(m) != key:
            A = np.full((len(T), 3 * TB + 2, len(T[0][0][1]) if T[0] else 2), np.nan)
            for j, tr in enumerate(T):
                for t, p in tr:
                    if 0 <= t < 3 * TB:
                        A[j, t + 1] = p
            self._ga = getattr(self, "_ga", {}); self._ga[m] = A
            self._gk = getattr(self, "_gk", {}); self._gk[m] = key
        A = self._ga[m]
        ti = np.clip(np.asarray(tau), 0, 3 * TB - 1) + 1
        best = np.full((A.shape[0], len(ti)), np.inf)
        seg_mode = os.environ.get("WM_SEG", "0") == "1" and A.shape[2] == 4   # B25: blade segment hurt (12.6%)
        if reach is None:
            reach = 0.4 if seg_mode else 0.55                 # v6 tip-only (42.9%) is the default
        for o in (-1, 0, 1):
            if not seg_mode:
                best = np.fmin(best, np.hypot(*(A[:, ti + o, :2] - rel[None]).transpose(2, 0, 1)))
                continue
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
        import os
        k = float(os.environ.get("WM_HIER", 0))
        if k > 0:   # hierarchical back-off (B30): sparse places borrow this move's hazard-at-this-time over all places
            Hm, Nm = Hs[m].sum((1, 2)), Ns[m].sum((1, 2))
            pm = (Hm[ti] + self.a) / (Nm[ti] + self.b)
            p = (Hs[m, ti, ri, ai] + k * pm) / (Ns[m, ti, ri, ai] + k)
        else:
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
        persist = __import__("os").environ.get("WM_PERSIST", "0") == "1"
        if (b.kind != "strike" and not persist) or mp.origin is None or env.t - mp.onset > 3 * TB:
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
        # engagement drive (B28/B30): the approach weight grows with time since this agent last landed a hit
        import os
        if getattr(self, "_bhp", None) != env.boss_hp or getattr(self, "_env", None) is not env:
            self._bhp, self._dmg_t, self._env = env.boss_hp, env.t, env
        w_close = float(os.environ.get("WM_CLOSE", 2.0)) * (1 + float(os.environ.get("WM_HUNGER", 0)) * (env.t - self._dmg_t) / 300.0)
        best, bv = NONE, -1e9
        for a in acts:
            pos, inv = self_model(env, a, self.H)
            if ms is None:
                risk = 0.0
            else:
                rel = np.array([self.maps.rel(env, q, *self.ref) for q in pos])
                taus = tau0 + np.arange(1, self.H + 1)
                rs = []
                taus_all = taus
                for m in ms:
                    taus = taus_all
                    if self.maps.phase_idx and self.ref[0] is None:
                        taus = self.maps.future_phases(m, self.H, self.rng)
                    rr = rel
                    if self.maps.track and self.ref[0] is None:
                        rr = self.maps.rel_tracked(env, pos, taus, tau0, m)
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
            v = gain - self.hc * risk - w_close * max(0.0, dend - reach)
            if btn in (LIGHT, HEAL):
                v -= self.hc * risk * 0.5        # commitment: cannot roll out
            if v > bv:
                best, bv = a, v
        if best % N_BUTTONS == NONE and best != NONE:
            self.hold, self.hold_n = best, 5
        return best
