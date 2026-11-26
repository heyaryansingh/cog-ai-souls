"""Boss = a body (scaled CMU humanoid + weapon) and a behaviour over motion units from the open library.

A boss is sampled, not authored:
  * repertoire: n strike units + locomotion units drawn from the allowed mocap subjects;
  * combo graph: P(next strike | last) from motion-matching cost (pose/velocity distance) and a random taste;
  * timing: per-move speed and a log-normal *windup hold* just before the weapon's fastest frame (delays);
  * targeting: yaw tracking toward the player until the strike peaks, plus a lunge so the peak lands in reach.
Whether a move hurts is decided by 3D contact in the env, never by a label.
"""
from __future__ import annotations

from dataclasses import dataclass

import os

import numpy as np

from .mocap import _qmul

WEAPON_OF = ["sword", "greatsword", "club"]


@dataclass
class BossSpec:
    name: str
    scale: float
    weapon: str
    strikes: list            # unit indices
    walks: list              # unit indices (approach / reposition)
    combo: np.ndarray        # (n+1, n) row 0 = opener distribution, row i+1 = after strike i; last col... see p_end
    p_end: np.ndarray        # (n,) chance to stop the combo after strike i
    speed: np.ndarray        # (n,) playback speed per strike
    hold_med: np.ndarray     # (n,) windup hold median (frames, 0 = no hold)
    hold_sig: float
    track: float             # rad/frame yaw tracking before the peak
    aggression: float        # chance to attack (vs reposition) when in range
    hp: float = 800.0
    phase2: bool = True
    idles: list = None       # unit indices: slow motions played as recovery / stalking (punish windows)
    p_recover: float = 0.7


def danger_frame(u, scale, v_min=4.0, z_lo=0.25, z_hi=1.8):
    """Fastest frame where the (scaled) tip is at a height that can hit a standing human; -1 if none."""
    z = u.tip_rel[:, 2] * scale
    v = u.tip_speed * scale
    ok = (v > v_min) & (z > z_lo) & (z < z_hi)
    ok[:3] = False
    return int(np.argmax(np.where(ok, v, -1))) if ok.any() else -1


def sample_boss(rng, units, allowed, name="boss", n_strikes=None, clips=None, verify=None):
    scale = float(np.exp(rng.uniform(np.log(1.2), np.log(2.6))))
    pool_s = [i for i in allowed if units[i].kind == "strike" and units[i].length < 75
              and danger_frame(units[i], scale) >= int(os.environ.get("MIN_WINDUP", 12))]
    pool_w = [i for i in allowed if units[i].kind == "move" and units[i].travel / (units[i].length * 0.03) > 0.6]
    n = int(n_strikes or rng.integers(6, 14))

    verify = (os.environ.get("ARENA3D_VERIFY", "1") == "1") if verify is None else verify
    if verify and clips is not None:
        cand = [int(i) for i in rng.permutation(pool_s)]
        strikes = None   # filled after the spec exists (needs scale/weapon)
    else:
        strikes = [int(i) for i in rng.choice(pool_s, n, replace=False)]
    walks = [int(i) for i in rng.choice(pool_w, min(6, len(pool_w)), replace=False)]
    pool_i = [i for i in allowed if units[i].kind == "move" and units[i].peak_speed < 3.0
              and units[i].travel / (units[i].length * 0.03) < 0.3 and units[i].length >= 15]
    idles = [int(i) for i in rng.choice(pool_i, min(6, len(pool_i)), replace=False)]
    weapon = str(rng.choice(WEAPON_OF))
    if strikes is None:
        from .aim import verify_strikes
        proto = BossSpec(name=name, scale=scale, weapon=weapon, strikes=[], walks=walks, combo=None, p_end=None,
                         speed=None, hold_med=None, hold_sig=0.45, track=0.04, aggression=1.0, idles=idles)
        strikes = verify_strikes(proto, units, clips, cand, n, rng)
        if len(strikes) < 3:
            strikes += [c for c in cand if c not in strikes][:3 - len(strikes)]
        n = len(strikes)
    # motion-matching cost end(i) -> start(j)
    S0 = np.stack([units[i].pose0 for i in strikes])
    S1 = np.stack([units[i].pose1 for i in strikes])
    D = np.linalg.norm(S1[:, None] - S0[None], axis=2)
    D = D / (np.median(D) + 1e-6)
    taste = rng.normal(0, 2.5, (n, n))            # peaked rows: each strike has a favourite follow-up (learnable chains)
    A = np.exp(-2.0 * D + taste)
    np.fill_diagonal(A, A.diagonal() * 0.3)
    A /= A.sum(1, keepdims=True)
    opener = rng.dirichlet(np.ones(n) * 0.7)
    return BossSpec(name=name, scale=scale, weapon=weapon, strikes=strikes, walks=walks,
                    combo=np.vstack([opener, A]), p_end=rng.uniform(0.25, 0.8, n),
                    speed=rng.uniform(0.7, 1.15, n) / np.sqrt(scale / 1.6),
                    hold_med=np.where(rng.random(n) < 0.45, rng.uniform(6, 30, n), 0.0), hold_sig=0.45,
                    track=float(rng.uniform(0.01, 0.06)), aggression=float(rng.uniform(0.35, 0.8)),
                    hp=float(350 + 150 * scale), idles=idles, p_recover=float(rng.uniform(0.85, 1.0)))


def _qz(a):
    return np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)])


def heading(q):
    """Facing direction of a CMU humanoid root: the root frame's x-axis heading + 90 deg (checked against
    the hip line: forward = up x (left hip - right hip))."""
    w, x, y, z = q
    return np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z)) + np.pi / 2


class BossRunner:
    """Plays units back-to-back, with blending, holds, tracking and lunges. Produces a qpos (63) per frame."""

    def __init__(self, spec: BossSpec, units, clips, rng):
        self.s, self.U, self.C, self.rng = spec, units, clips, rng
        self.reset(np.array([0.0, 4.0]), -np.pi / 2)

    def reset(self, pos, yaw):
        self.root = np.array([pos[0], pos[1]])
        self.yaw = yaw
        self.unit = None
        self.last_q = None
        self.blend = 0
        self.combo_last = -1      # index into strikes of the last strike in the current combo, -1 none
        self.move_id = -1         # current strike index or -1
        self.stagger = 0

    # --- choosing ----------------------------------------------------------------------------------
    def _choose(self, player_xy, phase):
        s, rng = self.s, self.rng
        dist = np.linalg.norm(player_xy - self.root)
        reach = 1.2 * s.scale + 0.5
        if self.combo_last >= 0 and rng.random() > s.p_end[self.combo_last]:
            j = int(rng.choice(len(s.strikes), p=s.combo[self.combo_last + 1]))
            return "strike", j
        if self.combo_last >= 0 and s.idles and rng.random() < s.p_recover:
            return "idle", int(rng.choice(len(s.idles)))      # recovery after a combo: the punish window
        if dist < reach + 1.5 and rng.random() < s.aggression * (1.25 if phase else 1.0):
            return "strike", int(rng.choice(len(s.strikes), p=s.combo[0]))
        return "walk", int(rng.choice(len(s.walks)))

    def _start(self, player_xy, phase):
        kind, j = self._choose(player_xy, phase)
        s, U = self.s, self.U
        u = U[s.strikes[j]] if kind == "strike" else (U[s.idles[j]] if kind == "idle" else U[s.walks[j]])
        self.unit, self.kind, self.move_id = u, kind, (j if kind == "strike" else -1)
        self.combo_last = j if kind == "strike" else -1
        self.f = 0.0
        self.speed = (s.speed[j] if kind == "strike" else 1.0) * (1.2 if phase else 1.0)
        c = self.C[u.clip]
        self.c = c
        self.q0yaw = heading(c.quat[u.start])
        self.p0 = c.pos[u.start].copy()
        self.origin = self.root.copy()
        to_p = player_xy - self.root
        want = np.arctan2(to_p[1], to_p[0])
        # face the player at the start (unit's own heading is undone, then the boss's facing applied)
        self.want0 = want
        self.peak = u.peak
        if kind == "strike":
            k = danger_frame(u, s.scale)
            self.peak = k
            p = u.tip_rel[k, :2] * s.scale
            # attack warping: rotate so the tip's position at the danger frame points at the player
            self.yaw_off = want - self.q0yaw - np.arctan2(p[1], p[0])
        elif kind == "idle":
            self.yaw_off = want - self.q0yaw
        else:   # aim the unit's travel (not its facing) at the player
            dp = c.pos[u.end - 1, :2] - c.pos[u.start, :2]
            self.yaw_off = want - np.arctan2(dp[1], dp[0]) + rng_ang(self.rng, 0.25)
        self.hold = 0
        self.hold_at = -1
        if kind == "strike" and s.hold_med[j] > 0:
            self.hold = int(np.clip(round(self.rng.lognormal(np.log(s.hold_med[j]), s.hold_sig)), 1, 90))
            self.hold_at = max(0, self.peak - 5)
        # lunge so the peak lands at the player: extra translation spread over pre-peak frames
        dist = np.linalg.norm(to_p)
        if kind == "strike":
            # translation so that the tip at the danger frame lands ~0.15 m inside the player's torso
            # (tip_rel already includes the unit's own root travel up to that frame)
            need = dist - np.linalg.norm(p) + 0.15
            self.lunge = float(np.clip(need, -1.5, 2.5 + s.scale))
        else:
            self.lunge = 0.0
        self.lunge_dir = to_p / (dist + 1e-6)
        self.lunge_done = 0.0
        self.blend = 8

    # --- stepping ----------------------------------------------------------------------------------
    def step(self, player_xy, phase=False):
        """Advance one frame. Returns qpos (63)."""
        if self.stagger > 0:
            self.stagger -= 1
            return self.last_q
        if self.unit is None or self.f >= self.unit.length - 1:
            self._start(player_xy, phase)
        u = self.unit
        if self.hold > 0 and int(self.f) >= self.hold_at:
            self.hold -= 1                      # windup freeze (slowly drift for life)
            q = self._pose(self.f, player_xy, track=True)
            return self._finish(q)
        pre_peak = self.kind == "strike" and self.f < self.peak
        self.f = min(self.f + self.speed, u.length - 1)
        if pre_peak and self.peak > 0:
            step = self.lunge * self.speed / max(self.peak, 1)
            self.lunge_done += step
        q = self._pose(self.f, player_xy, track=pre_peak)
        return self._finish(q)

    def _pose(self, f, player_xy, track):
        u, c, s = self.unit, self.c, self.s
        if track:
            to_p = player_xy - self.origin
            want = np.arctan2(to_p[1], to_p[0])
            err = (want - self.want0 + np.pi) % (2 * np.pi) - np.pi
            dy = float(np.clip(err, -s.track, s.track))
            self.yaw_off += dy
            self.want0 += dy
        t0 = u.start + int(f)
        a = f - int(f)
        t1 = min(t0 + 1, u.end - 1)
        d = (c.pos[t0] * (1 - a) + c.pos[t1] * a) - self.p0
        R = self._Rz(self.yaw_off)
        xy = self.origin + (R @ (d * s.scale))[:2] + self.lunge_dir * self.lunge_done
        z = (c.pos[t0, 2] * (1 - a) + c.pos[t1, 2] * a) * s.scale
        quat = self._root_quat(f)
        j = c.joints[t0] * (1 - a) + c.joints[t1] * a
        self.root = xy
        return np.concatenate([[xy[0], xy[1], z], quat, j])

    def _root_quat(self, f):
        t0 = self.unit.start + int(f)
        q = _qmul(_qz(self.yaw_off), self.c.quat[t0])
        return q / np.linalg.norm(q)

    @staticmethod
    def _Rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _finish(self, q):
        if self.last_q is not None and self.blend > 0:
            w = self.blend / 9.0
            q = q.copy()
            q[7:] = w * self.last_q[7:] + (1 - w) * q[7:]
            q0 = self.last_q[3:7] if np.dot(self.last_q[3:7], q[3:7]) >= 0 else -self.last_q[3:7]
            qq = w * q0 + (1 - w) * q[3:7]
            q[3:7] = qq / np.linalg.norm(qq)
            q[2] = w * self.last_q[2] + (1 - w) * q[2]
            self.blend -= 1
        self.last_q = q
        return q


def rng_ang(rng, s):
    return float(rng.normal(0, s))
