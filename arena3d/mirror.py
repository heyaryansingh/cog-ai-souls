"""B65 Mirror simulation: understand the boss by re-enacting what you have seen it do.

Motor-simulation theories of action understanding (mirror systems, "simulation theory") say we predict others'
actions by running them on our own body model. Here the agent keeps an episodic memory of every boss move it has
watched: the boss's perceived full-body pose over the move (noisy pose estimate), expressed in the move's onset
frame (origin = boss position at onset, x-axis = boss->player line at onset), plus when shockwaves/projectiles
appeared. To imagine a future it re-enacts a remembered episode of the hypothesised move on its own copy of the
arena (the same body/contact physics it lives in: capsules, blade, i-frames, rings), aligned to the current boss
position, facing and time since onset, and checks real 3D contact against its own imagined body. No boss program,
mocap library or boss simulator is used: identity comes from noisy reading (70% at 12 frames) and learned strings,
motion from memory.
"""
from __future__ import annotations

import os
import types

import numpy as np

from .env import N_BUTTONS, NONE, ROLL, LIGHT, HEAL
from .mocap import _qmul
from .boss import _qz


def _yaw_of(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class MirrorMemory:
    def __init__(self, n_moves, seed=0, read_k=12, acc=0.7, keep=10, noise_j=0.03, noise_p=0.03):
        self.n, self.read_k, self.acc, self.keep = n_moves, read_k, acc, keep
        self.nj, self.np_ = noise_j, noise_p
        self.rng = np.random.default_rng(seed)
        self.eps = {}            # label -> list of episodes
        self.ring_R = {}         # label -> observed final ring radii
        self.update = True

    def reset(self, env):
        self.serial, self.cur, self.label, self.onset, self.origin, self.fwd = -1, None, None, 0, None, 0.0
        self._rings = {}

    def _close(self):
        e = self.cur
        if e is not None and self.update and self.label is not None and len(e["q"]) >= 8:
            L = self.eps.setdefault(self.label, []); L.append(e); del L[:-self.keep]
        self.cur = None

    def to_frame(self, q, origin, fwd):
        """boss qpos (63) -> onset frame"""
        out = q.copy(); d = q[:2] - origin; c, s = np.cos(-fwd), np.sin(-fwd)
        out[:2] = [c * d[0] - s * d[1], s * d[0] + c * d[1]]
        out[3:7] = _qmul(_qz(-fwd), q[3:7])
        return out

    @staticmethod
    def from_frame(qf, origin, fwd):
        out = qf.copy(); c, s = np.cos(fwd), np.sin(fwd)
        out[:2] = origin + np.array([c * qf[0] - s * qf[1], s * qf[0] + c * qf[1]])
        q = _qmul(_qz(fwd), qf[3:7]); out[3:7] = q / np.linalg.norm(q)
        return out

    def observe(self, env):
        b = env.boss; s = getattr(env, "boss_serial", -1)
        # rings: follow each to learn how far it gets (attributed to the move it spawned in)
        cur = {}
        for rg in env.rings:
            k = (round(float(rg["c"][0]), 2), round(float(rg["c"][1]), 2))
            if k not in self._rings:
                self._rings[k] = {"lab": None, "r": 0.0, "ep": self.cur}
                if self.cur is not None: self.cur["rings"].append(len(self.cur["q"]))
            self._rings[k]["r"] = max(self._rings[k]["r"], rg["r"] + self.rng.normal(0, 0.05)); cur[k] = 1
        for k in list(self._rings):
            if k not in cur:
                e = self._rings.pop(k)
                lab = e["ep"]["label"] if e["ep"] is not None else None
                if lab is not None and self.update:
                    self.ring_R.setdefault(lab, []).append(e["r"]); self.ring_R[lab] = self.ring_R[lab][-12:]
        if self.cur is not None and len(env.projectiles) > self.cur.get("_np", 0):
            self.cur["proj"].append(len(self.cur["q"]))
        if self.cur is not None: self.cur["_np"] = len(env.projectiles)
        if b.kind == "strike" and s != self.serial:
            self._close()
            self.serial, self.onset, self.label = s, env.t, None
            to_p = env.p_xy - b.root
            self.origin = b.root + self.rng.normal(0, 0.1, 2)
            self.fwd = float(np.arctan2(to_p[1], to_p[0])) + self.rng.normal(0, 0.05)
            self.cur = {"q": [], "d0": float(np.linalg.norm(to_p)), "rings": [], "proj": [], "label": None, "_np": len(env.projectiles)}
        if self.cur is None:
            return
        if b.kind != "strike" and len(self.cur["q"]) > 0:
            # keep recording the recovery for a while (it's part of the move's reach in time), then close
            if env.t - self.onset > 90:
                self._close(); return
        tau = env.t - self.onset
        if tau >= self.read_k and self.label is None:
            self.label = b.move_id if self.rng.random() < self.acc else int(self.rng.integers(self.n))
            self.cur["label"] = self.label
        q = env.d.qpos[:63].copy()
        q[:3] += self.rng.normal(0, self.np_, 3); q[7:] += self.rng.normal(0, self.nj, 56)
        self.cur["q"].append(self.to_frame(q, self.origin, self.fwd))

    def sample(self, label, d_now, rng):
        L = self.eps.get(label)
        if not L: return None
        w = np.array([np.exp(-0.5 * ((e["d0"] - d_now) / 1.0) ** 2) for e in L]) + 1e-3
        return L[int(rng.choice(len(L), p=w / w.sum()))]


class ReplayBoss:
    """Stand-in for the boss in an imagined arena: re-enacts a remembered episode from frame f0."""

    def __init__(self, real, ep, f0, origin, fwd, mem, label):
        self.ep, self.f, self.origin, self.fwd = ep, int(f0), origin, fwd
        self.kind = "strike"; self.unit = ep; self.player_healing = False; self.move_id = label
        self.root = real.root.copy(); self.yaw_off = 0.0; self.hold = 0
        R = np.median(mem.ring_R[label]) if mem.ring_R.get(label) else 0.0
        self.peak = ep["rings"][0] if (ep["rings"] and R > 0) else (ep["proj"][0] if ep["proj"] else 10 ** 6)
        self.cur_move = types.SimpleNamespace(ring=float(R) if ep["rings"] else 0.0, projectile=bool(ep["proj"]))
        self.idle_q = None

    def step(self, player_xy, phase=False):
        Q = self.ep["q"]; i = min(self.f, len(Q) - 1)
        q = MirrorMemory.from_frame(Q[i], self.origin, self.fwd)
        if self.f >= len(Q): self.kind = "idle"
        self.f += 1; self.root = q[:2].copy()
        return q


class MirrorPlanner:
    """MPC over the agent's macro-actions; futures imagined by re-enacting remembered boss episodes."""

    def __init__(self, mem, grammar, seed=0, H=24, n=3):
        self.mem, self.g, self.H, self.n = mem, grammar, H, n
        self.rng = np.random.default_rng(seed)

    def reset(self, env):
        self.mem.reset(env); self.g.reset(env); self.hold, self.hold_n = None, 0
        self._bhp, self._dmg_t = env.boss_hp, env.t

    def _imagine(self, env, a, seed):
        rng = np.random.default_rng(seed); mem = self.mem; b = env.boss
        sim = env.clone(seed)
        tau = env.t - mem.onset
        if b.kind == "strike" and mem.cur is not None and tau < 90:
            lab = mem.label if mem.label is not None else self.g.sample(env, rng)
            ep = mem.sample(lab, mem.cur["d0"], rng)
            if ep is not None:
                sim.boss = ReplayBoss(b, ep, tau + 1, mem.origin, mem.fwd, mem, lab)
        elif os.environ.get("MIRROR_OPENER", "1") == "1":
            # boss between moves: it may open within 0-15 frames; frame = now
            lab = self.g.sample(env, rng, next_move=True)
            to_p = env.p_xy - b.root; d = float(np.linalg.norm(to_p))
            ep = mem.sample(lab, d, rng)
            if ep is not None:
                start = -int(rng.integers(0, 16))
                rb = ReplayBoss(b, ep, 0, b.root.copy(), float(np.arctan2(to_p[1], to_p[0])), mem, lab)
                if start < 0:     # hold the current pose for the delay, then re-enact
                    rb.f = start; q_now = env.d.qpos[:63].copy(); step0 = rb.step
                    def step(px, ph=False, rb=rb, q_now=q_now, step0=step0):
                        if rb.f < 0:
                            rb.f += 1; return q_now
                        return step0(px, ph)
                    rb.step = step
                sim.boss = rb
        hp0, b0 = sim.hp, sim.boss_hp
        for k in range(self.H):
            mv = a % N_BUTTONS == NONE
            _, _, done, info = sim.step(a if (k == 0 or (mv and k < 6)) else 0)
            if done: break
        v = (b0 - sim.boss_hp) - 2.5 * (hp0 - sim.hp) - (100 if sim.hp <= 0 else 0)
        reach = 0.8 + 0.25 * env.spec.scale
        v -= self._w * max(0.0, float(np.linalg.norm(b.root - sim.p_xy)) - reach)
        return v

    def act(self, env):
        self.g.observe(env)
        if not env.player_free():
            return 0
        if self.hold_n > 0:
            self.hold_n -= 1; return self.hold
        if env.boss_hp != self._bhp: self._bhp, self._dmg_t = env.boss_hp, env.t
        self._w = float(os.environ.get("MIRROR_CLOSE", 2.0)) * (1 + float(os.environ.get("MIRROR_HUNGER", 1.0)) * (env.t - self._dmg_t) / 300.0)
        acts = [NONE, 1 * N_BUTTONS + NONE, 5 * N_BUTTONS + NONE, 3 * N_BUTTONS + NONE, 7 * N_BUTTONS + NONE,
                LIGHT, 3 * N_BUTTONS + ROLL, 7 * N_BUTTONS + ROLL, 5 * N_BUTTONS + ROLL]
        if os.environ.get("WM_ROLL_FWD") == "1":
            acts += [1 * N_BUTTONS + ROLL, 2 * N_BUTTONS + ROLL, 8 * N_BUTTONS + ROLL]
        if env.flasks and env.hp < 60: acts.append(HEAL)
        seeds = [int(s) for s in self.rng.integers(1 << 30, size=self.n)]
        vals = [np.mean([self._imagine(env, a, s) for s in seeds]) for a in acts]
        a = acts[int(np.argmax(vals))]
        if a % N_BUTTONS == NONE and a != NONE: self.hold, self.hold_n = a, 5
        return a
