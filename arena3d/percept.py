"""B77 honest perception for the mirror learner (MIRROR_FAIR=1).

Everything the mirror learner knows about the boss comes through `FairView`, an env-shaped proxy that shows
what a pose estimator would deliver instead of the simulator's ground truth:

  * pose: boss qpos with correlated drift (AR(1) per joint) + white noise, root position noise, bursty
    dropout (a dropped frame repeats the last good one), and a fixed sensor latency of SENSE_LAT frames;
  * blade tip: delayed, with 15 cm noise;
  * move onsets: detected from the perceived pose stream (a pose-discontinuity detector), never from the
    simulator's move counter; a "move" is active for 90 frames after the last detected onset;
  * move identities: discovered, not given. READ_K frames after a detected onset the move is recognised
    against episodic memory; if nothing remembered is close it gets a new label (Chinese-restaurant style);
  * the program's move list (count, phase gating) is hidden: labels index a fixed pool of NMAX slots.

The player's own position is known (proprioception); the view shows it delayed so remembered geometry stays
consistent with the delayed boss. Boss HP and phase are visible (HP bar), rings/projectiles are visible.
"""
from __future__ import annotations

import os
import types
from collections import deque

import numpy as np

NMAX = 40


class _BossView:
    def __init__(self, view):
        self._v = view
        self.kind, self.root, self.origin, self.unit = "idle", np.zeros(2), np.zeros(2), None
        self.cur_move, self.f, self.peak = None, 0, 10 ** 6

    @property
    def move_id(self):
        return self._v._label()


class FairView:
    def __init__(self, env, mem=None, seed=0):
        e = os.environ.get
        self.L = int(e("SENSE_LAT", 6))
        self.nj, self.drift, self.rho = float(e("FAIR_NJ", 0.05)), float(e("FAIR_DRIFT", 0.08)), 0.97
        self.npos = float(e("FAIR_NPOS", 0.08))
        self.p_on, self.p_off = 0.95, 0.88
        self.th, self.ref = float(e("FAIR_TH", 1.2)), 12
        self.c_new = float(e("FAIR_CNEW", 2.5))
        self.mem = mem
        self.rng = np.random.default_rng(seed)
        self.attach(env)

    # -- per fight ------------------------------------------------------------------------------------
    def attach(self, env):
        self.env = env
        self.buf = deque(maxlen=self.L + 1)
        self.hist = deque(maxlen=8)
        self.o = np.zeros(56); self.on = True; self.last = None
        self.boss = _BossView(self)
        self.boss_serial, self.boss_onset_t, self._onset = -1, -10 ** 6, -10 ** 6
        self._lab, self._lab_serial = None, None
        self.d = types.SimpleNamespace(qpos=env.d.qpos[:63].copy(), site_xpos=env.d.site_xpos.copy())
        self.tip = env.tip
        self.spec = types.SimpleNamespace(scale=env.spec.scale, hp=env.spec.hp,
                                          program=types.SimpleNamespace(moves=[types.SimpleNamespace(phase=1)] * NMAX))
        self.t = env.t - self.L
        self.p_xy = env.p_xy.copy()
        self.n_det = 0
        self.update()

    def __getattr__(self, k):            # anything not perceived explicitly (hp, rings, phase...) from the env
        if k.startswith("__") or "env" not in self.__dict__:
            raise AttributeError(k)
        return getattr(self.__dict__["env"], k)

    def _noisy(self):
        env = self.env
        self.o = self.rho * self.o + np.sqrt(1 - self.rho ** 2) * self.drift * self.rng.normal(size=56)
        q = env.d.qpos[:63].copy()
        q[7:] += self.o + self.rng.normal(0, self.nj, 56)
        q[:2] += self.rng.normal(0, self.npos, 2)
        sx = env.d.site_xpos.copy() + self.rng.normal(0, 0.15, env.d.site_xpos.shape)
        self.on = self.rng.random() < (self.p_on if self.on else 1 - self.p_off)
        if self.on or self.last is None:
            self.last = (q, sx)
        return self.last[0].copy(), self.last[1].copy(), self.on

    def update(self):
        """call once per real frame, after env.step"""
        env = self.env
        q, sx, ok = self._noisy()
        self.buf.append((q, sx, ok, env.p_xy.copy(), env.t))
        q, sx, ok, pxy, t = self.buf[0]
        self.d.qpos, self.d.site_xpos, self.p_xy, self.t = q, sx, pxy, t
        self.hist.append((q, ok))
        b = self.boss; b.root = q[:2].copy()
        # onset detector: pose discontinuity of the trunk and arms between the last 3 and the 3 before
        if len(self.hist) >= 7:
            H = list(self.hist)
            a = [h[0][21:63] for h in H[-3:] if h[1]]; c = [h[0][21:63] for h in H[-7:-4] if h[1]]
            if a and c and np.linalg.norm(np.mean(a, 0) - np.mean(c, 0)) > self.th and t - self._onset > self.ref:
                self._onset = t; self.boss_serial += 1; self.boss_onset_t = t; self.n_det += 1
                b.origin = b.root.copy()
        b.kind = "strike" if t - self._onset < 90 else "idle"

    # -- discovered move identity --------------------------------------------------------------------
    def _label(self):
        if self._lab_serial == self.boss_serial:
            return self._lab
        from .mirror import recognise
        mem = self.mem; lab = None
        if mem is not None and mem.cur is not None and len(mem.cur["q"]) >= 2 and mem.eps:
            tau = self.t - mem.onset
            rec = recognise(mem, tau, d_now=float(np.linalg.norm(self.p_xy - self.boss.root)))
            if rec and getattr(mem, "_c0", 1e9) < self.c_new:
                lab = rec[int(np.argmax([r[0] for r in rec]))][3]
        if lab is None:     # nothing remembered is close: a new move (the count lives in memory, so it persists)
            n = getattr(mem, "n_labels", 0); lab = min(n, NMAX - 1)
            if mem is not None: mem.n_labels = min(n + 1, NMAX)
        self._lab, self._lab_serial = lab, self.boss_serial
        return lab
