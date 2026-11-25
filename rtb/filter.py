"""Exact forward filter for one boss hypothesis (the agent's "fast hands").

Hidden state at tick t:
    W[p, k, d]  in the wind-up of move k, started in phase p, d ticks elapsed (d = 1..D_MAX)
    A[k, j]     in the active window of move k, tick j
    R[k, j]     in recovery of move k, tick j

Per environment step the filter does
    update(cue_t)              p(s_t | y_1..t)       (emission of the state that produced cue_t)
    predict(dist_t, phase_t, ctx_t)  p(s_t+1 | y_1..t)  (transition: next move chosen with this tick's bin/phase/player context)

and accumulates log p(y_1..t | boss), the marginal likelihood used to weight hypotheses.
The state space is at most 2*K*D_MAX + K*(A_max + R_max) (~1k), so this is exact and cheap.
"""
from __future__ import annotations

import numpy as np

from .boss import Boss, D_MAX, dist_bin
from .primitives import N_CUES, RECOVER_CUE, active_cue, windup_cue


class BossFilter:
    def __init__(self, boss: Boss):
        self.boss = boss
        K = boss.K
        self.K = K
        pmf = boss.windup_pmf()                                  # (2, K, D)
        surv_prev = 1.0 - np.concatenate([np.zeros((2, K, 1)), np.cumsum(pmf, axis=2)[:, :, :-1]], axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.exit_p = np.where(surv_prev > 1e-12, pmf / surv_prev, 1.0)
        self.exit_p = np.clip(self.exit_p, 0.0, 1.0)
        self.exit_p[:, :, -1] = 1.0                              # must leave by D_MAX
        self.stay_p = 1.0 - self.exit_p
        self.a = np.array([m.active for m in boss.moves])
        self.r = np.array([m.recovery for m in boss.moves])
        self.Amax = max(1, int(self.a.max()))
        self.Rmax = max(1, int(self.r.max()))
        # Emission tables: true cue for each state
        self.cueW = np.zeros((K, D_MAX), dtype=int)
        for k, m in enumerate(boss.moves):
            for d in range(1, D_MAX + 1):
                self.cueW[k, d - 1] = windup_cue(m.primitive, d, boss.tell_ticks)
        self.cueA = np.array([active_cue(m.primitive) for m in boss.moves])
        self.eps = boss.cue_noise
        # masks for valid active/recovery slots
        self.maskA = np.arange(self.Amax)[None, :] < self.a[:, None]
        self.maskR = np.arange(self.Rmax)[None, :] < self.r[:, None]
        self.reset()

    def reset(self):
        self.W = np.zeros((2, self.K, D_MAX))
        self.A = np.zeros((self.K, self.Amax))
        self.R = np.zeros((self.K, self.Rmax))
        self.W[0, 0, 0] = 1.0            # every fight starts in move 0 (idle), phase 0, d = 1
        self.log_evidence = 0.0

    # -------------------------------------------------------------- update
    def likelihoods(self, cue: int):
        e, V = self.eps, N_CUES
        lw = np.where(self.cueW == cue, 1 - e + e / V, e / V)                # (K, D)
        la = np.where(self.cueA == cue, 1 - e + e / V, e / V)[:, None]       # (K, 1)
        lr = (1 - e + e / V) if cue == RECOVER_CUE else e / V
        return lw, la, lr

    def update(self, cue: int) -> float:
        """Condition on the cue emitted by the current state. Returns log p(cue | past)."""
        lw, la, lr = self.likelihoods(cue)
        self.W *= lw[None]
        self.A *= la
        self.R *= lr
        z = self.W.sum() + self.A.sum() + self.R.sum()
        if z <= 0:
            raise FloatingPointError("cue impossible under this boss hypothesis")
        self.W /= z
        self.A /= z
        self.R /= z
        self.log_evidence += np.log(z)
        return float(np.log(z))

    # ------------------------------------------------------------- predict
    def predict(self, dist: float, phase: int, ctx: int = 0):
        K, a, r = self.K, self.a, self.r
        W, A, R = self.W, self.A, self.R
        nW = np.zeros_like(W)
        nA = np.zeros_like(A)
        nR = np.zeros_like(R)
        ends = np.zeros(K)
        # wind-up: stay (d -> d+1) or exit
        nW[:, :, 1:] = W[:, :, :-1] * self.stay_p[:, :, :-1]
        exit_mass = (W * self.exit_p).sum(axis=(0, 2))           # (K,)
        to_A = a > 0
        to_R = (~to_A) & (r > 0)
        nA[to_A, 0] += exit_mass[to_A]
        nR[to_R, 0] += exit_mass[to_R]
        ends[(~to_A) & (r == 0)] += exit_mass[(~to_A) & (r == 0)]
        # active: shift or finish
        if self.Amax > 1:
            shift = A[:, :-1] * self.maskA[:, 1:]
            nA[:, 1:] += shift
        last_A = A[np.arange(K), np.maximum(a - 1, 0)] * (a > 0)
        nR[(a > 0) & (r > 0), 0] += last_A[(a > 0) & (r > 0)]
        ends[(a > 0) & (r == 0)] += last_A[(a > 0) & (r == 0)]
        # recovery: shift or finish
        if self.Rmax > 1:
            nR[:, 1:] += R[:, :-1] * self.maskR[:, 1:]
        last_R = R[np.arange(K), np.maximum(r - 1, 0)] * (r > 0)
        ends += last_R
        # new move
        nW[phase, :, 0] += ends @ self.boss.trans(dist_bin(dist), ctx)
        self.W, self.A, self.R = nW, nA, nR

    def step(self, obs: dict):
        """Convenience: update on this tick's cue, then predict the next tick's state."""
        lp = self.update(obs["cue"])
        self.predict(obs["dist"], obs["phase"], obs.get("ctx", 0))
        return lp

    # ------------------------------------------------------------- queries
    def move_marginal(self) -> np.ndarray:
        return self.W.sum(axis=(0, 2)) + self.A.sum(axis=1) + self.R.sum(axis=1)

    def stage_marginal(self) -> np.ndarray:
        return np.array([self.W.sum(), self.A.sum(), self.R.sum()])

    def sample_states(self, n: int, rng: np.random.Generator):
        """Sample (k, stage, j, p0) from the current belief."""
        flat = np.concatenate([self.W.ravel(), self.A.ravel(), self.R.ravel()])
        flat = flat / flat.sum()
        idx = rng.choice(len(flat), size=n, p=flat)
        nW, nA = self.W.size, self.A.size
        out = []
        for i in idx:
            if i < nW:
                p, k, d = np.unravel_index(i, self.W.shape)
                out.append((int(k), 0, int(d) + 1, int(p)))
            elif i < nW + nA:
                k, j = np.unravel_index(i - nW, self.A.shape)
                out.append((int(k), 1, int(j) + 1, 0))
            else:
                k, j = np.unravel_index(i - nW - nA, self.R.shape)
                out.append((int(k), 2, int(j) + 1, 0))
        return out
