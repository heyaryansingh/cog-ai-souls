"""The embodied Bayesian agent on Arena, v1 (foundation.md §2–4, first cut).

Everything the agent knows about a boss is learned from its own sensory stream:
  1. Perception: delayed noisy keypoints -> boss-centric pose -> online posterior over motion
     syllables (forward filter of a sticky HMM whose syllable library was learned, unsupervised, on
     many training bosses: the cross-boss schema).
  2. When it hurts: a discrete-time hazard model P(impact k ticks ahead | syllable), learned from the
     boss's impact *sounds* (the audio channel), with Beta-Bernoulli counts and a cross-boss prior.
  3. Where it hurts: a spatial hit map P(hit | syllable, player position in the boss's frame) on a
     polar grid, learned from being hit or not hit at impacts (proprioceptive HP drops).
  4. Body: the agent's own body dynamics (self-model; here the known Body, learnable later).
  5. Acting: a belief-space planner. For each candidate action it rolls its own body forward and
     scores expected damage taken (hazard x hit map x not-invulnerable) against expected damage dealt
     (boss in reach and not about to strike), with stamina and healing terms.

All boss-specific counts persist across attempts on the same boss: learning between deaths.
Timing is kept in "boss time": with perception delay L, the filter at real time t describes the boss
at t - L, and predictions are shifted by L.
"""
from __future__ import annotations

import pickle
from collections import deque

import numpy as np

from .env import (BOSS_R, DIR_ANGLES, GUARD, HEAL, HEAVY, JUMP, LIGHT, N_BUTTONS, NONE, ROLL, Body)
from .perception import boss_centric
from .skeleton import J, N_JOINTS

K_HAZ = 30                     # hazard horizon (ticks, boss time)
R_EDGES = np.array([0.0, 1.0, 1.7, 2.4, 3.1, 3.9, 4.8, 6.0, 99.0])
A_BINS = 12


def polar_bin(local):
    r = np.hypot(local[0], local[1])
    a = np.arctan2(local[1], local[0])
    ri = int(np.clip(np.searchsorted(R_EDGES, r) - 1, 0, len(R_EDGES) - 2))
    ai = int(((a + np.pi) / (2 * np.pi) * A_BINS)) % A_BINS
    return ri, ai


class SyllableLibrary:
    """PCA + standardisation + Gaussian sticky HMM, fitted across many bosses (see fit_library)."""

    def __init__(self, pca, mu, sd, hmm):
        self.pca, self.mu, self.sd = pca, mu, sd
        self.means = hmm.means_
        self.var = np.array([np.diag(c) for c in hmm.covars_]) if hmm.covars_.ndim == 3 else hmm.covars_
        self.logA = np.log(np.maximum(hmm.transmat_, 1e-12))
        self.logpi = np.log(np.maximum(hmm.startprob_, 1e-12))
        self.K = len(self.means)

    def featurize(self, pose_now, pose_prev):
        z = self.pca.transform(np.stack([pose_prev, pose_now]))
        f = np.concatenate([z[1], 3.0 * (z[1] - z[0])])
        return (f - self.mu) / self.sd

    def loglik(self, f):
        d = f[None] - self.means
        return -0.5 * (np.sum(d * d / self.var, 1) + np.sum(np.log(2 * np.pi * self.var), 1))

    def save(self, path):
        pickle.dump(self, open(path, "wb"))

    @staticmethod
    def load(path):
        return pickle.load(open(path, "rb"))


def fit_library(pose_seqs, K=40, seed=0):
    """pose_seqs: list of (T, 51) boss-centric pose arrays from many bosses."""
    from hmmlearn.hmm import GaussianHMM
    from sklearn.decomposition import PCA
    allp = np.vstack(pose_seqs)
    pca = PCA(n_components=12).fit(allp)
    feats = []
    for p in pose_seqs:
        z = pca.transform(p)
        dz = np.vstack([np.zeros((1, 12)), np.diff(z, axis=0)])
        feats.append(np.hstack([z, 3.0 * dz]))
    F = np.vstack(feats)
    mu, sd = F.mean(0), F.std(0) + 1e-3
    hmm = GaussianHMM(n_components=K, covariance_type="diag", n_iter=50, random_state=seed, min_covar=1e-2,
                      transmat_prior=1.0 + 50.0 * np.eye(K))
    hmm.fit((F - mu) / sd, [len(f) for f in feats])
    return SyllableLibrary(pca, mu, sd, hmm)


class BossMemory:
    """What the agent has learned about ONE boss (persists across attempts), on top of a prior."""

    def __init__(self, K, prior=None, prior_weight=1.0):
        self.K = K
        if prior is None:
            self.imp = np.full((K, K_HAZ), 0.02)      # pseudo-counts: impact k ticks after syllable z
            self.tot = np.full(K, 1.0)
            self.hit = np.full((K, len(R_EDGES) - 1, A_BINS), 0.3)
            self.miss = np.full((K, len(R_EDGES) - 1, A_BINS), 0.7)
        else:
            w = prior_weight
            self.imp, self.tot = prior.imp * w, prior.tot * w
            self.hit, self.miss = prior.hit * w, prior.miss * w

    def hazard(self, q):
        """q: syllable posterior -> P(impact exactly k ticks ahead), k = 1..K_HAZ."""
        return np.clip(q @ (self.imp / self.tot[:, None]), 0.0, 1.0)

    def p_hit(self, q, local):
        ri, ai = polar_bin(local)
        h, m = self.hit[:, ri, ai], self.miss[:, ri, ai]
        return float(q @ (h / (h + m)))

    def copy_scaled(self, w):
        c = BossMemory(self.K)
        c.imp, c.tot, c.hit, c.miss = self.imp * w, self.tot * w, self.hit * w, self.miss * w
        return c


def merge_prior(memories, weight=0.2):
    """Empirical-Bayes-style cross-boss prior: average of per-boss memories, down-weighted."""
    K = memories[0].K
    pr = BossMemory(K)
    pr.imp = np.mean([m.imp / m.tot[:, None] for m in memories], 0) * 10.0
    pr.tot = np.full(K, 10.0)
    hr = np.mean([m.hit / (m.hit + m.miss) for m in memories], 0)
    pr.hit, pr.miss = hr * 5.0, (1 - hr) * 5.0
    return pr.copy_scaled(weight)


class BayesAgent:
    """Policy object: call reset(env) at fight start, act(obs, env) when the body is free, observe(obs)
    every tick (the harness in run_fight does this). Uses only obs + own body; env is used for the
    body rules and latency (known self-model) and the action interface."""

    def __init__(self, lib: SyllableLibrary, memory: BossMemory, latency=6, learn=True, seed=0,
                 w_dmg=2.0, horizon=26):
        self.lib, self.mem, self.L, self.learn = lib, memory, latency, learn
        self.rng = np.random.default_rng(seed)
        self.w_dmg, self.horizon = w_dmg, horizon

    # ---------------------------------------------------------------- fight lifecycle
    def reset(self, env):
        self.body: Body = env.body
        self.logq = self.lib.logpi.copy()
        self.prev_pose = None
        self.q_hist = deque(maxlen=K_HAZ + 2)       # syllable posteriors in boss time (most recent last)
        self.body_hist = deque(maxlen=self.L + 3)   # (hp, invulnerable/guarding flag) in real time
        self.t = 0
        self.last_local = np.array([3.0, 0.0])
        self.boss_fwd = 0.0

    def observe(self, obs, env):
        """Call once per tick with the observation that step() returned."""
        self.t += 1
        kp = obs[:51].copy()
        pose = boss_centric(kp[None])[0]
        if self.prev_pose is None:
            self.prev_pose = pose
        f = self.lib.featurize(pose, self.prev_pose)
        self.prev_pose = pose
        # forward filter (boss time t - L)
        ll = self.lib.loglik(f)
        pred = np.logaddexp.reduce(self.logq[:, None] + self.lib.logA, axis=0)
        self.logq = pred + ll
        self.logq -= np.logaddexp.reduce(self.logq)
        q = np.exp(self.logq)
        # player's position in the boss's frame at boss time (from the delayed keypoints, which were
        # rendered relative to the player at that same time)
        kpr = kp.reshape(N_JOINTS, 3)
        pel = kpr[J["pelvis"], :2]
        left = kpr[J["l_hip"], :2] - kpr[J["r_hip"], :2]
        fwd = np.arctan2(-left[0], left[1])
        rel = -pel                                   # player relative to boss pelvis (lock-on frame)
        c, s = np.cos(-fwd), np.sin(-fwd)
        local = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])
        self.last_local, self.boss_fwd = local, fwd
        impact = obs[73] > 0.5                        # audio: [71] grunt, [72] whoosh, [73] impact
        p = env.player
        self.body_hist.append((p.hp, env.in_iframes() or env.airborne() or p.button == GUARD))
        if self.learn and impact and len(self.q_hist) >= 1:
            # hazard counts: an impact now (boss time) happened k ticks after each past posterior
            for k in range(1, min(K_HAZ, len(self.q_hist)) + 1):
                self.mem.imp[:, k - 1] += self.q_hist[-k]
            # hit map: was I hit at that boss time? (real time = L ticks ago)
            if len(self.body_hist) >= self.L + 2:
                # the impact happened during the step at real time t - L: compare HP before and after it
                hp_then = self.body_hist[-(self.L + 2)][0]
                hp_next, prot = self.body_hist[-(self.L + 1)]
                if not prot:
                    hit = hp_next < hp_then - 1e-6
                    ri, ai = polar_bin(local)
                    for k in range(1, min(8, len(self.q_hist)) + 1):
                        ww = self.q_hist[-k] / 8.0
                        if hit:
                            self.mem.hit[:, ri, ai] += ww
                        else:
                            self.mem.miss[:, ri, ai] += ww
        if self.learn and len(self.q_hist) >= K_HAZ:
            self.mem.tot += self.q_hist[-K_HAZ]       # exposure: K_HAZ ticks of future have now been seen
        self.q_hist.append(q)

    # ---------------------------------------------------------------- acting
    def _body_rollout(self, a, H):
        """Own body forward model: per future real tick, (position offset in lock-on frame,
        invulnerable, guarding, busy, attack-hit-tick (0/LIGHT/HEAVY))."""
        B = self.body
        d, btn = divmod(a, N_BUTTONS)
        ang = DIR_ANGLES[d] if not (btn == ROLL and d == 0) else np.pi
        u = np.array([np.cos(ang), np.sin(ang)])
        pos = np.zeros(2)
        out = []
        length = {ROLL: B.roll_ticks, LIGHT: B.light_ticks, HEAVY: B.heavy_ticks, GUARD: B.guard_ticks,
                  JUMP: B.jump_ticks, HEAL: B.heal_ticks}.get(btn, 2)
        for t in range(1, H + 1):
            inv = guard = False
            hit_tick = 0
            if t <= length:
                if btn == ROLL:
                    if t <= B.roll_ticks - 3:
                        pos = pos + B.roll_dist / (B.roll_ticks - 3) * u
                    inv = B.iframe_start <= t < B.iframe_start + B.iframe_len
                elif btn == JUMP:
                    pos = pos + u / B.jump_ticks
                    inv = B.air_start <= t < B.air_start + B.air_len    # (only vs low attacks; approximated)
                elif btn == GUARD:
                    guard = True
                elif btn == NONE and d > 0:
                    pos = pos + B.walk * u
                elif btn == LIGHT and t == B.light_hit:
                    hit_tick = LIGHT
                elif btn == HEAVY and t == B.heavy_hit:
                    hit_tick = HEAVY
            out.append((pos.copy(), inv, guard, t <= length, hit_tick))
        return out

    CANDS = [d * N_BUTTONS + b for d, b in [(0, NONE), (1, NONE), (5, NONE), (3, NONE), (7, NONE),
                                             (5, ROLL), (3, ROLL), (7, ROLL), (1, ROLL), (4, ROLL), (6, ROLL),
                                             (0, LIGHT), (0, HEAVY), (0, GUARD), (0, HEAL)]]

    def act(self, obs, env):
        B, p = self.body, env.player
        q = np.exp(self.logq)
        L, H = self.L, self.horizon
        haz_boss = self.mem.hazard(q)                       # boss-time offsets 1..K_HAZ
        haz = np.zeros(H)
        for k in range(1, H + 1):                           # real-time offset k == boss-time offset k + L
            kb = k + L
            if kb <= K_HAZ:
                haz[k - 1] = haz_boss[kb - 1]
        # the boss moves in the agent's own frame: player local position = last_local - own displacement
        c, s = np.cos(-self.boss_fwd), np.sin(-self.boss_fwd)
        rot = np.array([[c, -s], [s, c]])
        dist_now = np.hypot(*self.last_local) - BOSS_R
        best, best_v = self.CANDS[0], -np.inf
        dmg_scale = 25.0
        vals = []
        for a in self.CANDS:
            d, btn = divmod(a, N_BUTTONS)
            cost = {ROLL: B.roll_cost, LIGHT: B.light_cost, HEAVY: B.heavy_cost}.get(btn, 0.0)
            if p.stamina < cost or (btn == HEAL and p.flasks <= 0):
                vals.append(-np.inf)
                continue
            roll = self._body_rollout(a, H)
            exp_dmg, exp_deal = 0.0, 0.0
            for k, (dpos, inv, guard, busy, hit_tick) in enumerate(roll):
                local = self.last_local + rot @ dpos    # own displacement (lock-on frame) seen in the boss's frame
                ph = self.mem.p_hit(q, local)
                pd = haz[k] * ph
                if inv:
                    pd *= 0.05
                elif guard:
                    pd *= 0.15
                exp_dmg += pd * dmg_scale
                if hit_tick:
                    reach = B.light_reach if hit_tick == LIGHT else B.heavy_reach
                    dd = np.hypot(*local) - BOSS_R
                    if dd <= reach:
                        safe = 1.0 - haz[max(0, k - 4):k + 1].sum()
                        exp_deal += (B.light_dmg if hit_tick == LIGHT else B.heavy_dmg) * max(0.0, safe)
            v = exp_deal - self.w_dmg * exp_dmg
            v -= 0.05 * cost                                           # stamina is a resource
            if btn == HEAL:
                v += (0.6 * (B.max_hp - p.hp) if p.hp < 0.5 * B.max_hp else -20.0)
            if btn == NONE and d == 1 and dist_now > 2.0:
                v += 1.5                                               # close distance when safe
            if btn == NONE and d == 5 and dist_now < 1.2:
                v += 0.5
            vals.append(v)
        vals = np.array(vals)
        return self.CANDS[int(np.argmax(vals + self.rng.normal(0, 1e-3, len(vals))))]


def run_fight(env, agent, max_ticks=None):
    obs = env.reset()
    agent.reset(env)
    done = False
    while not done:
        a = agent.act(obs, env) if env.player_free() else 0
        obs, r, done, info = env.step(a)
        agent.observe(obs, env)
    return info["outcome"], env.t, env.boss.hp / env.spec.stats["hp"]
