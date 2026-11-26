"""The Bayesian agent on Arena3D (port of arena/agent.py to 3D bodies and mocap bosses).

Perception is either the full pixel pipeline (perception.Observer) or a calibrated sensor model that
reproduces its measured error: 13 body joints with Gaussian noise of 6% of body height per axis, bursty
dropout (~30% of frames missing), and 180 ms latency. No weapon keypoint: the pose network does not see
the weapon, so neither does the agent (a known gap).

  1. syllables: boss-centric, height-normalised 3D pose + velocity -> sticky HMM forward filter, library
     learned across training bosses (the cross-boss schema);
  2. when: hazard P(swing onset k frames ahead | syllable), learned from the *whoosh* sound;
  3. where: P(hit | syllable, my position in the boss's frame) on a polar grid in units of boss height,
     learned from being hit or not after each swing (proprioception);
  4. acting: belief-space planner over its own body rollouts (roll, walk, guard, attack, heal).
Boss-specific counts persist across attempts (learning between deaths) on top of a cross-boss prior.
"""
from __future__ import annotations

import pickle
from collections import deque

import numpy as np

from .env import GUARD, HEAL, LIGHT, N_BUTTONS, NONE, ROLL

J13 = ["head", "lhumerus", "rhumerus", "lradius", "rradius", "lwrist", "rwrist", "lfemur", "rfemur", "ltibia",
       "rtibia", "lfoot", "rfoot"]
K_HAZ = 40
R_EDGES = np.array([0.0, 0.35, 0.6, 0.85, 1.1, 1.4, 1.8, 2.4, 99.0])      # in boss heights
A_BINS = 12
SWING_WIN = 14                       # frames after a whoosh in which a hit is attributed to that swing


def polar_bin(local):
    r = np.hypot(local[0], local[1])
    a = np.arctan2(local[1], local[0])
    ri = int(np.clip(np.searchsorted(R_EDGES, r) - 1, 0, len(R_EDGES) - 2))
    ai = int((a + np.pi) / (2 * np.pi) * A_BINS) % A_BINS
    return ri, ai


class Sensor:
    """Calibrated stand-in for the pixel pipeline (see module doc). Returns (pose39 or None, frame)."""

    def __init__(self, env, latency=6, noise=0.06, p_stay_on=0.95, p_stay_off=0.88, seed=0, tip=None):
        import os
        latency = int(os.environ.get("SENSE_LAT", latency))
        noise = float(os.environ.get("SENSE_NOISE", noise))
        if os.environ.get("SENSE_DROP") == "0":
            p_stay_on = 1.0
        self.tip = tip if tip is not None else os.environ.get("SENSE_TIP") == "1"
        self.env, self.L, self.noise = env, latency, noise
        self.ids = [env.m.body("b_" + n).id for n in J13]
        self.rng = np.random.default_rng(seed)
        self.on, self.p_on, self.p_off = True, p_stay_on, p_stay_off
        self.q = deque(maxlen=latency + 1)

    def __call__(self):
        env = self.env
        P = env.d.xpos[self.ids].copy()
        if self.tip:     # weapon tip as a 14th keypoint (what a weapon-aware detector would add)
            P = np.vstack([P, env.d.site_xpos[env.tip]])
        h = 1.75 * env.spec.scale
        self.on = self.rng.random() < (self.p_on if self.on else 1 - self.p_off)
        P += self.rng.normal(0, self.noise * h, P.shape)
        self.q.append((P if self.on else None, env.p_xy.copy()))
        return self.q[0] if len(self.q) > self.L else (None, env.p_xy.copy())


def boss_frame(P):
    """Mid-hip centre, facing from the hip line, height from torso length. P: (13,3) world joints."""
    hip = 0.5 * (P[7] + P[8])
    sho = 0.5 * (P[1] + P[2])
    left = P[7] - P[8]
    fwd = np.arctan2(left[0], -left[1])            # up x left  -> forward (z-up)
    height = 3.3 * np.linalg.norm(sho - hip)
    return hip, fwd, height


def centric(P):
    hip, fwd, h = boss_frame(P)
    c, s = np.cos(-fwd), np.sin(-fwd)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    X = (P - np.array([hip[0], hip[1], 0.0])) @ R.T / max(h, 1e-3)
    return X.ravel(), hip, fwd, h


class SyllableLibrary:
    def __init__(self, pca, mu, sd, hmm):
        self.pca, self.mu, self.sd = pca, mu, sd
        self.means = hmm.means_
        self.var = np.array([np.diag(c) for c in hmm.covars_]) if hmm.covars_.ndim == 3 else hmm.covars_
        self.logA = np.log(np.maximum(hmm.transmat_, 1e-12))
        self.logpi = np.log(np.maximum(hmm.startprob_, 1e-12))
        self.K = len(self.means)

    def featurize(self, x, xprev):
        z = self.pca.transform(np.stack([xprev, x]))
        return (np.concatenate([z[1], 3.0 * (z[1] - z[0])]) - self.mu) / self.sd

    def loglik(self, f):
        d = f[None] - self.means
        return -0.5 * (np.sum(d * d / self.var, 1) + np.sum(np.log(2 * np.pi * self.var), 1))

    def save(self, path):
        pickle.dump(self, open(path, "wb"))

    @staticmethod
    def load(path):
        return pickle.load(open(path, "rb"))


def fit_library(seqs, K=40, seed=0, ncomp=12):
    from hmmlearn.hmm import GaussianHMM
    from sklearn.decomposition import PCA
    pca = PCA(n_components=ncomp).fit(np.vstack(seqs))
    feats = []
    for p in seqs:
        z = pca.transform(p)
        dz = np.vstack([np.zeros((1, ncomp)), np.diff(z, axis=0)])
        feats.append(np.hstack([z, 3.0 * dz]))
    F = np.vstack(feats)
    mu, sd = F.mean(0), F.std(0) + 1e-3
    hmm = GaussianHMM(n_components=K, covariance_type="diag", n_iter=40, random_state=seed, min_covar=1e-2,
                      transmat_prior=1.0 + 50.0 * np.eye(K))
    hmm.fit((F - mu) / sd, [len(f) for f in feats])
    return SyllableLibrary(pca, mu, sd, hmm)


class BossMemory:
    def __init__(self, K):
        self.K = K
        self.imp = np.full((K, K_HAZ), 0.01)
        self.tot = np.full(K, 1.0)
        self.hit = np.full((K, len(R_EDGES) - 1, A_BINS), 0.3)
        self.miss = np.full((K, len(R_EDGES) - 1, A_BINS), 0.7)

    def hazard(self, q):
        return np.clip(q @ (self.imp / self.tot[:, None]), 0.0, 1.0)

    def p_hit(self, q, local):
        ri, ai = polar_bin(local)
        h, m = self.hit[:, ri, ai], self.miss[:, ri, ai]
        return float(q @ (h / (h + m)))

    def scaled(self, w):
        c = BossMemory(self.K)
        c.imp, c.tot, c.hit, c.miss = self.imp * w, self.tot * w, self.hit * w, self.miss * w
        return c


def merge_prior(memories, weight=0.2):
    K = memories[0].K
    pr = BossMemory(K)
    pr.imp = np.mean([m.imp / m.tot[:, None] for m in memories], 0) * 10.0
    pr.tot = np.full(K, 10.0)
    hr = np.mean([m.hit / (m.hit + m.miss) for m in memories], 0)
    pr.hit, pr.miss = hr * 5.0, (1 - hr) * 5.0
    return pr.scaled(weight)


class BayesAgent3D:
    CANDS = [d * N_BUTTONS + b for d, b in [(0, NONE), (1, NONE), (5, NONE), (3, NONE), (7, NONE),
                                             (5, ROLL), (3, ROLL), (7, ROLL), (1, ROLL), (4, ROLL), (6, ROLL),
                                             (0, LIGHT), (0, GUARD), (0, HEAL)]]

    def __init__(self, lib, memory, latency=6, learn=True, seed=0, w_dmg=1.5, horizon=30, sensor="model"):
        self.lib, self.mem, self.L, self.learn = lib, memory, latency, learn
        self.rng = np.random.default_rng(seed)
        self.w_dmg, self.H, self.sensor_kind = w_dmg, horizon, sensor

    def reset(self, env):
        self.env = env
        self.boss_max = env.spec.hp       # the boss HP bar: its fraction is visible; max known after one hit
        self.sensor = Sensor(env, self.L, seed=int(self.rng.integers(1 << 30)))
        self.logq = self.lib.logpi.copy()
        self.prev_x = None
        self.q_hist = deque(maxlen=K_HAZ + 2)
        self.hp_hist = deque(maxlen=self.L + SWING_WIN + 4)
        self.swings = []                 # pending: (frames since, local bin, q-snapshot list)
        self.local, self.fwd, self.height, self.hip = np.array([2.0, 0.0]), 0.0, 1.75, np.zeros(2)
        self.whoosh_q = deque(maxlen=self.L + 1)

    def observe(self, obs, env):
        P, p_then = self.sensor()
        # sounds are heard without visual latency, but are attributed in boss time (they line up with
        # the delayed picture once L frames have passed)
        self.whoosh_q.append(obs["audio"][1] > 0.5)
        whoosh = self.whoosh_q[0] if len(self.whoosh_q) > self.L else False
        if P is not None:
            x, hip, fwd, h = centric(P)
            self.hip, self.fwd, self.height = hip[:2], fwd, 0.9 * self.height + 0.1 * h
            if self.prev_x is None:
                self.prev_x = x
            f = self.lib.featurize(x, self.prev_x)
            self.prev_x = x
            ll = self.lib.loglik(f)
        else:
            ll = np.zeros(self.lib.K)       # no picture: prediction only
        pred = np.logaddexp.reduce(self.logq[:, None] + self.lib.logA, axis=0)
        self.logq = pred + ll
        self.logq -= np.logaddexp.reduce(self.logq)
        q = np.exp(self.logq)
        rel = p_then - self.hip
        c, s = np.cos(-self.fwd), np.sin(-self.fwd)
        local = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]]) / max(self.height, 0.5)
        self.local = local
        prot = env.in_iframes() or env.act == GUARD
        self.hp_hist.append((env.hp, prot))
        if self.learn and whoosh and self.q_hist:
            for k in range(1, min(K_HAZ, len(self.q_hist)) + 1):
                self.mem.imp[:, k - 1] += self.q_hist[-k]
            self.swings.append([0, polar_bin(local), [self.q_hist[-k] for k in range(1, min(6, len(self.q_hist)) + 1)],
                                env.hp, False])
        # resolve pending swings: hit if HP dropped (unprotected) within SWING_WIN real frames
        for sw in self.swings:
            sw[0] += 1
            if env.hp < sw[3] - 1e-6:
                sw[4] = True
            if prot:
                sw.append("prot")
        done = [sw for sw in self.swings if sw[0] >= SWING_WIN or sw[4]]
        self.swings = [sw for sw in self.swings if not (sw[0] >= SWING_WIN or sw[4])]
        if self.learn:
            for sw in done:
                if "prot" in sw[5:] and not sw[4]:
                    continue                      # protected: says nothing about where it hits
                ri, ai = sw[1]
                for qq in sw[2]:
                    w = qq / len(sw[2])
                    if sw[4]:
                        self.mem.hit[:, ri, ai] += w
                    else:
                        self.mem.miss[:, ri, ai] += w
            if len(self.q_hist) >= K_HAZ:
                self.mem.tot += self.q_hist[-K_HAZ]
        self.q_hist.append(q)

    def _rollout(self, a):
        d, btn = divmod(a, N_BUTTONS)
        to_b = self.hip - self.env.p_xy
        face = np.arctan2(to_b[1], to_b[0])
        dd = d if d else (5 if btn == ROLL else 0)
        ang = face + (dd - 1) * np.pi / 4
        u = np.array([np.cos(ang), np.sin(ang)]) if dd else np.zeros(2)
        pos, out = np.zeros(2), []
        length = {ROLL: 20, LIGHT: 30, HEAL: 50, GUARD: 2}.get(btn, 2)
        for t in range(1, self.H + 1):
            inv = guard = False
            hit = False
            if t <= length:
                if btn == ROLL:
                    if t < 14:
                        pos = pos + 0.195 * u
                    inv = 2 <= t <= 11
                elif btn == GUARD:
                    guard = True
                    pos = pos + 0.06 * u
                elif btn == NONE:
                    pos = pos + 0.12 * u
                elif btn == LIGHT and t == 6:
                    hit = True
            elif btn == NONE:
                pos = pos + 0.12 * u
            out.append((pos.copy(), inv, guard, hit))
        return out

    def act(self, obs, env):
        q = np.exp(self.logq)
        haz_b = self.mem.hazard(q)
        # swing onsets in boss time; hits land ~4-10 frames after onset: smear, and shift by latency
        haz = np.zeros(self.H)
        for k in range(1, self.H + 1):
            for lag in range(3, 11):
                kb = k + self.L - lag
                if 1 <= kb <= K_HAZ:
                    haz[k - 1] += haz_b[kb - 1] / 8.0
        c, s = np.cos(-self.fwd), np.sin(-self.fwd)
        rot = np.array([[c, -s], [s, c]])
        own_now = (env.p_xy - self.sensor.q[0][1]) if self.sensor.q else np.zeros(2)  # moved since boss-time
        dist = np.linalg.norm(env.p_xy - self.hip)
        reach = 0.75 + 0.18 * self.height
        vals = []
        for a in self.CANDS:
            d, btn = divmod(a, N_BUTTONS)
            cost = {ROLL: 18, LIGHT: 12}.get(btn, 0)
            if env.stamina < cost or (btn == HEAL and env.flasks <= 0):
                vals.append(-np.inf)
                continue
            exp_dmg = exp_deal = 0.0
            for k, (dpos, inv, guard, hit) in enumerate(self._rollout(a)):
                local = self.local + (rot @ (own_now + dpos)) / max(self.height, 0.5)
                pd = haz[k] * self.mem.p_hit(q, local)
                pd *= 0.05 if inv else (0.3 if guard else 1.0)
                exp_dmg += pd * 22.0 / 100.0                     # fraction of my HP
                if hit:
                    dd = np.linalg.norm(env.p_xy + dpos - self.hip)
                    if dd <= reach:
                        safe = 1.0 - haz[max(0, k - 6):k + 6].sum()
                        exp_deal += 40.0 / self.boss_max * max(0.0, safe)   # fraction of boss HP (HP bar)
            v = exp_deal - self.w_dmg * exp_dmg * (100.0 / max(env.hp, 20.0)) ** 0.5 - 0.0005 * cost
            if btn == HEAL:
                v += 0.004 * (100 - env.hp) if env.hp < 50 else -0.2
            if btn == NONE and d == 1 and dist > reach * 0.9:
                v += 0.004
            vals.append(v)
        vals = np.array(vals)
        return self.CANDS[int(np.argmax(vals + self.rng.normal(0, 1e-3, len(vals))))]


def run_fight(env, agent):
    obs = env.reset()
    agent.reset(env)
    done = False
    while not done:
        a = agent.act(obs, env) if env.player_free() else 0
        obs, r, done, info = env.step(a)
        agent.observe(obs, env)
    return info["outcome"], env.t, env.boss_hp / env.spec.hp


class ThreatAgent:
    """Agent v2: learned threat network (cross-boss intuition) + per-boss Bayesian last layer, used by the
    same body-rollout planner. Learns between attempts from its own hits and near misses."""

    CANDS = BayesAgent3D.CANDS

    def __init__(self, ckpt="checkpoints/arena3d_threat.pt", head=None, learn=True, latency=6, seed=0,
                 w_dmg=1.5, horizon=26, update_every=120):
        import torch
        from . import threat as T
        self.T, self.torch = T, torch
        d = torch.load(ckpt, weights_only=False)
        self.net = T.ThreatNet()
        self.net.load_state_dict(d["net"])
        self.net.eval()
        self.mu, self.sd = d["mu"].numpy(), d["sd"].numpy()
        self.head = head or T.BayesHead(self.net)
        self.learn, self.L, self.w_dmg, self.H, self.every = learn, latency, w_dmg, horizon, update_every
        self.rng = np.random.default_rng(seed)

    def _embed(self, X):
        with self.torch.no_grad():
            return self.net.embed(self.torch.from_numpy(((X - self.mu) / self.sd).astype(np.float32))).numpy().astype(np.float64)

    def reset(self, env):
        self.env, self.boss_max = env, env.spec.hp
        self.sensor = Sensor(env, self.L, seed=int(self.rng.integers(1 << 30)))
        self.ph = self.T.PoseHistory()
        self.frame = (env.boss.root.copy(), 0.0)
        self.hist = []          # per frame: (feat, my xy, boss frame, height, hit, prot)
        self.buf_E, self.buf_y = [], []

    def observe(self, obs, env, info=None):
        P, _ = self.sensor()
        if P is not None:
            x, hip, fwd, h = centric(P)
            self.ph.push(x, h)
            self.frame = (hip[:2].copy(), fwd)
        else:
            self.ph.push(None, None)
        hit = bool(info and info.get("hit"))
        prot = env.in_iframes() or env.act == GUARD or bool(info and (info.get("blocked") or info.get("dodged")))
        self.hist.append((self.ph.feat(), env.p_xy.copy(), self.frame, self.ph.height, hit, prot))
        if not self.learn:
            return
        # resolve the sample made 26 frames ago for every horizon
        t = len(self.hist) - 1 - 26
        if t >= 0:
            f, _, (bxy, fwd), h, _, _ = self.hist[t]
            c, s = np.cos(-fwd), np.sin(-fwd)
            rows, ys = [], []
            for ki, k in enumerate(self.T.KS):
                win = self.hist[t + k:t + k + 3]
                y = any(w[4] for w in win)
                if not y and any(w[5] for w in win):
                    continue
                rel = self.hist[t + k][1] - bxy
                local = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]]) / max(h, 0.5)
                rows.append(self.T.query_feats(local[None], np.array([ki]), f)[0])
                ys.append(float(y))
            if rows:
                self.buf_E.append(self._embed(np.array(rows, np.float32)))
                self.buf_y.append(np.array(ys))
        if len(self.buf_y) >= self.every:
            self.flush()

    def flush(self):
        if self.buf_y:
            self.head.update(np.vstack(self.buf_E), np.concatenate(self.buf_y))
        self.buf_E, self.buf_y = [], []

    def _p_hits(self, locals_):
        """locals_: (n_cands, len(KS), 2) -> probabilities (n_cands, len(KS))."""
        f = self.ph.feat()
        nC, nK = locals_.shape[:2]
        ki = np.tile(np.arange(nK), nC)
        X = self.T.query_feats(locals_.reshape(-1, 2), ki, f)
        E = self._embed(X.astype(np.float32))
        return (1 / (1 + np.exp(-np.clip(self.head.logits(E), -30, 30)))).reshape(nC, nK)

    def act(self, obs, env):
        bxy, fwd = self.frame
        self.hip = bxy
        h = max(self.ph.height, 0.5)
        c, s = np.cos(-fwd), np.sin(-fwd)
        dist = np.linalg.norm(env.p_xy - bxy)
        reach = 0.75 + 0.18 * h
        cands, rolls = [], []
        for a in self.CANDS:
            d, btn = divmod(a, N_BUTTONS)
            cost = {ROLL: 18, LIGHT: 12}.get(btn, 0)
            if env.stamina < cost or (btn == HEAL and env.flasks <= 0):
                continue
            cands.append(a)
            rolls.append(BayesAgent3D._rollout(self, a))
        L = np.zeros((len(cands), len(self.T.KS), 2))
        for i, ro in enumerate(rolls):
            for j, k in enumerate(self.T.KS):
                rel = env.p_xy + ro[min(k, len(ro)) - 1][0] - bxy
                L[i, j] = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]]) / h
        P = self._p_hits(L)
        vals = []
        for i, a in enumerate(cands):
            d, btn = divmod(a, N_BUTTONS)
            ro = rolls[i]
            exp_dmg = exp_deal = 0.0
            for j, k in enumerate(self.T.KS):
                inv = any(ro[min(kk, len(ro)) - 1][1] for kk in range(k, k + 3))
                guard = ro[min(k, len(ro)) - 1][2]
                exp_dmg += P[i, j] * (0.05 if inv else (0.15 if guard else 1.0)) * 0.22
            if btn == LIGHT and dist <= reach:
                exp_deal = 40.0 / self.boss_max * (1.0 - P[i, :3].max())
            v = exp_deal - self.w_dmg * exp_dmg * (100.0 / max(env.hp, 20.0)) ** 0.5 - 0.0005 * {ROLL: 18, LIGHT: 12}.get(btn, 0)
            if btn == HEAL:
                v += 0.004 * (100 - env.hp) if env.hp < 50 else -0.2
            if btn == NONE and d == 1 and dist > reach * 0.9:
                v += 0.004
            vals.append(v)
        return cands[int(np.argmax(np.array(vals) + self.rng.normal(0, 1e-4, len(vals))))]


def run_fight_v2(env, agent):
    obs = env.reset()
    agent.reset(env)
    done, info = False, {}
    while not done:
        a = agent.act(obs, env) if env.player_free() else 0
        obs, r, done, info = env.step(a)
        agent.observe(obs, env, info)
    if hasattr(agent, "flush") and agent.learn:
        agent.flush()
    return info["outcome"], env.t, env.boss_hp / env.spec.hp
