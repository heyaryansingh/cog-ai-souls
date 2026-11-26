"""Episodic boss reader: learns *this* boss's timing between deaths, the way a player does.

No cross-boss training. During a fight the agent sees only what a player sees: noisy, delayed, dropping-out
3D keypoints of the boss (Sensor, calibrated to the pixel pipeline), the sounds (grunt at a move's start in
60% of moves, whoosh of a fast swing, impact), its own body (hp, flasks, current action).

Memory: every frame it stores a context vector (recent boss-centric pose, its change, time since the last
whoosh and grunt, distance, whether it is healing, boss phase) and, once the future is known, the label
"frames until the next danger" (danger = a contact the player feels: hit, blocked, or passed through during i-frames), censored at 40.
Decision: k nearest contexts in memory give a predictive distribution over time-to-danger:
  * roll when P(danger lands inside the roll's i-frames) is high (roll direction: Thompson bandit);
  * attack when P(danger within the attack's commitment) is low;
  * heal when far and P(danger within the heal) is low (and it learns heals get punished, from labels).
Memory persists across attempts (episodic learning between deaths); `forget=True` is the control.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .agent import Sensor, centric
from .env import HEAL, LIGHT, N_BUTTONS, NONE, ROLL

CAP = 40
ARM = np.r_[9:21]                      # radius + wrist joints in the 39-d centric pose


class EpisodicReader:
    def __init__(self, k=50, roll_thr=0.35, atk_thr=0.25, heal_thr=0.2, forget=False, seed=0, cap_mem=40000,
                 i_lo=3, i_hi=10):
        import os
        roll_thr = float(os.environ.get("READER_ROLL", roll_thr)); atk_thr = float(os.environ.get("READER_ATK", atk_thr))
        self.k, self.roll_thr, self.atk_thr, self.heal_thr = k, roll_thr, atk_thr, heal_thr
        self.forget, self.rng, self.cap_mem = forget, np.random.default_rng(seed), cap_mem
        self.i_lo, self.i_hi = i_lo, i_hi
        self.X = np.zeros((0, 0), np.float32)
        self.Y = np.zeros(0, np.float32)
        self.dir_ab = {d: [1.0, 1.0] for d in (3, 7, 5, 1)}      # roll direction bandit: (no-hit, hit)
        self.attempt = 0
        # offline calibration (B12): timing context alone (AUC .785) beats noisy pose (.66) and pose+context (.70)
        import os
        self.cols = {"ctx": np.r_[78:83], "all": np.r_[0:83], "pose": np.r_[0:78]}[os.environ.get("READER_FEAT", "ctx")]

    # -- per fight ---------------------------------------------------------------------------------
    def reset(self, env):
        if self.forget:
            self.X, self.Y = np.zeros((0, 0), np.float32), np.zeros(0, np.float32)
        self.sen = Sensor(env, seed=int(self.rng.integers(1 << 30)))
        self.poses = deque(maxlen=16)
        self.pending = deque()               # (t, feature)
        self.t, self.last_whoosh, self.last_grunt = 0, -999, -999
        self.last_x, self.hip, self.h, self.last_P = None, None, 3.0, None
        self.roll_dir, self.roll_t = None, -999
        self.attempt += 1
        self._mu = self.X.mean(0) if len(self.X) else None
        self._sd = self.X.std(0) + 1e-3 if len(self.X) else None

    def _feature(self, env):
        P, _ = self.sen()
        if P is not None:
            self.last_P = P
            x, hip, fwd, h = centric(P)
            self.last_x, self.hip, self.h = x, hip, h
        self.poses.append(self.last_x)
        ps = [p for p in self.poses if p is not None]
        if not ps:
            return None
        a = np.array(ps)
        m4 = a[-4:].mean(0)
        old = a[:max(1, len(a) - 8)][-4:].mean(0)
        dist = np.linalg.norm(self.hip[:2] - env.p_xy) / self.h if self.hip is not None else 1.0
        tw = min(self.t - self.last_whoosh, 60) / 20.0
        tg = min(self.t - self.last_grunt, 60) / 20.0
        heal = float(env.act == HEAL)
        return np.concatenate([m4, 2.0 * (m4 - old), [3 * tw, 3 * tg, 2 * dist, 3 * heal, 2 * float(env.phase)]]
                              ).astype(np.float32)

    def _predict(self, f):
        """Empirical distribution of time-to-danger among the k nearest remembered contexts (None if no memory)."""
        if len(self.Y) < 200:
            return None
        if self._mu is None:
            self._mu, self._sd = self.X.mean(0), self.X.std(0) + 1e-3
        sl = self.cols
        d = (((self.X[:, sl] - f[sl]) / self._sd[sl]) ** 2).sum(1)
        nn = np.argpartition(d, self.k)[:self.k]
        return self.Y[nn]

    def act(self, obs, env):
        import os
        f = self.f
        if f is None or not env.player_free():
            return 0
        y = self._predict(f)
        dist = np.linalg.norm(self.hip[:2] - env.p_xy) if self.hip is not None else 3.0   # perceived
        reach = 0.8 + 0.25 * env.spec.scale          # own body knowledge: how close the sword needs to be
        if y is None:                                  # first attempt, no memory yet: naive aggression
            return LIGHT if dist <= reach else 1 * N_BUTTONS + NONE
        p_roll = np.mean((y >= self.i_lo) & (y <= self.i_hi))
        if p_roll > self.roll_thr:
            ds = list(self.dir_ab)
            draws = [self.rng.beta(*self.dir_ab[d]) for d in ds]
            self.roll_dir, self.roll_t = ds[int(np.argmax(draws))], self.t
            return self.roll_dir * N_BUTTONS + ROLL
        p30 = np.mean(y < 30)
        if env.hp < 45 and env.flasks and dist > 3.0 and np.mean(y < 50) < self.heal_thr:
            return HEAL
        if dist <= reach:
            return LIGHT if p30 < self.atk_thr else 5 * N_BUTTONS + NONE
        if os.environ.get("READER_SPACING", "1") == "0":
            if np.mean(y < 15) > 0.5:
                return 5 * N_BUTTONS + NONE
            return 1 * N_BUTTONS + NONE
        # counterfactual query: "if I were in reach now, would I get hit soon?" (same context, distance swapped)
        g = f.copy(); g[80] = 2 * reach / max(self.h, 1e-3)
        y_in = self._predict(g)
        if np.mean(y_in < 30) < self.atk_thr:
            return 1 * N_BUTTONS + NONE                # opening: close in
        if dist < reach + 1.5:
            return 5 * N_BUTTONS + NONE                # hold spacing
        return 3 * N_BUTTONS + NONE                    # circle at range

    def observe(self, obs, env, info):
        au = obs["audio"]
        # danger = contact the player feels: a hit, a blocked hit, or a blade passing through during i-frames
        danger = bool(info.get("hit") is not None or info.get("blocked") or info.get("dodged"))
        # near-miss learning: a swing heard while the boss's hands were seen close to me also counts, so every
        # nearby swing teaches timing, not only the ones that connect (dense labels, as a watching player gets)
        import os
        if os.environ.get("READER_LABEL", "near") == "near" and au[1] and self.last_P is not None:
            dw = np.linalg.norm(self.last_P[[5, 6], :2] - env.p_xy, axis=1).min()
            danger = danger or dw < float(os.environ.get("READER_NEAR", 1.6))
        if au[1]:
            self.last_whoosh = self.t
        if au[0]:
            self.last_grunt = self.t
        if self.roll_dir is not None and self.t - self.roll_t == 20:
            self.dir_ab[self.roll_dir][0] += 1
            self.roll_dir = None
        if self.roll_dir is not None and info.get("hit") is not None:
            self.dir_ab[self.roll_dir][1] += 1
            self.roll_dir = None
        if danger:                                   # label everything still pending
            for (t0, f0) in self.pending:
                self._store(f0, self.t - t0)
            self.pending.clear()
        while self.pending and self.t - self.pending[0][0] >= CAP:
            t0, f0 = self.pending.popleft()
            self._store(f0, CAP)
        self.t += 1
        self.f = self._feature(env)
        if self.f is not None:
            self.pending.append((self.t, self.f))

    def _store(self, f, y):
        if self.X.size == 0:
            self.X = f[None].copy()
            self.Y = np.array([y], np.float32)
            return
        if len(self.Y) >= self.cap_mem:
            i = int(self.rng.integers(len(self.Y)))
            self.X[i], self.Y[i] = f, y
        else:
            self._buf = getattr(self, "_buf", [])
            self._buf.append((f, y))
            if len(self._buf) >= 256:
                self.X = np.vstack([self.X, np.array([b[0] for b in self._buf])])
                self.Y = np.concatenate([self.Y, np.array([b[1] for b in self._buf], np.float32)])
                self._buf = []
                self._mu = None


def run_attempt(env, agent):
    obs = env.reset()
    agent.reset(env)
    agent.f = agent._feature(env)
    done, info = False, {}
    c = {"roll": 0, "light": 0, "heal": 0, "hit": 0, "dodged": 0, "blocked": 0}
    while not done:
        a = agent.act(obs, env)
        b = a % N_BUTTONS
        c["roll"] += b == ROLL; c["light"] += b == LIGHT; c["heal"] += b == HEAL
        obs, r, done, info = env.step(a)
        c["hit"] += info.get("hit") is not None; c["dodged"] += bool(info.get("dodged")); c["blocked"] += bool(info.get("blocked"))
        agent.observe(obs, env, info)
    agent.counts = {k: int(v) for k, v in c.items()}
    return info["outcome"], env.t, env.boss_hp / env.spec.hp, env.hp


class EpisodicControl(EpisodicReader):
    """Hippocampal-style episodic control (Blundell et al. 2016; Lengyel & Dayan 2007) on the reader's
    context: for each action, remember (context -> what happened over the next H frames) and choose the action
    whose k nearest remembered outcomes were best. No thresholds; learning is purely between/within attempts.
    Return = damage dealt/10 - damage taken/10 - 10 on death, discounted over H = 60 frames."""
    ACTS = None

    def __init__(self, k=20, H=60, gamma=0.97, eps=0.1, hold=6, seed=0, forget=False, opt=2.0):
        super().__init__(seed=seed, forget=forget)
        from .env import N_BUTTONS as NB
        self.ACTS = [1 * NB + NONE, 5 * NB + NONE, 3 * NB + NONE, 7 * NB + NONE, LIGHT, HEAL,
                     3 * NB + ROLL, 7 * NB + ROLL, 5 * NB + ROLL, 1 * NB + ROLL]
        self.kq, self.H, self.g, self.eps, self.hold_n, self.opt = k, H, gamma, eps, hold, opt
        self.QX = {a: [] for a in range(len(self.ACTS))}
        self.QY = {a: [] for a in range(len(self.ACTS))}

    def reset(self, env):
        if self.forget:
            self.QX = {a: [] for a in range(len(self.ACTS))}
            self.QY = {a: [] for a in range(len(self.ACTS))}
        super().reset(env)
        self.open_q = []                    # (t, action index, state vector)
        self.rew = []
        self.cur, self.cur_left = 0, 0
        self.prev_bhp, self.prev_hp = env.boss_hp, env.hp
        self._arr = {}

    def _state(self, env):
        f = self.f
        return np.concatenate([f[78:83], [env.hp / 100 * 3, env.stamina / 100 * 2, env.flasks / 4,
                                         float(env.player_free())]]).astype(np.float32)

    def _q(self, ai, s):
        X = self._arr.get(ai)
        if X is None or len(X[0]) <= self.kq:
            return self.opt                           # optimism: try untried actions
        d = (((X[0] - s) / X[2]) ** 2).sum(1)
        nn = np.argpartition(d, self.kq)[:self.kq]
        return float(X[1][nn].mean())

    def act(self, obs, env):
        if self.f is None or not env.player_free():
            return 0
        if self.cur_left > 0 and self.ACTS[self.cur] % N_BUTTONS == NONE:
            self.cur_left -= 1
            return self.ACTS[self.cur]
        s = self._state(env)
        ok = [i for i, a in enumerate(self.ACTS) if not (a == HEAL and env.flasks == 0)]
        if self.rng.random() < self.eps:
            ai = int(self.rng.choice(ok))
        else:
            qs = [self._q(i, s) for i in ok]
            ai = ok[int(np.argmax(qs))]
        self.cur, self.cur_left = ai, self.hold_n - 1
        self.open_q.append((self.t, ai, s))
        return self.ACTS[ai]

    def observe(self, obs, env, info):
        super().observe(obs, env, info)
        r = (self.prev_bhp - env.boss_hp) / 10 - max(0.0, self.prev_hp - env.hp) / 10
        if env.hp <= 0:
            r -= 10
        self.prev_bhp, self.prev_hp = env.boss_hp, env.hp
        self.rew.append(r)
        done = env.hp <= 0 or env.boss_hp <= 0 or env.t >= env.max_frames
        while self.open_q and (self.t - self.open_q[0][0] >= self.H or done):
            t0, ai, s = self.open_q.pop(0)
            seg = self.rew[t0:t0 + self.H]
            G = float(sum(self.g ** i * x for i, x in enumerate(seg)))
            self.QX[ai].append(s); self.QY[ai].append(G)
        if done or self.t % 300 == 0:
            for ai in self.QX:
                if self.QX[ai]:
                    X = np.array(self.QX[ai]); self._arr[ai] = (X, np.array(self.QY[ai]), X.std(0) + 0.05)
