"""Deep-RL baseline on Arena: recurrent PPO (GRU) on the raw observation stream, RL^2-style trials.

One RL episode is a *trial*: `fights` consecutive fights against one boss, memory carried across
fights, so a meta-learned policy can adapt in-context across attempts (the model-free counterpart of
the Bayesian agent). The agent decides when its body is free; a movement-only action is held for
`repeat` ticks. Observations: the 88-float Arena observation (noisy keypoints delayed by `latency`,
projectiles, audio, body), plus [new fight, previous outcome].
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn

from .env import Arena, Body, N_ACTIONS, N_BUTTONS, OBS_DIM
from .moves import sample_boss

IN_DIM = OBS_DIM + 2
SCALE = np.ones(IN_DIM, np.float32)
SCALE[:51] = 0.2           # keypoints (m)
SCALE[53:69] = 0.2         # projectiles


class TrialEnv:
    def __init__(self, sampler, fights=3, latency=6, seed=0, repeat=2, body=Body()):
        self.sampler, self.F, self.latency, self.repeat, self.body = sampler, fights, latency, repeat, body
        self.rng = np.random.default_rng(seed)

    def reset_trial(self, boss=None):
        self.boss = boss if boss is not None else self.sampler(self.rng)
        self.fight_i, self.prev, self.outcomes = 0, 0.0, []
        return self._new_fight()

    def _new_fight(self):
        self.env = Arena(self.boss, self.body, seed=int(self.rng.integers(1 << 30)), latency=self.latency)
        return self._enc(self.env.reset(), 1.0)

    def _enc(self, obs, new):
        return np.concatenate([obs, [new, self.prev]]).astype(np.float32) * SCALE

    def step(self, a):
        total, n = 0.0, 0
        move_only = a % N_BUTTONS == 0
        while True:
            obs, r, done, info = self.env.step(a if (n == 0 or move_only) else 0)
            total += r
            n += 1
            if done:
                break
            if self.env.player_free() and (not move_only or n >= self.repeat):
                break
        if done:
            self.outcomes.append(info["outcome"])
            self.prev = 1.0 if info["outcome"] == "win" else -1.0
            self.fight_i += 1
            if self.fight_i >= self.F:
                return np.zeros(IN_DIM, np.float32), total / 100.0, True, True
            return self._new_fight(), total / 100.0, False, True
        return self._enc(obs, 0.0), total / 100.0, False, False


class GRUPolicy(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(IN_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.pi = nn.Linear(hidden, N_ACTIONS)
        self.v = nn.Linear(hidden, 1)
        self.hidden = hidden

    def forward(self, x, h):
        z, h = self.gru(self.inp(x), h)
        return self.pi(z), self.v(z).squeeze(-1), h


def train_sampler(exclude_archetypes=()):
    archs = [a for a in ("knight", "beast", "caster", "giant") if a not in exclude_archetypes]

    def f(rng):
        return sample_boss(rng, str(rng.choice(archs)))
    return f


def ppo_train(n_envs=16, T=128, updates=2000, fights=3, latency=6, lr=3e-4, gamma=0.995, lam=0.95,
              clip=0.2, epochs=4, ent=0.01, seed=0, hidden=256, log=print, ckpt=None, ckpt_every=50):
    torch.manual_seed(seed)
    pol = GRUPolicy(hidden)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    envs = [TrialEnv(train_sampler(), fights, latency, seed=seed * 1000 + i) for i in range(n_envs)]
    obs = np.stack([e.reset_trial() for e in envs])
    h = torch.zeros(1, n_envs, hidden)
    wins, fights_done, ticks = deque(maxlen=500), 0, 0
    import time
    t0 = time.time()
    for u in range(updates):
        h0 = h.clone()
        O, A, LP, V, R, Dn = [], [], [], [], [], []
        with torch.no_grad():
            for t in range(T):
                logits, v, h = pol(torch.from_numpy(obs)[:, None], h)
                d = torch.distributions.Categorical(logits=logits[:, 0])
                a = d.sample()
                O.append(obs); A.append(a.numpy()); LP.append(d.log_prob(a).numpy()); V.append(v[:, 0].numpy())
                rs, ds, nobs = [], [], []
                for i, e in enumerate(envs):
                    t_before = e.env.t
                    o, r, done, ended = e.step(int(a[i]))
                    ticks += max(1, e.env.t - t_before) if not ended else 1
                    if ended:
                        wins.append(e.outcomes[-1] == "win")
                        fights_done += 1
                    if done:
                        o = e.reset_trial()
                        h[:, i] = 0
                    rs.append(r); ds.append(done); nobs.append(o)
                R.append(rs); Dn.append(ds)
                obs = np.stack(nobs)
            _, v_last, _ = pol(torch.from_numpy(obs)[:, None], h)
            v_last = v_last[:, 0].numpy()
        R, Dn, V = np.array(R, np.float32), np.array(Dn, np.float32), np.array(V)
        adv = np.zeros_like(R)
        last = np.zeros(n_envs, np.float32)
        for t in reversed(range(T)):
            nv = v_last if t == T - 1 else V[t + 1]
            delta = R[t] + gamma * nv * (1 - Dn[t]) - V[t]
            last = delta + gamma * lam * (1 - Dn[t]) * last
            adv[t] = last
        ret = adv + V
        O_t = torch.from_numpy(np.stack(O, 1)); A_t = torch.from_numpy(np.stack(A, 1)); LP_t = torch.from_numpy(np.stack(LP, 1))
        adv_t = torch.from_numpy(adv.T.copy()); adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.from_numpy(ret.T.copy()); done_t = torch.from_numpy(Dn.T.copy())
        for _ in range(epochs):
            hh, lg_all, v_all = h0, [], []
            for t in range(T):
                lg, vv, hh = pol(O_t[:, t:t + 1], hh)
                lg_all.append(lg[:, 0]); v_all.append(vv[:, 0])
                hh = hh * (1 - done_t[:, t])[None, :, None]
            lg_all, v_all = torch.stack(lg_all, 1), torch.stack(v_all, 1)
            d = torch.distributions.Categorical(logits=lg_all)
            ratio = torch.exp(d.log_prob(A_t) - LP_t)
            loss_pi = -torch.min(ratio * adv_t, torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t).mean()
            loss_v = 0.5 * ((v_all - ret_t) ** 2).mean()
            loss = loss_pi + 0.5 * loss_v - ent * d.entropy().mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(pol.parameters(), 0.5); opt.step()
        if u % 10 == 0:
            log(f"update {u:5d}  ticks {ticks/1e6:7.2f}M  fights {fights_done:6d}  recent win {np.mean(wins) if wins else 0:.3f}  "
                f"loss_v {loss_v.item():.3f}  ent {d.entropy().mean().item():.2f}  {ticks/(time.time()-t0):.0f} ticks/s")
        if ckpt and u % ckpt_every == 0:
            torch.save(pol.state_dict(), ckpt)
    return pol


def evaluate_policy(pol, bosses, attempts=6, latency=6, seed=0, greedy=False):
    out = []
    for bi, boss in enumerate(bosses):
        env = TrialEnv(None, attempts, latency, seed=seed + bi)
        obs = env.reset_trial(boss)
        h = torch.zeros(1, 1, pol.hidden)
        done = False
        with torch.no_grad():
            while not done:
                logits, _, h = pol(torch.from_numpy(obs)[None, None], h)
                a = int(logits[0, 0].argmax()) if greedy else int(torch.distributions.Categorical(logits=logits[0, 0]).sample())
                obs, _, done, _ = env.step(a)
        out.append(env.outcomes)
    return out
