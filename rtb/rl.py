"""RL baselines: PPO with a GRU policy, trained RL^2-style.

One RL "episode" is a *trial*: `fights_per_trial` consecutive fights against the same boss,
with the recurrent state carried across fights (reset only at trial start). A recurrent
policy that has meta-learned can therefore adapt to a boss in-context across attempts,
which is the fair model-free counterpart of the slow mind. With fights_per_trial=1 and a
feed-forward-like memory reset each fight it is plain recurrent PPO.

The agent decides only when the player is free (like the planners); reward is summed over
the ticks between decisions. Observations use the same perceptual stream as the planners
(cue delayed by `latency` ticks).
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn

from .boss import sample_boss, theta_star, SPREAD, TELL_TICKS, CUE_NOISE, DMG_SCALE, sample_meta_theta
from .env import MiniSouls, Rules, N_ACTIONS
from .primitives import N_CUES

OBS_DIM = N_CUES + 1 + 10


def encode(obs, cue, new_fight, prev_outcome):
    v = np.zeros(OBS_DIM, np.float32)
    if cue is not None:
        v[cue] = 1.0
    else:
        v[N_CUES] = 1.0
    o = N_CUES + 1
    v[o:o + 10] = [obs["dist"] / 8.0, obs["phase"], obs["stamina"] / 100.0, obs["hp"] / 100.0,
                   obs["boss_hp"] / 200.0, obs["heals"] / 2.0, obs["ctx"] == 1, obs["ctx"] == 2,
                   new_fight, prev_outcome]
    return v


class TrialEnv:
    """Decision-level wrapper: one step = one decision; a trial = several fights vs one boss."""

    def __init__(self, boss_sampler, fights_per_trial=4, latency=4, rules=Rules(), seed=0):
        self.sampler, self.F, self.latency, self.rules = boss_sampler, fights_per_trial, latency, rules
        self.rng = np.random.default_rng(seed)

    def reset_trial(self, boss=None):
        self.boss = boss if boss is not None else self.sampler(self.rng)
        self.fight_i, self.prev_outcome = 0, 0.0
        self.outcomes = []
        return self._new_fight()

    def _new_fight(self):
        self.env = MiniSouls(self.boss, self.rules, seed=int(self.rng.integers(1 << 30)))
        obs = self.env.reset()
        self.pending = deque()
        self.obs = obs
        return encode(obs, None, 1.0, self.prev_outcome)

    def _seen_cue(self):
        if self.latency == 0:
            return self.obs["cue"]
        # exactly `latency` ticks old: the planners' filters see the same stream
        return self.pending[0]["cue"] if len(self.pending) > self.latency else None

    def step(self, a):
        """Returns (obs_vec, reward, trial_done, fight_ended)."""
        total = 0.0
        a_now = a
        while True:
            obs, r, done, info = self.env.step(a_now)
            total += r
            self.pending.append(obs)
            if len(self.pending) > self.latency + 1:
                self.pending.popleft()
            self.obs = obs
            a_now = 0
            if done or obs["free"]:
                break
        if done:
            win = info["outcome"] == "win"
            self.outcomes.append(info["outcome"])
            self.prev_outcome = 1.0 if win else -1.0
            self.fight_i += 1
            if self.fight_i >= self.F:
                return np.zeros(OBS_DIM, np.float32), total / 100.0, True, True
            return self._new_fight(), total / 100.0, False, True
        return encode(obs, self._seen_cue(), 0.0, self.prev_outcome), total / 100.0, False, False


class GRUPolicy(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.pi = nn.Linear(hidden, N_ACTIONS)
        self.v = nn.Linear(hidden, 1)
        self.hidden = hidden

    def forward(self, x, h):
        z, h = self.gru(self.inp(x), h)
        return self.pi(z), self.v(z).squeeze(-1), h


def train_boss_sampler(kind="meta"):
    """'meta': bosses from the broad meta-prior (pretraining distribution);
    'theta': bosses from the narrow Theta* (same distribution as the train/test splits)."""
    th = theta_star(SPREAD)

    def sampler(rng):
        t = sample_meta_theta(rng) if kind == "meta" else th
        return sample_boss(t, rng, tell_ticks=TELL_TICKS, cue_noise=CUE_NOISE, dmg_scale=DMG_SCALE)
    return sampler


def ppo_train(n_envs=16, steps_per_update=256, updates=500, fights_per_trial=4, latency=4, lr=3e-4,
              gamma=0.995, lam=0.95, clip=0.2, epochs=4, ent=0.01, seed=0, sampler="theta", log=print,
              hidden=128):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    pol = GRUPolicy(hidden)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    envs = [TrialEnv(train_boss_sampler(sampler), fights_per_trial, latency, seed=seed * 1000 + i)
            for i in range(n_envs)]
    obs = np.stack([e.reset_trial() for e in envs])
    h = torch.zeros(1, n_envs, hidden)
    wins, fights = deque(maxlen=400), 0
    for u in range(updates):
        h0 = h.clone()
        O, A, LP, V, R, Dn = [], [], [], [], [], []
        with torch.no_grad():
            for t in range(steps_per_update):
                logits, v, h = pol(torch.from_numpy(obs)[:, None], h)
                dist = torch.distributions.Categorical(logits=logits[:, 0])
                a = dist.sample()
                O.append(obs)
                A.append(a.numpy())
                LP.append(dist.log_prob(a).numpy())
                V.append(v[:, 0].numpy())
                rs, ds, nobs = [], [], []
                for i, e in enumerate(envs):
                    o, r, done, ended = e.step(int(a[i]))
                    if ended:
                        wins.append(e.outcomes[-1] == "win")
                        fights += 1
                    if done:
                        o = e.reset_trial()
                        h[:, i] = 0
                    rs.append(r)
                    ds.append(done)
                    nobs.append(o)
                R.append(rs)
                Dn.append(ds)
                obs = np.stack(nobs)
            _, v_last, _ = pol(torch.from_numpy(obs)[:, None], h)
            v_last = v_last[:, 0].numpy()
        R, Dn, V = np.array(R, np.float32), np.array(Dn, np.float32), np.array(V)
        adv = np.zeros_like(R)
        last = np.zeros(n_envs, np.float32)
        for t in reversed(range(steps_per_update)):
            nv = v_last if t == steps_per_update - 1 else V[t + 1]
            delta = R[t] + gamma * nv * (1 - Dn[t]) - V[t]
            last = delta + gamma * lam * (1 - Dn[t]) * last
            adv[t] = last
        ret = adv + V
        O_t = torch.from_numpy(np.stack(O, 1))            # (N, T, obs)
        A_t = torch.from_numpy(np.stack(A, 1))
        LP_t = torch.from_numpy(np.stack(LP, 1))
        adv_t = torch.from_numpy(adv.T.copy())
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.from_numpy(ret.T.copy())
        done_t = torch.from_numpy(Dn.T.copy())
        for _ in range(epochs):
            # replay the chunk with hidden resets at trial boundaries
            hh, logits_all, v_all = h0, [], []
            for t in range(steps_per_update):
                lg, vv, hh = pol(O_t[:, t:t + 1], hh)
                logits_all.append(lg[:, 0])
                v_all.append(vv[:, 0])
                hh = hh * (1 - done_t[:, t])[None, :, None]
            logits_all = torch.stack(logits_all, 1)
            v_all = torch.stack(v_all, 1)
            d = torch.distributions.Categorical(logits=logits_all)
            lp = d.log_prob(A_t)
            ratio = torch.exp(lp - LP_t)
            loss_pi = -torch.min(ratio * adv_t, torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t).mean()
            loss_v = 0.5 * ((v_all - ret_t) ** 2).mean()
            loss = loss_pi + 0.5 * loss_v - ent * d.entropy().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
            opt.step()
        if u % 10 == 0:
            log(f"update {u:4d}  fights {fights:6d}  recent win {np.mean(wins) if wins else 0:.3f}  "
                f"loss_v {loss_v.item():.3f}  ent {d.entropy().mean().item():.3f}")
    return pol


def evaluate_policy(pol, bosses, attempts=12, latency=4, seed=0, greedy=False):
    """Per-boss attempt curves with memory carried across attempts (RL^2 protocol)."""
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
