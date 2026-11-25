"""M3: the intuition net, amortized inference over boss programs.

A small network reads a fight log and proposes a boss program in one forward pass:
for every primitive in the library, how many moves of that primitive the boss has (0/1/2+)
and their typical wind-up (log-median). It is trained by inference compilation: simulate
(boss, fight log) pairs from the broad META prior and regress program from log.

The log is compressed to run-length tokens (cue, run length, distance, phase, ctx), so a
600-tick fight is ~60-120 tokens.

Uses:
  propose(trace)       -> a Boss to seed the slow mind's particles (net + SMC)
  amortization gap     -> net-only program vs SMC-refined program, in vs out of distribution
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn

from .boss import (Boss, Move, N_DIST_BINS, reach_pref, sample_boss, sample_meta_theta, theta_star, SPREAD,
                   TELL_TICKS, CUE_NOISE, DMG_SCALE)
from .env import MiniSouls
from .policies import oracle_continuation
from .primitives import N_CUES, PRIMITIVES

LIB = [n for n in PRIMITIVES if n != "idle"]          # includes the held-out grab
MAX_TOK = 256
TOK_DIM = N_CUES + 5


def tokenize(trace) -> np.ndarray:
    toks = []
    prev, run = None, 0
    for (c, d, p, x) in trace:
        if c == prev:
            run += 1
            continue
        if prev is not None:
            toks.append((prev, run, last))
        prev, run, last = c, 1, (d, p, x)
    if prev is not None:
        toks.append((prev, run, last))
    toks = toks[:MAX_TOK]
    out = np.zeros((MAX_TOK, TOK_DIM), np.float32)
    for i, (c, run, (d, p, x)) in enumerate(toks):
        out[i, c] = 1.0
        out[i, N_CUES:] = [np.log(run) / 4.0, d / 8.0, p, x == 1, x == 2]
    return out, len(toks)


def program_targets(boss: Boss):
    cnt = np.zeros(len(LIB), np.int64)
    lmed = np.zeros(len(LIB), np.float32)
    for m in boss.moves[1:]:
        i = LIB.index(m.primitive.name)
        cnt[i] += 1
        lmed[i] += np.log(m.median)
    has = cnt > 0
    lmed[has] /= cnt[has]
    idle = np.log(boss.moves[0].median)
    return np.minimum(cnt, 2), lmed, has.astype(np.float32), np.float32(idle)


def simulate(n: int, seed: int, meta: bool = True, max_ticks: int = 1500):
    """(tokens, length, targets) for n simulated fights. Fights are played by the state-cheat
    heuristic with occasional random actions, so logs cover deaths and wins."""
    rng = np.random.default_rng(seed)
    X, L, C, M, H, I = [], [], [], [], [], []
    th_star = theta_star(SPREAD)
    for _ in range(n):
        th = sample_meta_theta(rng) if meta else th_star
        if rng.random() < 0.2:                   # sometimes allow the full library (novel moves)
            th = replace(th, allowed=tuple(LIB))
        boss = sample_boss(th, rng, tell_ticks=TELL_TICKS, cue_noise=CUE_NOISE, dmg_scale=DMG_SCALE)
        env = MiniSouls(boss, seed=int(rng.integers(1 << 30)), max_ticks=max_ticks)
        obs = env.reset()
        tr = []
        eps = rng.uniform(0.0, 0.4)
        while True:
            if obs["free"]:
                a = int(rng.integers(8)) if rng.random() < eps else oracle_continuation(env)
            else:
                a = 0
            obs, _, done, _ = env.step(a)
            tr.append((obs["cue"], obs["dist"], obs["phase"], obs["ctx"]))
            if done:
                break
        x, l = tokenize(tr)
        c, m, h, i = program_targets(boss)
        X.append(x); L.append(l); C.append(c); M.append(m); H.append(h); I.append(i)
    return (np.stack(X), np.array(L), np.stack(C), np.stack(M), np.stack(H), np.array(I, np.float32))


class IntuitionNet(nn.Module):
    def __init__(self, d=96, layers=3, heads=4):
        super().__init__()
        self.inp = nn.Linear(TOK_DIM, d)
        self.pos = nn.Parameter(torch.randn(MAX_TOK, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.count = nn.Linear(d, len(LIB) * 3)
        self.lmed = nn.Linear(d, len(LIB) * 2)   # mean and log-sd
        self.idle = nn.Linear(d, 2)

    def forward(self, x, lengths):
        B, T, _ = x.shape
        mask = torch.arange(T)[None, :] >= lengths[:, None]
        z = self.enc(self.inp(x) + self.pos[None, :T], src_key_padding_mask=mask)
        z = z.masked_fill(mask[..., None], 0).sum(1) / lengths[:, None].clamp(min=1)
        return (self.count(z).view(B, len(LIB), 3), self.lmed(z).view(B, len(LIB), 2), self.idle(z))


def loss_fn(net, batch):
    x, l, c, m, h, i = batch
    lc, lm, li = net(x, l)
    ce = nn.functional.cross_entropy(lc.reshape(-1, 3), c.reshape(-1))
    mu, ls = lm[..., 0], lm[..., 1].clamp(-3, 2)
    nll = ((0.5 * ((m - mu) / ls.exp()) ** 2 + ls) * h).sum() / h.sum().clamp(min=1)
    nll_i = (0.5 * ((i - li[:, 0]) / li[:, 1].clamp(-3, 2).exp()) ** 2 + li[:, 1].clamp(-3, 2)).mean()
    return ce + nll + nll_i, ce.item(), nll.item()


def propose(net, trace, template: Boss, theta=None, rng=None) -> Boss:
    """Turn the net's output on one log into a Boss (mode of counts, mean wind-ups)."""
    theta = theta or theta_star(SPREAD)
    x, l = tokenize(trace)
    with torch.no_grad():
        lc, lm, li = net(torch.from_numpy(x)[None], torch.tensor([l]))
    cnt = lc[0].argmax(-1).numpy()
    moves = [Move(PRIMITIVES["idle"], float(np.exp(li[0, 0].item())), theta.within_sigma["idle"])]
    for k, name in enumerate(LIB):
        for j in range(int(cnt[k])):
            med = float(np.exp(lm[0, k, 0].item() + (0.15 if j else 0.0)))
            moves.append(Move(PRIMITIVES[name], med, theta.within_sigma[name]))
    if len(moves) == 1:
        moves.append(Move(PRIMITIVES["slam"], 14.0, theta.within_sigma["slam"]))
    K = len(moves)
    pi = np.stack([np.stack([reach_pref(moves, b, theta.reach_bias) for b in range(N_DIST_BINS)])
                   for _ in range(K)])
    return Boss(moves, pi, float(np.mean(theta.phase2_speed)), template.tell_ticks, template.cue_noise,
                name="intuition", dmg_scale=template.dmg_scale)
