"""Privileged-teacher DAgger ("learning by cheating", Entry 32).

The teacher plans with the TRUE boss program. The student sees belief features computed from the
slow mind's LEARNED mixture (the prior on attempt 1, then the posterior after each death), so it
learns how to act when its model is wrong. Train bosses, `--attempts` fights per boss per iteration,
the slow mind learning between fights as in learn_curve.py.

python scripts/train_reflex_priv.py --init checkpoints/reflex48_it4.pt --iters 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from rtb.boss import make_splits, theta_star, SPREAD
from rtb.learner import SlowMind
from rtb.policies import OraclePlanner
from rtb.reflex import MixtureReflexPolicy, load_reflex, make_net, mixture_features
from rtb.run import fight
from scripts.train_reflex import fit


class Privileged:
    """Teacher (true program) and student (learned mixture) watch the same fight."""

    def __init__(self, boss, particles, weights, net, beta, n_roll, seed):
        self.teacher = OraclePlanner(n_rollouts=n_roll, seed=seed)
        self.student = MixtureReflexPolicy(particles, weights, net=net, seed=seed + 1)
        self.beta, self.rng, self.data = beta, np.random.default_rng(seed + 2), []

    @property
    def latency(self):
        return self.teacher.latency

    @latency.setter
    def latency(self, v):
        self.teacher.latency = self.student.latency = v

    def reset(self, env):
        self.teacher.reset(env)
        self.student.reset(env)

    def observe(self, obs):
        self.teacher.observe(obs)
        self.student.observe(obs)

    def act(self, obs, env):
        s = self.student
        lw = s.prior_logw + np.array([f.log_evidence for f in s.filters])
        x = mixture_features(s.filters, lw, obs, env, s.projected)
        q = self.teacher.q_values(obs, env)
        self.data.append((x, q.astype(np.float32)))
        if s.net is None or self.rng.random() < self.beta:
            return int(np.argmax(q))
        with torch.no_grad():
            return int(torch.argmax(s.net(torch.from_numpy(x)[None])[0]))


def job(args):
    torch.set_num_threads(1)
    idx, it, sd_or_spec, beta, n_roll, attempts, latency = args
    boss = make_splits(0)["train"][idx]
    if isinstance(sd_or_spec, str):
        net = load_reflex(sd_or_spec)
    elif sd_or_spec is not None:
        net = make_net(); net.load_state_dict(sd_or_spec); net.eval()
    else:
        net = None
    sm = SlowMind(boss, theta=theta_star(SPREAD), n_particles=4, sweeps=3, seed=1000 * it + idx)
    X, Q, wins = [], [], []
    for a in range(attempts):
        seed = 70_000 + 1000 * it + 31 * idx + a
        pol = Privileged(boss, sm.particles, sm.weights, net, beta, n_roll, seed)
        r = fight(boss, pol, seed=seed, latency=latency)
        X += [d[0] for d in pol.data]
        Q += [d[1] for d in pol.data]
        wins.append(r["outcome"] == "win")
        sm.update(r["trace"])
    return np.stack(X), np.stack(Q), wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=None)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--teacher", type=int, default=48)
    ap.add_argument("--latency", type=int, default=4)
    ap.add_argument("--tag", default="priv")
    a = ap.parse_args()
    betas = [0.5, 0.25] + [0.0] * a.iters
    Xs, Qs, cur, log = [], [], a.init, []
    with Pool(4) as pool:
        for it in range(a.iters):
            t0 = time.time()
            out = pool.map(job, [(i, it, cur, betas[it], a.teacher, a.attempts, a.latency) for i in range(40)])
            Xs += [o[0] for o in out]
            Qs += [o[1] for o in out]
            wins = np.array([o[2] for o in out], float)
            X, Q = np.concatenate(Xs), np.concatenate(Qs)
            sd, agree, regret = fit(X, Q, seed=it)
            torch.save(sd, f"checkpoints/{a.tag}_it{it}.pt")
            cur = sd
            rec = dict(iter=it, beta=betas[it], n=len(X), win_by_attempt=wins.mean(0).tolist(), agree=agree,
                       regret=regret, minutes=(time.time() - t0) / 60)
            log.append(rec)
            print(json.dumps(rec), flush=True)
            json.dump(log, open(f"results/{a.tag}_dagger.json", "w"), indent=1)


if __name__ == "__main__":
    main()
