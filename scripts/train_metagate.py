"""R7: learn when to deliberate. Metacontrol from regret labels (Hamrick et al. 2017; Lee et al. 2014).

On training bosses, with the slow mind learning between fights (attempts 1-3), at every decision:
  a_plan  = the 12-rollout planner's choice over the learned mixture
  a_hab   = the reflex's choice over the learned mixture
  Q_true  = 48-rollout Q-values of the TRUE-program teacher
  label   = Q_true[a_plan] - Q_true[a_hab]   (the gain from deliberating here)
Inputs: belief features + meta features (odd-cue count, running log predictive, worst recent
predictive, reflex confidence, program-weight entropy, attempt). The executed action alternates at
random between a_plan and a_hab so both controllers' state distributions are covered.

python scripts/train_metagate.py --reflex A+B+C --out checkpoints/metagate.pt
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
import torch.nn as nn

from rtb.boss import make_splits, theta_star, SPREAD
from rtb.learner import SlowMind
from rtb.policies import OraclePlanner
from rtb.reflex import MixtureReflexPolicy, N_FEAT, load_reflex
from rtb.run import fight

N_META = 6


def make_gate():
    return nn.Sequential(nn.Linear(N_FEAT + N_META, 64), nn.ReLU(), nn.Linear(64, 1))


class Collector(MixtureReflexPolicy):
    def __init__(self, bosses, weights, net, teacher_boss, seed, **kw):
        super().__init__(bosses, weights, net=net, seed=seed, **kw)
        self.teacher = OraclePlanner(n_rollouts=48, seed=seed + 7, model_boss=teacher_boss)
        self.rows, self.coin = [], np.random.default_rng(seed + 3)

    def reset(self, env):
        super().reset(env)
        self.teacher.reset(env)

    def observe(self, obs):
        super().observe(obs)
        self.teacher.observe(obs)

    def act(self, obs, env):
        self.teacher.latency = self.latency
        x, p = self.reflex_probs(obs, env)
        a_hab = int(np.argmax(p))
        a_plan = self.actions[int(np.argmax(self.q_values(obs, env)))]
        qt = self.teacher.q_values(obs, env)
        self.rows.append((np.concatenate([x, self.meta_features(p)]), float(qt[a_plan] - qt[a_hab])))
        return a_plan if self.coin.random() < 0.5 else a_hab


def job(args):
    torch.set_num_threads(1)
    idx, spec, attempts, latency = args
    boss = make_splits(0)["train"][idx]
    net = load_reflex(spec)
    sm = SlowMind(boss, theta=theta_star(SPREAD), n_particles=4, sweeps=3, seed=500 + idx)
    rows = []
    for a in range(attempts):
        seed = 90_000 + 31 * idx + a
        pol = Collector(sm.particles, sm.weights, net, boss, seed, n_rollouts=12, horizon=45)
        pol.attempt = a
        r = fight(boss, pol, seed=seed, latency=latency)
        rows += pol.rows
        sm.update(r["trace"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reflex", required=True)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--latency", type=int, default=4)
    ap.add_argument("--out", default="checkpoints/metagate.pt")
    a = ap.parse_args()
    t0 = time.time()
    with Pool(4) as pool:
        out = pool.map(job, [(i, a.reflex, a.attempts, a.latency) for i in range(40)])
    rows = [r for o in out for r in o]
    X = torch.from_numpy(np.stack([r[0] for r in rows]))
    y = torch.tensor([r[1] for r in rows], dtype=torch.float32)[:, None]
    np.savez_compressed("results/metagate_data.npz", X=X.numpy(), y=y.numpy())
    # boss-held-out validation: last 8 bosses
    n_val = sum(len(o) for o in out[-8:])
    Xtr, ytr, Xva, yva = X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]
    torch.manual_seed(0)
    g = make_gate()
    opt = torch.optim.Adam(g.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(60):
        for b in torch.randperm(len(Xtr)).split(256):
            loss = nn.functional.huber_loss(g(Xtr[b]), ytr[b], delta=5.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pv = g(Xva)
    # value of the gate on held-out bosses: mean realised gain if we plan where predicted gain > c
    rep = {"n": len(rows), "frac_planner_better": float((y > 0).float().mean()), "frac_tie": float((y == 0).float().mean()),
           "mean_gain_always_plan": float(yva.mean())}
    for c in (0.0, 1.0, 3.0):
        m = pv[:, 0] > c
        rep[f"c{c}"] = {"plan_frac": float(m.float().mean()), "gain_captured": float((yva[:, 0] * m).mean()),
                        "oracle_gain": float(yva.clamp(min=0).mean())}
    rep["corr"] = float(np.corrcoef(pv[:, 0].numpy(), yva[:, 0].numpy())[0, 1])
    rep["minutes"] = (time.time() - t0) / 60
    torch.save(g.state_dict(), a.out)
    print(json.dumps(rep, indent=1))
    json.dump(rep, open("results/metagate_fit.json", "w"), indent=1)


if __name__ == "__main__":
    main()
