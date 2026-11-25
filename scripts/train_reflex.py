"""DAgger: distil the planner into a reflex over belief features (Entry 28).

python scripts/train_reflex.py --teacher 48 --iters 6 --fights 4
Teacher = OraclePlanner with the TRUE boss program, `--teacher` rollouts, latency 4, train bosses.
Each iteration: run fights with the current reflex (teacher action with prob beta), label every
decision with the teacher's Q-values, aggregate, refit; then evaluate the reflex alone on the
standard evaluation seeds (40 train bosses x 2 attempts), which no training fight uses.
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

from rtb.evaluate import _splits, wilson
from rtb.reflex import ReflexPolicy, make_net
from rtb.run import fight


def _net_from(sd):
    if sd is None:
        return None
    net = make_net()
    net.load_state_dict(sd)
    net.eval()
    return net


def collect_job(args):
    torch.set_num_threads(1)
    idx, seed, sd, beta, n_roll, latency, split = args
    boss = _splits(0)[split][idx]
    pol = ReflexPolicy(net=_net_from(sd), collect=True, beta=beta, n_rollouts=n_roll, seed=seed)
    r = fight(boss, pol, seed=seed, latency=latency)
    X = np.stack([d[0] for d in pol.data]) if pol.data else np.zeros((0, 79), np.float32)
    Q = np.stack([d[1] for d in pol.data]) if pol.data else np.zeros((0, 8), np.float32)
    return X, Q, r["outcome"]


def eval_job(args):
    torch.set_num_threads(1)
    idx, seed, sd, gate, n_roll, latency, split = args
    boss = _splits(0)[split][idx]
    pol = ReflexPolicy(net=_net_from(sd), gate=gate, n_rollouts=n_roll, seed=seed)
    r = fight(boss, pol, seed=seed, latency=latency)
    return r["outcome"], r["seconds"], pol.planned / max(pol.decisions, 1)


def fit(X, Q, sd=None, tau=2.0, epochs=40, seed=0):
    torch.manual_seed(seed)
    net = make_net()
    if sd is not None:
        net.load_state_dict(sd)
    Xt, Qt = torch.from_numpy(X), torch.from_numpy(Q)
    target = torch.softmax((Qt - Qt.max(1, keepdim=True).values) / tau, -1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    n = len(X)
    perm = torch.randperm(n)
    val, tr = perm[: n // 10], perm[n // 10:]
    for ep in range(epochs):
        net.train()
        for b in tr[torch.randperm(len(tr))].split(256):
            loss = -(target[b] * torch.log_softmax(net(Xt[b]), -1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pv = net(Xt[val]).argmax(1)
        agree = (pv == Qt[val].argmax(1)).float().mean().item()
        # regret: teacher Q of the reflex's choice vs the teacher's best
        regret = (Qt[val].max(1).values - Qt[val].gather(1, pv[:, None])[:, 0]).mean().item()
    return net.state_dict(), agree, regret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=int, default=48)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--fights", type=int, default=4, help="fights per train boss per iteration")
    ap.add_argument("--latency", type=int, default=4)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--tag", default="reflex")
    a = ap.parse_args()
    betas = [1.0, 0.5, 0.25] + [0.0] * max(0, a.iters - 3)
    Xs, Qs, sd, log = [], [], None, []
    os.makedirs("checkpoints", exist_ok=True)
    pool = Pool(a.procs)
    for it in range(a.iters):
        t0 = time.time()
        jobs = [(i, 50_000 + 1000 * it + 41 * i + s, sd, betas[it], a.teacher, a.latency, "train")
                for i in range(40) for s in range(a.fights)]
        out = pool.map(collect_job, jobs)
        Xs += [o[0] for o in out]
        Qs += [o[1] for o in out]
        roll_win = np.mean([o[2] == "win" for o in out])
        X, Q = np.concatenate(Xs), np.concatenate(Qs)
        sd, agree, regret = fit(X, Q)
        torch.save(sd, f"checkpoints/{a.tag}_it{it}.pt")
        ev = pool.map(eval_job, [(i, 1000 + 37 * i + s, sd, 0.0, a.teacher, a.latency, "train")
                                 for i in range(40) for s in range(2)])
        w = np.mean([e[0] == "win" for e in ev])
        lo, hi = wilson(int(round(w * len(ev))), len(ev))
        rec = dict(iter=it, beta=betas[it], n=len(X), rollout_win=roll_win, agree=agree, regret=regret,
                   reflex_win=w, ci=[lo, hi], s_fight=float(np.mean([e[1] for e in ev])), minutes=(time.time() - t0) / 60)
        log.append(rec)
        print(json.dumps(rec), flush=True)
        json.dump(log, open(f"results/{a.tag}_dagger.json", "w"), indent=1)


if __name__ == "__main__":
    main()
