"""M2 headline: learning curves. Each boss is fought `attempts` times in a row; between
attempts the slow mind updates its particle set from the fight log. Results -> results/*.json.

  python scripts/learn_curve.py --bosses 20 --attempts 10 --latency 4 --mode mixture
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtb.boss import make_splits                       # noqa: E402
from rtb.env import Rules, T4_RULES                    # noqa: E402
from rtb.evaluate import wilson                        # noqa: E402
from rtb.learner import SlowMind, evidence             # noqa: E402
from rtb.policies import MixturePlanner                # noqa: E402
from rtb.run import fight                              # noqa: E402


def run_boss(args):
    split, idx, attempts, latency, mode, n_particles, sweeps, n_rollouts, horizon = args
    boss = make_splits(0)[split][idx]
    rules = T4_RULES if split == "T4" else Rules()
    sm = SlowMind(boss, n_particles=n_particles, sweeps=sweeps, seed=idx)
    out = []
    for a in range(attempts):
        seed = 5000 + 101 * idx + a
        bosses, weights = ([boss], [1.0]) if mode == "oracle" else (sm.particles, sm.weights)
        pol = MixturePlanner(bosses, weights, mode="map" if mode in ("oracle", "frozen") else mode, n_rollouts=n_rollouts, horizon=horizon,
                             seed=seed)
        r = fight(boss, pol, rules=rules, seed=seed, latency=latency)
        t0 = time.time()
        if mode not in ("oracle", "frozen"):      # control conditions: no learning
            sm.update(r["trace"])
        out.append({"boss": idx, "attempt": a + 1, "outcome": r["outcome"], "boss_hp_frac": r["boss_hp_frac"],
                    "hp": r["hp"], "ticks": r["ticks"], "fight_s": r["seconds"], "learn_s": time.time() - t0,
                    "K_map": sm.map_particle().K, "K_true": boss.K,
                    "accepted": sm.history[-1]["accepted"] if sm.history else []})
    out[-1]["map_describe"] = sm.map_particle().describe()
    out[-1]["true_describe"] = boss.describe()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--bosses", type=int, default=20)
    ap.add_argument("--attempts", type=int, default=10)
    ap.add_argument("--latency", type=int, default=4)
    ap.add_argument("--mode", default="mixture")
    ap.add_argument("--particles", type=int, default=4)
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--rollouts", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=45)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    jobs = [(a.split, i, a.attempts, a.latency, a.mode, a.particles, a.sweeps, a.rollouts, a.horizon)
            for i in range(a.bosses)]
    t0 = time.time()
    with Pool(a.procs) as pool:
        res = pool.map(run_boss, jobs)
    flat = [r for boss in res for r in boss]
    os.makedirs("results", exist_ok=True)
    name = f"results/learn_{a.split}_{a.mode}_lat{a.latency}_b{a.bosses}_a{a.attempts}{a.tag}.json"
    json.dump({"args": vars(a), "results": flat}, open(name, "w"), indent=1)
    print(f"saved {name}  ({time.time() - t0:.0f}s)")
    print("attempt  win%   [95% CI]     death%  boss-HP-left")
    for k in range(1, a.attempts + 1):
        rs = [r for r in flat if r["attempt"] == k]
        w = sum(r["outcome"] == "win" for r in rs)
        lo, hi = wilson(w, len(rs))
        print(f"{k:5d}   {w / len(rs):5.1%}  [{lo:.0%}-{hi:.0%}]  {np.mean([r['outcome'] == 'death' for r in rs]):5.1%}"
              f"   {np.mean([r['boss_hp_frac'] for r in rs]):.2f}")
    firsts = []
    for boss in res:
        wins = [r["attempt"] for r in boss if r["outcome"] == "win"]
        firsts.append(wins[0] - 1 if wins else None)
    solved = [f for f in firsts if f is not None]
    print(f"deaths-to-first-win: median {np.median(solved) if solved else 'n/a'} over {len(solved)}/{len(firsts)} bosses "
          f"that were beaten; never beaten: {firsts.count(None)}")
    late = [r for r in flat if r["attempt"] > a.attempts // 2]
    w = sum(r["outcome"] == "win" for r in late)
    lo, hi = wilson(w, len(late))
    print(f"second-half win rate: {w / len(late):.1%} [{lo:.0%}-{hi:.0%}] n={len(late)}")


if __name__ == "__main__":
    main()
