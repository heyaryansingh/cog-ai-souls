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
from rtb.learner import RulesBelief, SlowMind, evidence             # noqa: E402
from rtb.policies import MixturePlanner, OnlineMixturePlanner                # noqa: E402
from rtb.run import fight                              # noqa: E402


def load_prior(prior: str):
    """'star' = the true hyperprior Theta*; 'vague' = same structure, flat timing prior;
    a path = an empirical-Bayes hyperprior fitted from learned training programs (scripts/fit_prior.py)."""
    from rtb.boss import theta_star, SPREAD
    th = theta_star(SPREAD)
    if prior == "star":
        return th
    if prior == "vague":
        th.log_median = {n: float(np.log(15.0)) for n in th.log_median}
        th.log_median_sd = {n: 1.0 for n in th.log_median}
        return th
    d = json.load(open(prior))
    th.log_median = {n: float(np.log(15.0)) for n in th.log_median}     # unseen primitives stay vague
    th.log_median_sd = {n: 1.0 for n in th.log_median}
    th.log_median.update(d["log_median"])
    th.log_median_sd.update(d["log_median_sd"])
    return th


def run_boss(args):
    split, idx, attempts, latency, mode, n_particles, sweeps, n_rollouts, horizon, prior, n_test, self_model = args
    ss = int(os.environ.get("RTB_SPLIT_SEED", 0))      # fresh boss sets for pre-registered replications
    boss = make_splits(ss, n_test=n_test)[split][idx]
    theta = load_prior(prior)
    rb = RulesBelief(Rules())
    rules = T4_RULES if split == "T4" else Rules()
    sm = SlowMind(boss, theta=theta, n_particles=n_particles, sweeps=sweeps, seed=idx)
    out = []
    for a in range(attempts):
        seed = 5000 + 101 * idx + a + 100_000 * ss
        bosses, weights = ([boss], [1.0]) if mode == "oracle" else (sm.particles, sm.weights)
        pol = MixturePlanner(bosses, weights, mode="map" if mode in ("oracle", "frozen", "map") else "mixture", n_rollouts=n_rollouts, horizon=horizon,
                             seed=seed)
        if mode in ("online", "online_netsup"):
            pol = OnlineMixturePlanner(sm, every=200, n_rollouts=n_rollouts, horizon=horizon, seed=seed)
        if os.environ.get("RTB_REFLEX") and a < int(os.environ.get("RTB_REFLEX_ATTEMPTS", 99)):   # Entry 28: act with the distilled reflex over the mixture belief
            import torch
            from rtb.reflex import MixtureReflexPolicy, load_reflex
            torch.set_num_threads(1)
            net = load_reflex(os.environ["RTB_REFLEX"])
            pol = MixtureReflexPolicy(bosses, weights, net=net, gate=float(os.environ.get("RTB_GATE", 0)),
                                      surprise=int(os.environ.get("RTB_SURPRISE", 0)),
                                      blend=float(os.environ.get("RTB_BLEND", 0)),
                                      mode="map" if mode in ("oracle", "frozen", "map") else "mixture",
                                      n_rollouts=n_rollouts, horizon=horizon, seed=seed)
        if self_model != "true":           # believed body rules (T4); "true" = agent knows its body
            pol.believed_rules = rb.current
        if self_model == "active":         # learned self-model + information-seeking rolls
            pol.rules_belief, pol.info_weight = rb, 15.0
        r = fight(boss, pol, rules=rules, seed=seed, latency=latency)
        if self_model in ("learned", "proprio", "iframes", "active"):     # ablations learn one part each
            rb.update(r["events"], true_rules=rules if self_model != "iframes" else None,
                      learn_iframes=self_model != "proprio")
        t0 = time.time()
        if mode in ("net", "netmix", "netsup", "online_netsup") and a == 0:   # M3: seed the slow mind with the intuition net's proposal
            import torch
            from rtb.intuition import IntuitionNet, propose
            torch.set_num_threads(1)
            net = IntuitionNet(); net.load_state_dict(torch.load("checkpoints/intuition.pt")); net.eval()
            nb = propose(net, r["trace"], boss, theta)
            if mode in ("netsup", "online_netsup"):         # superset proposal: the net's moves + every library move it left out
                from rtb.learner import _add, initial_belief
                have = {m.primitive.name for m in nb.moves}
                for m in initial_belief(theta, boss.tell_ticks, boss.cue_noise, boss.dmg_scale).moves:
                    if m.primitive.name not in have:
                        nb = _add(nb, m)
            sm = SlowMind(boss, theta=theta, n_particles=n_particles, sweeps=sweeps, seed=idx,
                          init=nb if mode in ("net", "netsup", "online_netsup") else [nb, None])
            sm.update(r["trace"], weigh_after_move=mode == "netmix")
        elif mode not in ("oracle", "frozen"):      # control conditions: no learning
            sm.update(r["trace"])
        out.append({"boss": idx, "attempt": a + 1, "outcome": r["outcome"], "boss_hp_frac": r["boss_hp_frac"],
                    "hp": r["hp"], "ticks": r["ticks"], "fight_s": r["seconds"], "learn_s": time.time() - t0,
                    "K_map": sm.map_particle().K, "reflex_frac": getattr(pol, "reflexed", 0) / max(getattr(pol, "decisions", 1), 1), "iframes": [rb.current.iframe_start, rb.current.iframe_len], "true_iframes": [rules.iframe_start, rules.iframe_len], "K_true": boss.K,
                    "accepted": sm.history[-1]["accepted"] if sm.history else []})
    mp = sm.map_particle()
    out[-1]["map_describe"] = mp.describe()
    out[-1]["map_moves"] = [(m.primitive.name, m.median) for m in mp.moves]
    out[-1]["map_laws"] = [list(l) for l in mp.laws]
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
    ap.add_argument("--prior", default="star")
    ap.add_argument("--n_test", type=int, default=10)
    ap.add_argument("--self_model", default="true", help="true | train | learned")
    a = ap.parse_args()
    jobs = [(a.split, i, a.attempts, a.latency, a.mode, a.particles, a.sweeps, a.rollouts, a.horizon, a.prior, a.n_test, a.self_model)
            for i in range(a.bosses)]
    t0 = time.time()
    with Pool(a.procs) as pool:
        res = pool.map(run_boss, jobs)
    flat = [r for boss in res for r in boss]
    os.makedirs("results", exist_ok=True)
    pr = "" if a.prior == "star" else "_" + os.path.basename(a.prior).replace(".json", "")
    pr += "" if a.self_model == "true" else f"_self-{a.self_model}"
    pr += "_reflex" if os.environ.get("RTB_REFLEX") else ""
    pr += f"_ss{os.environ['RTB_SPLIT_SEED']}" if os.environ.get("RTB_SPLIT_SEED") else ""
    pr += f"_surp{os.environ['RTB_SURPRISE']}" if os.environ.get("RTB_SURPRISE") else ""
    pr += f"_ra{os.environ['RTB_REFLEX_ATTEMPTS']}" if os.environ.get("RTB_REFLEX_ATTEMPTS") else ""
    pr += f"_blend{os.environ['RTB_BLEND']}" if os.environ.get("RTB_BLEND") else ""
    name = f"results/learn_{a.split}_{a.mode}{pr}_lat{a.latency}_b{a.bosses}_a{a.attempts}{a.tag}.json"
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
