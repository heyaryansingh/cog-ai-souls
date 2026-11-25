"""E6: the amortization gap. After ONE observed fight, how good is the boss program from
  generic prior | intuition net alone | SMC from prior | SMC seeded by the net
measured by held-out log-evidence per tick (vs the true program) on 3 unseen fights, on
in-distribution (T1), novel-move (T3) and out-of-meta-prior (timings x1.8) bosses.
"""
import os, sys, time, json
from dataclasses import replace
from multiprocessing import Pool
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtb.boss import make_splits, Move
from rtb.env import MiniSouls
from rtb.policies import oracle_continuation
from rtb.learner import SlowMind, evidence, initial_belief
from rtb.intuition import IntuitionNet, propose
from rtb.boss import theta_star, SPREAD


def trace(boss, seed):
    env = MiniSouls(boss, seed=seed); obs = env.reset(); tr = []
    rng = np.random.default_rng(seed)
    while True:
        a = (int(rng.integers(8)) if rng.random() < 0.2 else oracle_continuation(env)) if obs["free"] else 0
        obs, _, done, _ = env.step(a); tr.append((obs["cue"], obs["dist"], obs["phase"], obs["ctx"]))
        if done: return tr


def stretch(boss, f=1.8):
    b = replace(boss, moves=[Move(m.primitive, m.median * (f if k else 1.0), m.sigma) for k, m in enumerate(boss.moves)])
    b._pmf = None; b._trans = None
    return b


def job(args):
    split, i = args
    torch.set_num_threads(1)
    net = IntuitionNet(); net.load_state_dict(torch.load("checkpoints/intuition.pt")); net.eval()
    S = make_splits(0, n_test=30)
    boss = S["T1" if split == "OOD" else split][i]
    if split == "OOD":
        boss = stretch(boss)
    obs = [trace(boss, 10 + i)]
    test = [trace(boss, 900 + 7 * i + j) for j in range(3)]
    ticks = sum(len(t) for t in test)
    ev_true = evidence(boss, test)
    th = theta_star(SPREAD)
    out = {"split": split, "boss": i}
    t = time.time(); prior = initial_belief(th, boss.tell_ticks, boss.cue_noise, boss.dmg_scale)
    out["prior"] = (evidence(prior, test) - ev_true) / ticks
    t = time.time(); netb = propose(net, obs[0], boss); out["t_net"] = time.time() - t
    out["net"] = (evidence(netb, test) - ev_true) / ticks
    for name, init in (("smc", None), ("net+smc", netb), ("mix+smc", [netb, None]), ("mix2+smc", [netb, None])):
        t = time.time(); sm = SlowMind(boss, n_particles=4, sweeps=3, seed=i, init=init)
        sm.update(obs[0], weigh_after_move=name == "mix2+smc")
        out["t_" + name] = time.time() - t
        out[name] = (evidence(sm.map_particle(), test) - ev_true) / ticks
    return out


if __name__ == "__main__":
    jobs = [(s, i) for s in ("T1", "T3", "OOD") for i in range(30)]
    with Pool(4) as p:
        res = p.map(job, jobs)
    json.dump(res, open("results/amortization_gap.json", "w"), indent=1)
    print("held-out log-evidence per tick relative to the true program (0 = as good as truth); mean [sem]")
    for s in ("T1", "T3", "OOD"):
        rs = [r for r in res if r["split"] == s]
        row = "  ".join(f"{k} {np.mean([r[k] for r in rs]):+.3f} [{np.std([r[k] for r in rs])/np.sqrt(len(rs)):.3f}]"
                        for k in ("prior", "net", "smc", "net+smc", "mix+smc", "mix2+smc"))
        print(f"{s:4s} {row}   time net {np.mean([r['t_net'] for r in rs]):.3f}s smc {np.mean([r['t_smc'] for r in rs]):.2f}s")
