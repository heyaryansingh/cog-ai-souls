"""Entry 18: which part of the net's proposal hurts SMC out of distribution? Swap one component at a time."""
import os, sys, json
from multiprocessing import Pool
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from amortization_gap import trace, stretch
from rtb.boss import make_splits, Move, theta_star, SPREAD
from rtb.learner import SlowMind, evidence, initial_belief, make_boss
from rtb.intuition import IntuitionNet, propose


def job(args):
    split, i = args
    torch.set_num_threads(1)
    net = IntuitionNet(); net.load_state_dict(torch.load("checkpoints/intuition.pt")); net.eval()
    S = make_splits(0, n_test=30)
    boss = S["T1"][i] if split != "T3" else S["T3"][i]
    if split == "OOD":
        boss = stretch(boss)
    obs = trace(boss, 10 + i); test = [trace(boss, 900 + 7 * i + j) for j in range(3)]
    ticks = sum(len(t) for t in test); ev_true = evidence(boss, test)
    th = theta_star(SPREAD)
    prior = initial_belief(th, boss.tell_ticks, boss.cue_noise, boss.dmg_scale)
    netb = propose(net, obs, boss)
    net_med = {}
    for m in netb.moves:
        net_med.setdefault(m.primitive.name, m.median)
    # A: net structure, prior medians
    A = make_boss(netb, [Move(m.primitive, float(np.exp(th.log_median[m.primitive.name])), m.sigma) for m in netb.moves],
                  netb.pi, netb.phase2_speed)
    # B: prior structure, net medians where the net has that primitive
    B = make_boss(prior, [Move(m.primitive, net_med.get(m.primitive.name, m.median), m.sigma) for m in prior.moves],
                  prior.pi, prior.phase2_speed)
    have = {m.primitive.name for m in netb.moves}
    extra = [m for m in prior.moves if m.primitive.name not in have]
    from rtb.learner import _add
    Sup = netb
    for m in extra:
        Sup = _add(Sup, m)
    out = {"split": split}
    for name, init, sw in (("prior", None, 3), ("net", netb, 3), ("netStruct_priorMed", A, 3), ("priorStruct_netMed", B, 3), ("net_sweeps12", netb, 12), ("union", [netb, prior], 3), ("superset", Sup, 3)):
        sm = SlowMind(boss, n_particles=4, sweeps=sw, seed=i, init=init); sm.update(obs, weigh_after_move=name == "union")
        out[name] = (evidence(sm.map_particle(), test) - ev_true) / ticks
    return out


if __name__ == "__main__":
    with Pool(4) as p:
        res = p.map(job, [(s, i) for s in ("T1", "OOD", "T3") for i in range(30)])
    json.dump(res, open("results/swap_diag.json", "w"))
    for s in ("T1", "OOD", "T3"):
        rs = [r for r in res if r["split"] == s]
        print(s, "  ".join(f"{k} {np.mean([r[k] for r in rs]):+.3f} [{np.std([r[k] for r in rs])/np.sqrt(len(rs)):.3f}]"
                           for k in ("prior", "net", "netStruct_priorMed", "priorStruct_netMed", "net_sweeps12", "union", "superset")))
