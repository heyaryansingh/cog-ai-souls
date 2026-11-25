"""Paired evaluation of distilled reflexes vs planners (Entry 28).
python scripts/reflex_eval.py SPLIT N_BOSSES SEEDS ckpt1 [ckpt2 ...] [--planners 12,48] [--gate g]
All agents use the TRUE boss program (acting-ceiling setting), latency 4, the same fight seeds."""
import json, os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from scipy.stats import binomtest
from rtb.boss import make_splits
from rtb.evaluate import wilson
from rtb.policies import OraclePlanner
from rtb.reflex import ReflexPolicy, load_reflex
from rtb.run import fight

def job(a):
    torch.set_num_threads(1)
    split, n_test, idx, seed, kind, arg, gate = a
    boss = make_splits(0, n_test=n_test)[split][idx]
    if kind == "planner":
        pol = OraclePlanner(n_rollouts=arg, seed=seed)
    else:
        net = load_reflex(arg)
        pol = ReflexPolicy(net=net, gate=gate, seed=seed, n_rollouts=48)   # gate falls back to the teacher
    r = fight(boss, pol, seed=seed, latency=4)
    return r["outcome"] == "win", r["seconds"], getattr(pol, "planned", 0) / max(getattr(pol, "decisions", 1), 1)

if __name__ == "__main__":
    args = sys.argv[1:]
    planners = [12, 48]; gate = 0.0
    if "--planners" in args:
        i = args.index("--planners"); planners = [int(x) for x in args[i + 1].split(",") if x]; del args[i:i + 2]
    if "--gate" in args:
        i = args.index("--gate"); gate = float(args[i + 1]); del args[i:i + 2]
    split, nb, ns, ckpts = args[0], int(args[1]), int(args[2]), args[3:]
    n_test = max(10, nb) if split != "train" else 10
    agents = [("planner", n) for n in planners] + [("reflex", c) for c in ckpts]
    pairs = [(i, 9000 + 53 * i + s) for i in range(nb) for s in range(ns)]
    res = {}
    with Pool(4) as pool:
        for kind, arg in agents:
            out = pool.map(job, [(split, n_test, i, sd, kind, arg, gate) for i, sd in pairs])
            name = f"{kind}:{arg}" + (f"@gate{gate}" if kind == "reflex" and gate else "")
            res[name] = out
            w = [o[0] for o in out]; lo, hi = wilson(sum(w), len(w))
            print(f"{name:40s} win {np.mean(w):5.1%} [{lo:.0%}-{hi:.0%}]  s/fight {np.mean([o[1] for o in out]):.2f}  planned {np.mean([o[2] for o in out]):.2f}  n={len(w)}", flush=True)
    names = list(res)
    base = names[0]
    for n in names[1:]:
        a = sum(x[0] and not y[0] for x, y in zip(res[n], res[base])); b = sum(y[0] and not x[0] for x, y in zip(res[n], res[base]))
        print(f"  {n} vs {base}: only-{n.split(':')[0]} {a}, only-base {b}, sign p={binomtest(a, a + b).pvalue if a + b else 1:.4f}")
    tag = os.environ.get("TAG", "")
    json.dump({k: [list(map(float, o)) for o in v] for k, v in res.items()}, open(f"results/reflex_eval_{split}_b{nb}_s{ns}{tag}.json", "w"))
