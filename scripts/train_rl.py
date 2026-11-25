"""Train the RL^2 GRU-PPO baseline, then evaluate it on the learn_curve protocol (40 train bosses x 12 attempts).
  python scripts/train_rl.py --updates 1500 --sampler theta
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtb.boss import make_splits
from rtb.evaluate import wilson
from rtb.rl import ppo_train, evaluate_policy

ap = argparse.ArgumentParser()
ap.add_argument("--updates", type=int, default=1500)
ap.add_argument("--sampler", default="theta")
ap.add_argument("--fights", type=int, default=4)
ap.add_argument("--latency", type=int, default=4)
ap.add_argument("--attempts", type=int, default=12)
a = ap.parse_args()
os.makedirs("checkpoints", exist_ok=True); os.makedirs("results", exist_ok=True)
tag = f"rl2_{a.sampler}_u{a.updates}_f{a.fights}"
log = open(f"results/{tag}.log", "w")
def p(s):
    print(s); log.write(s + "\n"); log.flush()
t0 = time.time()
pol = ppo_train(updates=a.updates, n_envs=16, steps_per_update=256, fights_per_trial=a.fights,
                latency=a.latency, sampler=a.sampler, log=p)
torch.save(pol.state_dict(), f"checkpoints/{tag}.pt")
p(f"trained in {time.time()-t0:.0f}s")
bosses = make_splits(0)["train"]
for greedy in (False, True):
    outs = evaluate_policy(pol, bosses, attempts=a.attempts, latency=a.latency, seed=5000, greedy=greedy)
    p(f"greedy={greedy}")
    for k in range(a.attempts):
        w = sum(o[k] == "win" for o in outs); lo, hi = wilson(w, len(outs))
        p(f"  attempt {k+1:2d}: {w/len(outs):.1%} [{lo:.0%}-{hi:.0%}]")
    json.dump(outs, open(f"results/{tag}_greedy{int(greedy)}.json", "w"))
