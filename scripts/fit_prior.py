"""Empirical Bayes: fit timing hyperparameters (per-primitive mean and sd of log wind-up median)
from the programs the slow mind learned on training bosses.
  python scripts/fit_prior.py results/learn_train_mixture_vague_....json results/prior_eb.json
"""
import json, re, sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
res = json.load(open(src))["results"]
by = {}
for r in res:
    if "map_moves" in r:
        moves = r["map_moves"]
    elif "map_describe" in r:
        moves = [(m[0], float(m[1])) for m in re.findall(r"\]\s+(\w+)\s+windup~\s*([\d.]+)t", r["map_describe"])]
    else:
        continue
    for name, med in moves:
        by.setdefault(name, []).append(np.log(med))
lm, sd, n = {}, {}, {}
for name, v in by.items():
    v = np.array(v)
    lm[name] = float(v.mean())
    # shrink the sd towards a weak default with few samples; floor it so the prior never becomes a spike
    k = len(v)
    s2 = (np.sum((v - v.mean()) ** 2) + 2 * 0.5 ** 2) / (k + 2 - 1)
    sd[name] = float(max(np.sqrt(s2), 0.1))
    n[name] = k
json.dump({"log_median": lm, "log_median_sd": sd, "n": n, "source": src}, open(dst, "w"), indent=1)
for name in lm:
    print(f"{name:13s} n={n[name]:3d}  median {np.exp(lm[name]):5.1f}t  sd {sd[name]:.2f}")
