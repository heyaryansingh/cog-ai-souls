"""Reporting with the rigor rules adopted at B77: wins are the headline, damage is capped at 100%, central
tendency is the interquartile mean (IQM) with 95% stratified-bootstrap CIs (strata = series for hand-authored
bosses, boss for the procedural suite), and families of comparisons are Holm-corrected.

usage: python scripts/stats.py results/log_godrick_mirror_plan2.txt [more logs or gl6h_*.json ...]
       python scripts/stats.py --vs BASE.txt A.txt B.txt ...   (paired tests of each against BASE, Holm-corrected)
"""
import json
import re
import sys

import numpy as np
from scipy.stats import wilcoxon


def load(path):
    """-> list of (stratum, key, outcome, damage_capped, frames)"""
    out = []
    if path.endswith(".json"):
        for r in json.load(open(path)):
            out.append((r[1], (r[1], r[2]), r[3], min(1.0, float(r[4])), r[5]))
        return out
    for line in open(path):
        m = re.match(r"FIGHT \S+ (\d+) (\d+) (\w+) ([\d.]+) (\d+)", line)
        if m:
            out.append((int(m[1]), int(m[2]), m[3], min(1.0, float(m[4])), int(m[5])))
    return out


def iqm(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())


def boot(rows, stat, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    strata = {}
    for r in rows: strata.setdefault(r[0], []).append(r)
    vals = []
    for _ in range(B):
        s = []
        for v in strata.values():
            s += [v[i] for i in rng.integers(len(v), size=len(v))]
        vals.append(stat(s))
    return np.percentile(vals, [2.5, 97.5])


def summary(path):
    rows = load(path)
    win = lambda s: np.mean([r[2] == "win" for r in s])
    dmg = lambda s: iqm([r[3] for r in s])
    w, d = win(rows), dmg(rows)
    wl, wh = boot(rows, win); dl, dh = boot(rows, dmg)
    return (f"{path}: n={len(rows)}  wins {100 * w:.0f}% [{100 * wl:.0f}, {100 * wh:.0f}]  "
            f"damage IQM {100 * d:.0f}% [{100 * dl:.0f}, {100 * dh:.0f}]  mean {100 * np.mean([r[3] for r in rows]):.1f}%")


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1.0, run)
    return adj


def versus(base, others):
    B = {r[1]: r for r in load(base)}; ps, lines = [], []
    for o in others:
        O = {r[1]: r for r in load(o)}; k = sorted(set(B) & set(O))
        a = np.array([B[i][3] for i in k]); b = np.array([O[i][3] for i in k])
        p = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
        ps.append(p); lines.append((o, len(k), b.mean() - a.mean(), sum(O[i][2] == "win" for i in k) - sum(B[i][2] == "win" for i in k)))
    for (o, n, dd, dw), p, pa in zip(lines, ps, holm(np.array(ps))):
        print(f"{o} vs {base}: n={n} damage {100 * dd:+.1f} pts, wins {dw:+d}, p={p:.4f}, Holm p={pa:.4f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--vs":
        versus(a[1], a[2:])
    else:
        for p in a: print(summary(p))
