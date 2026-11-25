"""P1 analysis as pre-registered: per-boss total wins, one-sided Wilcoxon (boss-clustered), Holm over H1-H5,
bootstrap 95% CI of the mean per-boss win-rate difference (10,000 boss resamples)."""
import json, numpy as np
from scipy.stats import wilcoxon

F = {("T1", "a"): "learn_T1_mixture_ss1_lat4_b80_a3_P1.json",
     ("T1", "b"): "learn_T1_mixture_ss1_lat4_b80_a3_P1_r48.json",
     ("T1", "c"): "learn_T1_mixture_reflex_ss1_lat4_b80_a3_P1.json",
     ("T1", "d"): "learn_T1_mixture_reflex_ss1_blend3_lat4_b80_a3_P1.json",
     ("T3", "a"): "learn_T3_netsup_ss1_lat4_b80_a4_P1.json",
     ("T3", "b"): "learn_T3_netsup_ss1_lat4_b80_a4_P1_r48.json",
     ("T3", "c"): "learn_T3_netsup_reflex_ss1_lat4_b80_a4_P1.json",
     ("T3", "d"): "learn_T3_netsup_reflex_ss1_blend3_lat4_b80_a4_P1.json",
     ("T3", "e"): "learn_T3_netsup_reflex_ss1_ra1_lat4_b80_a4_P1.json"}


def load(k):
    r = json.load(open("results/" + F[k]))["results"]
    nb = max(x["boss"] for x in r) + 1
    na = max(x["attempt"] for x in r)
    W = np.zeros((nb, na))
    for x in r:
        W[x["boss"], x["attempt"] - 1] = x["outcome"] == "win"
    return W


D = {k: load(k) for k in F}
out = {"per_attempt": {f"{s}-{c}": D[(s, c)].mean(0).round(3).tolist() for (s, c) in D},
       "overall": {f"{s}-{c}": float(D[(s, c)].mean()) for (s, c) in D}}
H = [("H1", "T1", "d", "a"), ("H2", "T3", "d", "a"), ("H3", "T3", "d", "b"), ("H4", "T1", "c", "b"), ("H5", "T3", "e", "a")]
rng = np.random.default_rng(0)
rows = []
for h, s, x, y in H:
    a, b = D[(s, x)].sum(1), D[(s, y)].sum(1)
    diff = (a - b) / D[(s, x)].shape[1]
    try:
        p = wilcoxon(a, b, alternative="greater", zero_method="wilcox").pvalue
    except ValueError:
        p = 1.0
    bs = [diff[rng.integers(len(diff), size=len(diff))].mean() for _ in range(10000)]
    rows.append({"h": h, "split": s, "cmp": f"{x}>{y}", "mean_diff": float(diff.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p": float(p)})
# Holm
order = np.argsort([r["p"] for r in rows])
m = len(rows)
running = 0
for rank, i in enumerate(order):
    adj = min(1.0, max(running, (m - rank) * rows[i]["p"]))
    running = adj
    rows[i]["p_holm"] = adj
    rows[i]["supported"] = adj < 0.05
out["tests"] = rows
json.dump(out, open("results/P1_analysis.json", "w"), indent=1)
for k, v in out["per_attempt"].items():
    print(k, v, round(out["overall"][k], 3))
for r in rows:
    print(r)
