"""Paper figures from results/*.json -> docs/figures/*.png"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtb.evaluate import wilson

TXT, TXT2, GRID = "#191919", "#666666", "#e6e6e6"
S1, S2, S3, GRAY = "#257de0", "#f47230", "#00b894", "#9a9a9a"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TXT2, "xtick.color": TXT2,
                     "ytick.color": TXT2, "axes.spines.top": False, "axes.spines.right": False})


BINS = [(1, 1), (2, 3), (4, 6), (7, 9), (10, 12)]
LABELS = ["1", "2–3", "4–6", "7–9", "10–12"]


def curve(f):
    d = json.load(open(f))["results"]
    ys, lo, hi = [], [], []
    for a, b in BINS:
        rs = [r for r in d if a <= r["attempt"] <= b]
        w = sum(r["outcome"] == "win" for r in rs); l, h = wilson(w, len(rs))
        ys.append(w / len(rs)); lo.append(l); hi.append(h)
    return np.arange(len(BINS)), np.array(ys), np.array(lo), np.array(hi)


fig, ax = plt.subplots(figsize=(7, 4.6), dpi=200)
series = [("Planner given the true program", "results/learn_train_oracle_lat4_b40_a12.json", TXT2, "--"),
          ("Slow mind + within-fight learning", "results/learn_train_online_lat4_b40_a12.json", S2, "-"),
          ("Slow mind (between deaths)", "results/learn_train_mixture_lat4_b40_a12.json", S1, "-"),
          ("Generic prior, no learning", "results/learn_train_frozen_lat4_b40_a12.json", GRAY, "-")]
for name, f, c, ls in series:
    x, y, lo, hi = curve(f)
    if ls == "-":
        ax.fill_between(x, lo, hi, color=c, alpha=0.10, lw=0)
    ax.plot(x, y, color=c, lw=2, ls=ls, marker="o", ms=4, label=name)
ax.axhline(0.133, color=S3, lw=1.5, ls=":", label="RL² GRU-PPO (29k training fights), overall")
ax.set_xlabel("Attempts on the same boss (pooled)"); ax.set_ylabel("Win rate (40 bosses, 95% CI)")
ax.set_ylim(0, 0.6); ax.set_xticks(range(len(BINS))); ax.set_xticklabels(LABELS); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.grid(axis="y", color=GRID, lw=0.8); ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, labelcolor=TXT)
ax.set_title("Learning a boss from deaths (latency 200 ms)", color=TXT, loc="left", fontsize=11)
fig.tight_layout(); fig.savefig("docs/figures/learning_curves.png")

fig, ax = plt.subplots(figsize=(7, 3.4), dpi=200)
splits = ["train\n(same dist.)", "T2\nnew combos", "T3\nnovel move", "T4\nnew body rules"]
frozen = [12.8, 11.7, 13.3, 20.0]; slow = [37.2, 25.8, 71.7, 48.3]; orac = [42.5, 34.2, 80.0, 40.0]
x = np.arange(len(splits)); w = 0.26
for off, vals, c, lab in ((-w, frozen, GRAY, "No learning"), (0, slow, S1, "Slow mind"), (w, orac, TXT2, "True program / true body")):
    bars = ax.bar(x + off, vals, w - 0.02, color=c, label=lab)
for xi, v in zip(x, slow):
    ax.text(xi, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8, color=TXT)
ax.set_xticks(x); ax.set_xticklabels(splits); ax.set_ylabel("Late-attempt win rate (%)")
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_ylim(0, 95); ax.legend(frameon=False, fontsize=8, labelcolor=TXT, loc="upper left", ncol=3)
ax.set_title("Generalization to held-out bosses and rules", color=TXT, loc="left", fontsize=11)
fig.tight_layout(); fig.savefig("docs/figures/generalization.png")
print("ok")
