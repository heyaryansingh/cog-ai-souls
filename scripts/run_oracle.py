"""M1 check: win rate of random, dodge-everything, and the oracle planner on training bosses."""
import sys, time, argparse
import numpy as np
sys.path.insert(0, ".")
from rtb.boss import make_splits, prior_mean_boss
from rtb.policies import RandomPolicy, DodgeEverything, OraclePlanner
from rtb.run import fight

ap = argparse.ArgumentParser()
ap.add_argument("--bosses", type=int, default=6)
ap.add_argument("--attempts", type=int, default=2)
ap.add_argument("--rollouts", type=int, default=12)
ap.add_argument("--horizon", type=int, default=30)
args = ap.parse_args()

bosses = make_splits(0)["train"][: args.bosses]
policies = {"random": lambda s: RandomPolicy(s), "dodge-all": lambda s: DodgeEverything(),
            "oracle": lambda s: OraclePlanner(args.rollouts, args.horizon, seed=s)}
policies_per_boss = {"prior-model": lambda b, s: OraclePlanner(args.rollouts, args.horizon, seed=s,
                                                               model_boss=prior_mean_boss(b))}
allp = {**{n: (lambda mk: lambda b, s: mk(s))(mk) for n, mk in policies.items()}, **policies_per_boss}
for name, mk in allp.items():
    res = [fight(b, mk(b, i * 10 + a), seed=1000 + i * 10 + a) for i, b in enumerate(bosses) for a in range(args.attempts)]
    wins = np.mean([r["outcome"] == "win" for r in res])
    print(f"{name:10s} win {wins:5.0%}  boss HP left {np.mean([r['boss_hp_frac'] for r in res]):.2f}  "
          f"player HP left {np.mean([r['hp'] for r in res]):5.1f}  outcomes {[r['outcome'][0] for r in res]}  "
          f"sec/fight {np.mean([r['seconds'] for r in res]):.1f}", flush=True)
