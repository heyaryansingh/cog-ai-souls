"""How much is boss-specific knowledge worth? Compare planners that believe in:
  oracle   - the true boss program
  prior    - the right move list, but typical timings and uniform transitions
  generic  - only Theta: one move per primitive, typical timings, uniform transitions
across difficulty settings. A good benchmark needs oracle >> generic (room for learning)."""
import sys, argparse, itertools
import numpy as np
sys.path.insert(0, ".")
from rtb.boss import make_splits, prior_mean_boss, generic_boss, theta_star
from rtb.policies import OraclePlanner
from rtb.run import fight
from rtb.env import Rules

ap = argparse.ArgumentParser()
ap.add_argument("--bosses", type=int, default=8)
ap.add_argument("--spreads", default="0.35")
ap.add_argument("--tells", default="6")
ap.add_argument("--noise", type=float, default=0.03)
ap.add_argument("--dmg", type=float, default=1.4)
ap.add_argument("--iframes", type=int, default=4)
ap.add_argument("--heals", type=int, default=2)
args = ap.parse_args()
rules = Rules(iframe_len=args.iframes, heal_charges=args.heals)
print(f"dmg x{args.dmg}, i-frames {args.iframes}, heals {args.heals}, cue noise {args.noise}")
for spread, tell in itertools.product(map(float, args.spreads.split(",")), map(int, args.tells.split(","))):
    bosses = make_splits(0, spread=spread, tell_ticks=tell, cue_noise=args.noise, dmg_scale=args.dmg)["train"][: args.bosses]
    row = []
    for name in ["oracle", "prior", "generic"]:
        res = []
        for i, b in enumerate(bosses):
            model = {"oracle": None, "prior": prior_mean_boss(b, theta_star(spread)),
                     "generic": generic_boss(theta_star(spread), tell, args.noise, args.dmg)}[name]
            res.append(fight(b, OraclePlanner(12, 30, seed=i, model_boss=model), rules=rules, seed=500 + i))
        row.append(f"{name} {np.mean([r['outcome']=='win' for r in res]):4.0%} (hp {np.mean([r['hp'] for r in res]):4.0f})")
    print(f"spread={spread:.2f} tell={tell}:  " + "   ".join(row), flush=True)
