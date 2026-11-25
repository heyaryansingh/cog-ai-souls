"""The ceiling ladder: cheat / oracle / prior-model / generic-model, at a given perception latency."""
import sys, argparse
sys.path.insert(0, ".")
from rtb.evaluate import evaluate, summarize
ap = argparse.ArgumentParser()
ap.add_argument("--latency", type=int, default=0)
ap.add_argument("--bosses", type=int, default=40)
ap.add_argument("--attempts", type=int, default=2)
a = ap.parse_args()
print(f"latency={a.latency}")
for agent in ["cheat", "oracle", "prior-model", "generic-model"]:
    kw = {} if agent == "cheat" else dict(n_rollouts=12, horizon=45, latency=a.latency)
    print(f"  {agent:14s} " + summarize(evaluate(agent, n_bosses=a.bosses, attempts=a.attempts, **kw)), flush=True)
