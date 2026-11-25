"""E32 realism check (review R3): 100%-physical shield (no chip) with guard-break stagger vs default rules.
True-program planner, 40 train bosses x 2 attempts, same seeds as E24/E27. Also counts hits while blocking."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtb.evaluate import evaluate, summarize
for rules in ("default", "shield"):
    if rules == "shield":
        os.environ["RTB_RULES"] = "shield"
    for lat in (4, 0):
        r = evaluate("oracle", n_bosses=40, attempts=2, n_rollouts=12, horizon=45, latency=lat)
        print(f"rules={rules} latency={lat}: {summarize(r)}", flush=True)
