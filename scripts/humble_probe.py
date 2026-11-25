import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtb.evaluate import evaluate, summarize
for h in (False, True):
    for lat in (4, 0):
        r = evaluate("oracle", n_bosses=40, attempts=2, n_rollouts=12, horizon=45, latency=lat, humble=h)
        print(f"humble={h} latency={lat}: {summarize(r)}", flush=True)
