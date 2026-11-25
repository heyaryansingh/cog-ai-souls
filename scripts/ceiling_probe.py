"""Does more planning compute raise the latency-4 oracle ceiling? 40 bosses x 2 attempts per row."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtb.evaluate import evaluate, summarize
for n, h in [(12, 45), (24, 45), (48, 45), (24, 90)]:
    r = evaluate("oracle", n_bosses=40, attempts=2, n_rollouts=n, horizon=h, latency=4)
    print(f"rollouts {n:3d} horizon {h:3d}: {summarize(r)}", flush=True)
