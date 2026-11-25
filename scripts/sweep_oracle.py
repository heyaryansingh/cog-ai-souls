"""M1: how strong is the true-model planner as a function of rollouts and horizon?"""
import sys
sys.path.insert(0, ".")
from rtb.evaluate import evaluate, summarize
for n, h in [(12, 30), (24, 30), (12, 45), (24, 45)]:
    print(f"rollouts={n:2d} horizon={h}: " + summarize(evaluate("oracle", n_bosses=12, attempts=2, n_rollouts=n, horizon=h)), flush=True)
