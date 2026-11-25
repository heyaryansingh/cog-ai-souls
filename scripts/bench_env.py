"""Ticks per second of the raw simulator, and of simulator + exact filter."""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from rtb.boss import make_splits
from rtb.env import MiniSouls
from rtb.filter import BossFilter

boss = make_splits(0)["train"][0]
env = MiniSouls(boss, seed=0); rng = np.random.default_rng(0)
acts = rng.integers(0, 8, size=200000)
n, t0 = 0, time.time()
while n < 50000:
    _, _, done, _ = env.step(int(acts[n])); n += 1
    if done: env.reset()
print(f"env only:      {n/(time.time()-t0):,.0f} ticks/s")
env.reset(); f = BossFilter(boss); n, t0 = 0, time.time()
while n < 10000:
    obs, _, done, _ = env.step(int(acts[n])); f.step(obs); n += 1
    if done: env.reset(); f.reset()
print(f"env + filter:  {n/(time.time()-t0):,.0f} ticks/s   (state size {f.W.size + f.A.size + f.R.size})")
