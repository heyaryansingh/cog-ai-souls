"""Acting ceiling on Arena: true-state planner vs scripted references. python scripts/arena_ceiling.py N_PER_ARCH"""
import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from multiprocessing import Pool
from arena.env import Arena
from arena.moves import sample_boss
from arena.planner import ArenaPlanner, continuation
from arena.scripted import CheatDodger

class Cont:
    def __init__(self): self.rng = np.random.default_rng(0)
    def __call__(self, obs, env): return continuation(env, self.rng)

def job(a):
    arch, i, agent = a
    boss = sample_boss(np.random.default_rng(10_000 + 97 * i + 1000 * ["knight", "beast", "caster", "giant"].index(arch)), arch)
    P = {"cheat": CheatDodger(), "cont": Cont(), "planner": ArenaPlanner(seed=i)}[agent]
    env = Arena(boss, seed=i); obs = env.reset(); done = False; t0 = time.time()
    while not done:
        act = P(obs, env) if env.player_free() else 0
        obs, r, done, info = env.step(act)
    return arch, agent, info["outcome"], env.t, env.boss.hp / boss.stats["hp"], time.time() - t0

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    jobs = [(a, i, ag) for ag in ("cheat", "cont", "planner") for a in ("knight", "beast", "caster", "giant") for i in range(n)]
    with Pool(int(os.environ.get("PROCS", 2))) as pool:
        out = pool.map(job, jobs)
    for ag in ("cheat", "cont", "planner"):
        rs = [o for o in out if o[1] == ag]
        print(ag, "win %.2f" % np.mean([o[2] == "win" for o in rs]), {a: round(np.mean([o[2] == "win" for o in rs if o[0] == a]), 2) for a in ("knight", "beast", "caster", "giant")},
              "bossHP %.2f" % np.mean([o[4] for o in rs]), "s/fight %.0f" % np.mean([o[5] for o in rs]), flush=True)
    json.dump(out, open("results/arena_ceiling.json", "w"))
