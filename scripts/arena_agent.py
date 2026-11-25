"""Bayesian agent v1 on Arena: build the syllable library and cross-boss prior on training bosses, then
learning curves on held-out bosses. python scripts/arena_agent.py [stage] (stage: lib | prior | test | all)"""
import os, sys, json, pickle, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from multiprocessing import Pool
from arena.env import Arena
from arena.moves import sample_boss
from arena.perception import boss_centric
from arena.scripted import CheatDodger
from arena.agent import BayesAgent, BossMemory, SyllableLibrary, fit_library, merge_prior, run_fight

ARCHS = ["knight", "beast", "caster", "giant"]
LAT = 6
def train_bosses(n=6): return [sample_boss(np.random.default_rng(100 + i + 50 * j), a) for j, a in enumerate(ARCHS) for i in range(n)]
def test_bosses(n=6): return [sample_boss(np.random.default_rng(9000 + i + 50 * j), a) for j, a in enumerate(ARCHS) for i in range(n)]

def record(args):
    boss, seed = args
    env = Arena(boss, seed=seed); obs = env.reset(); P = CheatDodger(); X = []; done = False
    while not done and env.t < 2400:
        a = P(obs, env) if env.player_free() else 0
        obs, r, done, _ = env.step(a); X.append(obs[:51])
    return boss_centric(np.array(X))

def prior_job(args):
    boss, i = args
    lib = SyllableLibrary.load("checkpoints/arena_syllables.pkl")
    mem = BossMemory(lib.K)
    for a in range(3):
        run_fight(Arena(boss, seed=500 + 10 * i + a, latency=LAT), BayesAgent(lib, mem, latency=LAT, seed=a))
    return mem

def test_job(args):
    boss, i, learn, attempts, use_prior = args
    lib = SyllableLibrary.load("checkpoints/arena_syllables.pkl")
    prior = pickle.load(open("checkpoints/arena_prior.pkl", "rb"))
    mem = prior.copy_scaled(1.0) if use_prior else BossMemory(lib.K)
    out = []
    for a in range(attempts):
        t0 = time.time()
        o, T, bhp = run_fight(Arena(boss, seed=7000 + 13 * i + a, latency=LAT), BayesAgent(lib, mem, latency=LAT, learn=learn, seed=a))
        out.append(dict(boss=i, arch=boss.archetype, attempt=a + 1, outcome=o, ticks=T, boss_hp=bhp, s=time.time() - t0))
    return out

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    procs = int(os.environ.get("PROCS", 2))
    os.makedirs("checkpoints", exist_ok=True)
    if stage in ("lib", "all"):
        t0 = time.time()
        with Pool(procs) as pool:
            seqs = pool.map(record, [(b, s) for b in train_bosses() for s in range(2)])
        lib = fit_library(seqs, K=40)
        lib.save("checkpoints/arena_syllables.pkl")
        print(f"library fitted on {sum(len(s) for s in seqs)} frames in {time.time()-t0:.0f}s", flush=True)
    if stage in ("prior", "all"):
        with Pool(procs) as pool:
            mems = pool.map(prior_job, [(b, i) for i, b in enumerate(train_bosses())])
        pickle.dump(merge_prior(mems), open("checkpoints/arena_prior.pkl", "wb"))
        print("prior built", flush=True)
    if stage in ("test", "all"):
        res = {}
        for cond, learn, up in [("learn", True, True), ("frozen_prior", False, True), ("learn_noprior", True, False)]:
            with Pool(procs) as pool:
                out = pool.map(test_job, [(b, i, learn, 6, up) for i, b in enumerate(test_bosses())])
            flat = [r for o in out for r in o]
            res[cond] = flat
            curve = [np.mean([r["outcome"] == "win" for r in flat if r["attempt"] == k]) for k in range(1, 7)]
            print(cond, "win by attempt", [round(c, 2) for c in curve], "boss-hp-left", round(np.mean([r["boss_hp"] for r in flat]), 2),
                  "s/fight", round(np.mean([r["s"] for r in flat]), 1), flush=True)
        json.dump(res, open("results/arena_agent_v1.json", "w"))
