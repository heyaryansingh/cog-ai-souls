"""Margit-like grounded boss: build from held-out mocap, run controls (aggro, state oracle, clairvoyant)."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from multiprocessing import Pool
sys.argv = sys.argv[:2] + ["none"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "arena3d_baselines.py")).read().split("if __name__")[0])
from arena3d.program import build_program_boss

SPEC = None


def spec():
    global SPEC
    if SPEC is None:
        SPEC = build_program_boss(np.random.default_rng(42), U, TEST, clips())
    return SPEC


def one(args):
    pol, k = args
    env = Arena3D(spec(), U, clips(), seed=k, player_strike=pstrike())
    env.reset(); rng = np.random.default_rng(k)
    f = getattr(S, pol); done = False; ev = {"hit": 0, "dodged": 0, "blocked": 0, "proj": 0, "ring": 0}
    while not done:
        _, _, done, info = env.step(f(env, rng))
        if info["hit"]:
            ev["hit"] += 1; ev["proj"] += info["hit"][0] == "projectile"; ev["ring"] += info["hit"][0] == "shockwave"
        ev["dodged"] += bool(info["dodged"]); ev["blocked"] += bool(info["blocked"])
    print("FIGHT", pol, k, info["outcome"], round(env.boss_hp / env.spec.hp, 3), env.t, flush=True)
    return pol, info["outcome"], env.boss_hp / env.spec.hp, env.t, ev


if __name__ == "__main__":
    pols = os.environ.get("POLS", "aggro,state_oracle,oracle_dodger").split(",")
    n = int(os.environ.get("N", 8))
    with Pool(int(os.environ.get("PROCS", 3))) as p:
        res = p.map(one, [(q, k) for q in pols for k in range(int(os.environ.get("SEED0", 0)), int(os.environ.get("SEED0", 0)) + n)])
    for q in pols:
        r = [x for x in res if x[0] == q]
        print(q, "win", np.mean([x[1] == "win" for x in r]).round(2), "boss hp left", np.mean([x[2] for x in r]).round(2),
              "mean secs", round(np.mean([x[3] for x in r]) * 0.03), "events", {k: sum(x[4][k] for x in r) for k in r[0][4]}, flush=True)
