"""Generalisation suite: procedural program bosses (random moves, strings, branches, phase-2 switch) built from
held-out-subject mocap. Runs a list of policies on NB bosses x NF fights each (paired seeds)."""
import os, sys, json
sys.path.insert(0, "/home/user/cog-ai-souls"); os.chdir("/home/user/cog-ai-souls")
exec(open("scripts/arena3d_margit.py").read().split('if __name__ == "__main__"')[0].replace("os.path.dirname(os.path.abspath(__file__))", '"scripts"'))
from arena3d.program import random_program, build_program_boss
import arena3d.scripted as SC
BOSSES = {}


def boss(b):
    if b not in BOSSES:
        rng = np.random.default_rng(7000 + b)
        moves, strings = random_program(rng)
        BOSSES[b] = build_program_boss(rng, U, TEST, clips(), moves=moves, strings=strings,
                                       scale=float(np.exp(rng.uniform(np.log(1.4), np.log(2.4)))), name=f"proc{b}",
                                       weapon=str(rng.choice(["sword", "greatsword", "club"])))
    return BOSSES[b]


def one(args):
    pol, b, k = args
    sp = boss(b); SC._LGP = None
    env = Arena3D(sp, U, clips(), seed=100 * b + k, player_strike=pstrike()); env.reset()
    rng = np.random.default_rng(100 * b + k); f = getattr(SC, pol); done = False
    while not done:
        _, _, done, info = env.step(f(env, rng))
    r = (pol, b, k, info["outcome"], round(1 - env.boss_hp / sp.hp, 3), env.t)
    print("FIGHT", *r, flush=True)
    return r


if __name__ == "__main__":
    from multiprocessing import Pool
    pols = os.environ.get("POLS", "mpc_uniform,mpc_grammar").split(",")
    NB, NF = int(os.environ.get("NB", 6)), int(os.environ.get("NF", 2))
    with Pool(int(os.environ.get("PROCS", 3))) as p:
        B0 = int(os.environ.get("B0", 0))
        res = p.map(one, [(q, b, k) for b in range(B0, B0 + NB) for k in range(NF) for q in pols])
    json.dump(res, open(os.environ.get("OUT", "results/general_suite.json"), "w"))
    for q in pols:
        r = [x for x in res if x[0] == q]
        print(q, "wins", sum(x[3] == "win" for x in r), "/", len(r), "damage", np.mean([x[4] for x in r]).round(3))
