import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from multiprocessing import Pool
from arena3d import library as L, mocap
from arena3d.boss import sample_boss
from arena3d.env import Arena3D
from arena3d import scripted as S

U = L.load("checkpoints/arena3d_units.pkl")
_, _, SUBS = L.split_subjects(U)
TEST = [i for i, u in enumerate(U) if u.subject in SUBS]
C = None


def pstrike():
    """Player's light attack: a fast, forward, chest-high swing with little travel (from training subjects)."""
    best = None
    for u in U:
        if u.subject in SUBS or u.kind != "strike" or not 18 <= u.length <= 40 or u.travel > 0.3:
            continue
        k = int(np.argmax(u.tip_speed)); p = u.tip_rel[k]
        if 0.7 < p[0] < 1.4 and abs(p[1]) < 0.35 and 0.8 < p[2] < 1.6 and k < 15:
            if best is None or u.peak_speed > best.peak_speed:
                best = u
    return best


def run(args):
    global C
    if C is None:
        C = mocap.load()
    pol, b = args
    spec = sample_boss(np.random.default_rng(5000 + b), U, TEST, name=f"test{b}")
    env = Arena3D(spec, U, C, seed=b, player_strike=pstrike())
    env.reset()
    rng = np.random.default_rng(b)
    f = getattr(S, pol)
    done, t0 = False, time.time()
    while not done:
        _, _, done, info = env.step(f(env, rng))
    return pol, b, info["outcome"], env.boss_hp / spec.hp, env.hp, env.t, time.time() - t0


if __name__ == "__main__":
    nb = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    pols = sys.argv[2].split(",") if len(sys.argv) > 2 else ["random_policy", "aggro", "oracle_dodger"]
    jobs = [(p, b) for p in pols for b in range(nb)]
    with Pool(int(os.environ.get("PROCS", 3))) as pool:
        res = pool.map(run, jobs)
    out = {}
    for p in pols:
        r = [x for x in res if x[0] == p]
        win = np.mean([x[2] == "win" for x in r])
        out[p] = {"win": win, "boss_hp_left": float(np.mean([x[3] for x in r])),
                  "outcomes": [x[2] for x in r], "sec_per_fight": float(np.mean([x[6] for x in r]))}
        print(p, "win", round(win, 2), "boss hp left", round(out[p]["boss_hp_left"], 2), out[p]["outcomes"],
              "s/fight", round(out[p]["sec_per_fight"], 1), flush=True)
    json.dump(out, open("results/arena3d_baselines.json", "w"), indent=1)
