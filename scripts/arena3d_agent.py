"""Arena3D Bayesian agent: stages lib | prior | test (like scripts/arena_agent.py).

lib:   perceived (sensor-model) boss poses from 24 training bosses -> syllable library
prior: the learning agent fights 24 training bosses (3 attempts each) -> cross-boss prior
test:  12 held-out-subject bosses x 6 attempts, conditions learn / frozen_prior / learn_noprior
"""
import os, sys, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from multiprocessing import Pool
from arena3d import library as L, mocap
from arena3d.boss import sample_boss
from arena3d.env import Arena3D, N_BUTTONS, NONE, ROLL, GUARD
from arena3d import agent as A

U = L.load("checkpoints/arena3d_units.pkl")
_, _, SUBS = L.split_subjects(U)
TRAIN = [i for i, u in enumerate(U) if u.subject not in SUBS]
TEST = [i for i, u in enumerate(U) if u.subject in SUBS]
C = None
LIB = "checkpoints/arena3d_syllables.pkl"
PRIOR = "checkpoints/arena3d_prior.pkl"
NB_TRAIN, NB_TEST, ATT = int(os.environ.get("NB_TRAIN", 24)), int(os.environ.get("NB_TEST", 12)), int(os.environ.get("ATT", 6))


def clips():
    global C
    if C is None:
        C = mocap.load()
    return C


def pstrike():
    best = None
    for u in U:
        if u.subject in SUBS or u.kind != "strike" or not 18 <= u.length <= 40 or u.travel > 0.3:
            continue
        k = int(np.argmax(u.tip_speed)); p = u.tip_rel[k]
        if 0.7 < p[0] < 1.4 and abs(p[1]) < 0.35 and 0.8 < p[2] < 1.6 and k < 15:
            if best is None or u.peak_speed > best.peak_speed:
                best = u
    return best


def train_boss(i):
    return sample_boss(np.random.default_rng(100 + i), U, TRAIN, name=f"train{i}")


def test_boss(i):
    return sample_boss(np.random.default_rng(5000 + i), U, TEST, name=f"test{i}")


def collect(i, frames=1500):
    env = Arena3D(train_boss(i), U, clips(), seed=i, player_strike=pstrike())
    env.reset()
    sen = A.Sensor(env, 6, seed=i)
    rng = np.random.default_rng(i)
    seqs, cur, a = [], [], 0
    for t in range(frames):
        if t % 20 == 0:
            a = int(rng.choice([1, 5, 3, 7, 0])) * N_BUTTONS + int(rng.choice([NONE, NONE, NONE, ROLL, GUARD]))
        _, _, done, _ = env.step(a if env.player_free() else 0)
        P, _ = sen()
        if P is None:
            if len(cur) > 20:
                seqs.append(np.array(cur))
            cur = []
        else:
            cur.append(A.centric(P)[0])
        if done:
            env.reset()
    if len(cur) > 20:
        seqs.append(np.array(cur))
    return seqs


def fight_series(args):
    spec_fn, i, mem, learn, attempts, seed = args
    lib = A.SyllableLibrary.load(LIB)
    env = Arena3D(spec_fn(i), U, clips(), seed=seed, player_strike=pstrike())
    agent = A.BayesAgent3D(lib, mem, learn=learn, seed=seed)
    out = []
    for k in range(attempts):
        env.rng = np.random.default_rng(seed * 100 + k)
        env.boss.rng = env.rng
        o, t, bhp = A.run_fight(env, agent)
        out.append((o, t, bhp, env.hp))
    return out, agent.mem


if __name__ == "__main__":
    stage = sys.argv[1]
    procs = int(os.environ.get("PROCS", 3))
    t0 = time.time()
    if stage == "lib":
        with Pool(procs) as p:
            S = sum(p.map(collect, range(NB_TRAIN)), [])
        print("frames", sum(len(s) for s in S), "segments", len(S), flush=True)
        lib = A.fit_library(S, K=40)
        lib.save(LIB)
        print("library saved", time.time() - t0)
    elif stage == "prior":
        lib = A.SyllableLibrary.load(LIB)
        jobs = [(train_boss, i, A.BossMemory(lib.K), True, int(os.environ.get("PRIOR_ATT", 3)), i) for i in range(NB_TRAIN)]
        with Pool(procs) as p:
            res = p.map(fight_series, jobs)
        mems = [m for _, m in res]
        pickle.dump(A.merge_prior(mems, 1.0), open(PRIOR, "wb"))
        wins = np.array([[o[0] == "win" for o in r] for r, _ in res])
        print("training-boss win by attempt", wins.mean(0).round(2), "boss hp left",
              np.mean([[o[2] for o in r] for r, _ in res], 0).round(2), time.time() - t0, flush=True)
    elif stage == "test":
        lib = A.SyllableLibrary.load(LIB)
        prior = pickle.load(open(PRIOR, "rb"))
        conds = {"learn": lambda: prior.scaled(0.2), "frozen_prior": lambda: prior.scaled(0.2),
                 "learn_noprior": lambda: A.BossMemory(lib.K)}
        learn = {"learn": True, "frozen_prior": False, "learn_noprior": True}
        out = {}
        for c in os.environ.get("CONDS", "learn,frozen_prior,learn_noprior").split(","):
            jobs = [(test_boss, i, conds[c](), learn[c], ATT, 7000 + i) for i in range(NB_TEST)]
            with Pool(procs) as p:
                res = p.map(fight_series, jobs)
            W = np.array([[o[0] == "win" for o in r] for r, _ in res])
            first = [int(np.argmax(w)) + 1 if w.any() else None for w in W]
            out[c] = {"win_by_attempt": W.mean(0).tolist(), "first_win": first,
                      "boss_hp_left": np.mean([[o[2] for o in r] for r, _ in res], 0).tolist(),
                      "outcomes": [[o[0] for o in r] for r, _ in res]}
            print(c, "win by attempt", W.mean(0).round(2), "won ever", W.any(1).mean().round(2), "boss hp left",
                  np.round(out[c]["boss_hp_left"], 2), round(time.time() - t0), flush=True)
        json.dump(out, open("results/arena3d_agent_v1.json", "w"), indent=1)
