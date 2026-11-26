"""Cross-boss learning (R5) on procedural program bosses: does a library pooled from K training bosses make a
new boss faster to learn? For each test boss: scout (watch) W frames, then NF fights with the learned-world-model
planner (no simulator). Prior = danger maps pooled by move prototype (shape + ender/ring/projectile/leap flags)
from K training bosses, injected as pseudo-counts (strength PRIOR_N)."""
import os, sys, json, pickle
sys.path.insert(0, "/home/user/cog-ai-souls"); os.chdir("/home/user/cog-ai-souls")
__file__ = "/home/user/cog-ai-souls/scripts/margit_worldmodel.py"
exec(open(__file__).read().rsplit("if __name__", 1)[0])
from arena3d.program import random_program, build_program_boss
from arena3d.worldmodel import DangerMaps, LearnedModelPlanner
BOSSES = {}


def proto(m):
    return (m.shape, m.ender, m.ring > 0, m.projectile, m.lunge_boost > 0)


def pboss(b):
    if b not in BOSSES:
        rng = np.random.default_rng(7000 + b)
        moves, strings = random_program(rng)
        BOSSES[b] = build_program_boss(rng, U, TEST, clips(), moves=moves, strings=strings,
                                       scale=float(np.exp(rng.uniform(np.log(1.4), np.log(2.4)))), name=f"proc{b}",
                                       weapon=str(rng.choice(["sword", "greatsword", "club"])))
    return BOSSES[b]


def scout(sp, frames, maps, g, seed):
    for ep in range(2):
        env = Arena3D(sp, U, clips(), seed=seed + ep, player_strike=pstrike()); env.reset(); maps.reset(env); g.reset(env)
        if ep == 1: env.boss_hp = 0.45 * sp.hp; env.phase = True
        rng = np.random.default_rng(seed + ep)
        for t in range(frames):
            g.observe(env)
            dist = np.linalg.norm(env.boss.root - env.p_xy); want = [1.5, 3.0, 5.0, 8.0][(t // 300) % 4]
            if env.player_free() and rng.random() < 0.03:
                a = int(rng.choice([3, 5, 7])) * N_BUTTONS + ROLL
            else:
                a = (1 if dist > want + 0.4 else 5 if dist < want - 0.4 else int(rng.choice([3, 7]))) * N_BUTTONS + NONE
            _, _, _, info = env.step(a); maps.observe(env, info); env.hp = 1e9
            if ep == 0 and env.boss_hp < 0.5 * sp.hp: env.boss_hp = sp.hp


def train_library(K, frames):
    path = f"checkpoints/general_library_K{K}_W{frames}_{os.environ.get('WM_FRAME', 'onset')}_n{os.environ.get('WM_NORM', '0')}.pkl"
    if os.path.exists(path): return pickle.load(open(path, "rb"))
    lib = {}
    for b in range(100, 100 + K):
        sp = pboss(b); n = len(sp.program.moves)
        maps = DangerMaps(n, seed=b); g = SC.LearnedGrammarPlanner(n, seed=b)
        scout(sp, frames, maps, g, 900 + b)
        for i, m in enumerate(sp.program.moves):
            k = proto(m); N0, H0 = lib.get(k, (0, 0))
            lib[k] = (N0 + maps.N[i], H0 + maps.H[i])
    pickle.dump(lib, open(path, "wb")); return lib


def one(args):
    cond, b, K = args
    sp = pboss(b); n = len(sp.program.moves); W = int(os.environ.get("WATCH", 1500))
    acc = float(os.environ.get("WM_ACC", 0.7))
    maps = DangerMaps(n, seed=b, acc=acc); g = (SC.HierGrammar if os.environ.get("WM_HIERG") == "1" else SC.LearnedGrammarPlanner)(n, seed=b, acc=acc)
    if K > 0:
        lib = train_library(K, int(os.environ.get("LIB_WATCH", 4500))); c = float(os.environ.get("PRIOR_N", 0.25))
        for i, m in enumerate(sp.program.moves):
            if proto(m) in lib:
                N0, H0 = lib[proto(m)]; maps.N[i] += c * N0 / max(1, K); maps.H[i] += c * H0 / max(1, K)
    if W > 0: scout(sp, W, maps, g, 500 + b)
    pl = LearnedModelPlanner(maps, g, seed=b); out = []
    for k in range(int(os.environ.get("NF", 2))):
        if k and os.environ.get("NOCARRY") == "1":     # control: a fresh learner (same scouting) for every attempt
            maps = DangerMaps(n, seed=b, acc=acc); g = (SC.HierGrammar if os.environ.get("WM_HIERG") == "1" else SC.LearnedGrammarPlanner)(n, seed=b, acc=acc)
            if W > 0: scout(sp, W, maps, g, 500 + b)
            pl = LearnedModelPlanner(maps, g, seed=b)
        env = Arena3D(sp, U, clips(), seed=100 * b + k, player_strike=pstrike()); env.reset(); pl.reset(env); done = False
        while not done:
            _, _, done, info = env.step(pl.act(env)); maps.observe(env, info)
        r = (cond, b, k, info["outcome"], round(1 - env.boss_hp / sp.hp, 3), env.t); print("FIGHT", *r, flush=True); out.append(r)
    return out


if __name__ == "__main__":
    from multiprocessing import Pool
    Ks = [int(x) for x in os.environ.get("KS", "0,12").split(",")]
    for K in Ks:
        if K: train_library(K, int(os.environ.get("LIB_WATCH", 4500)))
    tests = [int(x) for x in os.environ["BS"].split(",")] if os.environ.get("BS") else range(int(os.environ.get("NB", 6)))
    with Pool(int(os.environ.get("PROCS", 2))) as p:
        res = sum(p.map(one, [(f"K{K}", b, K) for b in tests for K in Ks]), [])
    json.dump(res, open(os.environ.get("OUT", "results/general_learner.json"), "w"))
    for K in Ks:
        r = [x for x in res if x[0] == f"K{K}"]
        print(f"K{K}", "wins", sum(x[3] == "win" for x in r), "/", len(r), "damage", np.mean([x[4] for x in r]).round(3), "frames", np.mean([x[5] for x in r]).round(0))
