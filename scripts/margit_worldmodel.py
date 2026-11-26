"""R3 test: planning with a learned danger-map world model (no boss simulator) on the Margit-like boss.
Phase A: watch/learn (circling + probing player with infinite HP, 2 x WATCH frames). Phase B: fights on seeds
8..8+N with the learned model (maps and strings keep learning across fights in a series)."""
import os, sys, pickle
sys.path.insert(0, "/home/user/cog-ai-souls"); os.chdir("/home/user/cog-ai-souls")
exec(open("scripts/arena3d_margit.py").read().split('if __name__ == "__main__"')[0].replace("os.path.dirname(os.path.abspath(__file__))", '"scripts"'))
import arena3d.scripted as SC
from arena3d.worldmodel import DangerMaps, LearnedModelPlanner
from arena3d.env import N_BUTTONS, NONE, ROLL, LIGHT
CK = os.environ.get("WM_CK", "checkpoints/margit_worldmodel.pkl")


VARIANTS = [   # B47: the agent's candidate world models; it keeps all and selects by predictive likelihood
    {"WM_FRAME": "current", "WM_TRACK": "0", "WM_PERSIST": "0", "WM_HIER": "0"},          # plain v6
    {"WM_FRAME": "onset", "WM_TRACK": "1", "WM_PERSIST": "1", "WM_HIER": "4"},            # aim + attention + back-off
]


class Multi:
    """Fans perception out to several world-model variants; `best` = highest per-frame predictive log-likelihood."""
    def __init__(self, ms): self.ms = ms
    def reset(self, env): [m.reset(env) for m in self.ms]
    def observe(self, env, info): [m.observe(env, info) for m in self.ms]
    @property
    def update(self): return self.ms[0].update
    @update.setter
    def update(self, v):
        for m in self.ms: m.update = v
    def scores(self): return [m.ll / max(1, m.ll_n) for m in self.ms]
    def best(self): return self.ms[int(np.argmax(self.scores()))]


def watch(frames):
    sp = spec(); n = len(sp.program.moves)
    if os.environ.get("WM_SELECT") == "1":
        maps = Multi([DangerMaps(n, seed=3, cfg=c) for c in VARIANTS])
    else:
        maps = DangerMaps(n, seed=3)
    g = SC.LearnedGrammarPlanner(n, seed=3)
    for ep in range(2):
        env = Arena3D(sp, U, clips(), seed=600 + ep, player_strike=pstrike()); env.reset(); maps.reset(env); g.reset(env)
        if ep == 1: env.boss_hp = 0.45 * sp.hp; env.phase = True
        rng = np.random.default_rng(ep)
        for t in range(frames):
            g.observe(env)
            dist = np.linalg.norm(env.boss.root - env.p_xy); want = [1.5, 3.0, 5.0, 8.0][(t // 300) % 4]
            if env.player_free() and rng.random() < 0.03:
                a = int(rng.choice([3, 5, 7])) * N_BUTTONS + ROLL
            else:
                a = (1 if dist > want + 0.4 else 5 if dist < want - 0.4 else int(rng.choice([3, 7]))) * N_BUTTONS + NONE
            _, _, _, info = env.step(a); maps.observe(env, info); env.hp = 1e9
            if env.boss_hp < 0.5 * sp.hp and ep == 0: env.boss_hp = sp.hp
    return maps, g


def series(rep):
    maps, g = pickle.load(open(CK, "rb")); out = []
    maps.update = g.update = os.environ.get("LEARN", "1") == "1"
    multi = maps if isinstance(maps, Multi) else None
    pl = LearnedModelPlanner(maps.best() if multi else maps, g, seed=rep)
    for k in range(int(os.environ.get("N", 6))):
        if multi:
            pl.maps = multi.best(); print("SELECT", rep, k, int(np.argmax(multi.scores())), [round(x, 4) for x in multi.scores()], flush=True)
        seed = 8 + rep * 100 + k if rep else 8 + k
        env = Arena3D(spec(), U, clips(), seed=seed, player_strike=pstrike()); env.reset(); pl.reset(env)
        if multi: multi.reset(env)
        counts = g.counts; g.counts = counts; done = False
        while not done:
            a = pl.act(env)
            _, _, done, info = env.step(a); maps.observe(env, info)
        r = ("worldmodel", rep, seed, info["outcome"], round(1 - env.boss_hp / env.spec.hp, 3), env.t)
        print("FIGHT", *r, flush=True); out.append(r)
    return out


if __name__ == "__main__":
    from multiprocessing import Pool
    if not os.path.exists(CK):
        pickle.dump(watch(int(os.environ.get("WATCH", 9000))), open(CK, "wb"))
        m, _ = pickle.load(open(CK, "rb")); m = m.ms[0] if isinstance(m, Multi) else m; print("exposures", int(m.N.sum()), "contacts", int(m.H.sum()), flush=True)
    with Pool(int(os.environ.get("PROCS", 2))) as p:
        res = sum(p.map(series, range(int(os.environ.get("R0", 0)), int(os.environ.get("R0", 0)) + int(os.environ.get("R", 2)))), [])
    print("worldmodel wins", sum(r[3] == "win" for r in res), "/", len(res), "damage", np.mean([r[4] for r in res]).round(3))
