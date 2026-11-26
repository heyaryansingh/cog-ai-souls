"""R3 test: planning with a learned danger-map world model (no boss simulator) on the Margit-like boss.
Phase A: watch/learn (circling + probing player with infinite HP, 2 x WATCH frames). Phase B: fights on seeds
8..8+N with the learned model (maps and strings keep learning across fights in a series)."""
import os, sys, pickle
sys.path.insert(0, "/home/user/cog-ai-souls"); os.chdir("/home/user/cog-ai-souls")
exec(open("scripts/arena3d_margit.py").read().split('if __name__ == "__main__"')[0].replace("os.path.dirname(os.path.abspath(__file__))", '"scripts"'))
import arena3d.scripted as SC
from arena3d.worldmodel import DangerMaps, LearnedModelPlanner, Multi, VARIANTS, family_select
from arena3d.env import N_BUTTONS, NONE, ROLL, LIGHT
CK = os.environ.get("WM_CK", "checkpoints/margit_worldmodel.pkl")


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
    rets = [[] for _ in (multi.ms if multi else [])]; frng = np.random.default_rng(31 * rep + 5)
    pl = LearnedModelPlanner(maps.best() if multi else maps, g, seed=rep)
    for k in range(int(os.environ.get("N", 6))):
        if multi and os.environ.get("WM_SELECT_BY") == "combined":
            # prior from predictive likelihood (the model that 'reads' the boss best), evidence from returns:
            # Gaussian posterior mean per model, prior mean = 0.45 + bonus for the likelihood favourite, prior n0 = 2
            fav = int(np.argmax(multi.scores()))
            post = [((0.45 + (0.08 if i == fav else 0.0)) * 2 + sum(v)) / (2 + len(v)) for i, v in enumerate(rets)]
            sel = int(np.argmax(post)); pl.maps = multi.ms[sel]
            print("SELECT", rep, k, sel, [round(x, 3) for x in post], flush=True)
        elif multi and os.environ.get("WM_SELECT_BY") == "return":     # bandit over world models, scored by damage dealt
            n_ = [len(v) for v in rets]
            if min(n_) == 0:
                sel = int(np.argmin(n_))
            else:
                ucb = [np.mean(v) + 0.3 * np.sqrt(2 * np.log(k + 1) / len(v)) for v in rets]; sel = int(np.argmax(ucb))
            pl.maps = multi.ms[sel]; print("SELECT", rep, k, sel, [round(float(np.mean(v)), 3) if v else None for v in rets], flush=True)
        elif multi and os.environ.get("WM_SELECT_BY") == "family":     # B52: family prior + Thompson on returns
            sel, post = family_select(rets, frng); pl.maps = multi.ms[sel]
            print("SELECT", rep, k, sel, [round(x, 3) for x in post], flush=True)
        elif multi:
            sel = int(np.argmax(multi.scores()))
            pl.maps = multi.best(); print("SELECT", rep, k, sel, [round(x, 4) for x in multi.scores()], flush=True)
        seed = 8 + rep * 100 + k if rep else 8 + k
        env = Arena3D(spec(), U, clips(), seed=seed, player_strike=pstrike()); env.reset(); pl.reset(env)
        if multi: multi.reset(env)
        counts = g.counts; g.counts = counts; done = False
        while not done:
            a = pl.act(env)
            _, _, done, info = env.step(a); maps.observe(env, info)
        r = ("worldmodel", rep, seed, info["outcome"], round(1 - env.boss_hp / env.spec.hp, 3), env.t)
        if multi: rets[sel].append(r[4])
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
