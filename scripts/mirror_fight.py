"""B65: fights with the mirror-simulation learner (arena3d/mirror.py) on a hand-authored boss (BOSS=godrick|margit).
Scout (watch with infinite HP, 2 x WATCH frames, same protocol as margit_worldmodel), then series of N fights;
memory and strings keep learning across fights."""
import os, sys, pickle
sys.path.insert(0, "/home/user/cog-ai-souls"); os.chdir("/home/user/cog-ai-souls")
exec(open("scripts/arena3d_margit.py").read().split('if __name__ == "__main__"')[0].replace("os.path.dirname(os.path.abspath(__file__))", '"scripts"'))
import arena3d.scripted as SC
from arena3d.mirror import MirrorMemory, MirrorPlanner
from arena3d.env import N_BUTTONS, NONE, ROLL
from arena3d.percept import FairView, NMAX
FAIR = os.environ.get("MIRROR_FAIR") == "1"     # B77: boss known only through a delayed noisy pose stream
CK = os.environ.get("MIRROR_CK", f"checkpoints/mirror_{os.environ.get('BOSS', 'margit')}.pkl")


def watch(frames):
    sp = spec(); n = len(sp.program.moves)
    nz = float(os.environ.get("MIRROR_NOISE", 1.0))
    mem = MirrorMemory(n, seed=3, noise_j=0.03 * nz, noise_p=0.03 * nz); g = SC.LearnedGrammarPlanner(n, seed=3)
    if FAIR:
        mem = MirrorMemory(NMAX, seed=3, acc=1.0, noise_j=0.0, noise_p=0.0); g = SC.LearnedGrammarPlanner(NMAX, seed=3, acc=1.0)
    for ep in range(2):
        env = Arena3D(sp, U, clips(), seed=600 + ep, player_strike=pstrike()); env.reset(); mem.reset(env); g.reset(env)
        O = FairView(env, mem, seed=ep) if FAIR else env
        if ep == 1: env.boss_hp = 0.45 * sp.hp; env.phase = True
        rng = np.random.default_rng(ep)
        for t in range(frames):
            g.observe(O)
            dist = np.linalg.norm(env.boss.root - env.p_xy); want = [1.5, 3.0, 5.0, 8.0][(t // 300) % 4]
            if env.player_free() and rng.random() < 0.03:
                a = int(rng.choice([3, 5, 7])) * N_BUTTONS + ROLL
            else:
                a = (1 if dist > want + 0.4 else 5 if dist < want - 0.4 else int(rng.choice([3, 7]))) * N_BUTTONS + NONE
            env.step(a)
            if FAIR: O.update()
            mem.observe(O); env.hp = 1e9
            if env.boss_hp < 0.5 * sp.hp and ep == 0: env.boss_hp = sp.hp
    return mem, g


def series(rep):
    mem, g = pickle.load(open(CK, "rb")); out = []
    pl = MirrorPlanner(mem, g, seed=rep, n=int(os.environ.get("MIRROR_N", 3)))
    view = None
    for k in range(int(os.environ.get("N", 6))):
        seed = 8 + rep * 100 + k if rep else 8 + k
        env = Arena3D(spec(), U, clips(), seed=seed, player_strike=pstrike()); env.reset(); pl.reset(env); done = False
        if FAIR:
            view = view or FairView(env, mem, seed=rep); view.attach(env); pl.view = view
        while not done:
            _, _, done, info = env.step(pl.act(env))
            if FAIR: view.update()
            mem.observe(view if FAIR else env)
        r = ("mirror", rep, seed, info["outcome"], round(1 - env.boss_hp / env.spec.hp, 3), env.t)
        print("FIGHT", *r, flush=True); out.append(r)
        print("STILL", rep, k, getattr(pl, "n_still", 0), getattr(pl, "n_imag", 0), flush=True)
        if FAIR: print("FAIR", rep, k, "onsets", view.n_det, "labels", getattr(mem, "n_labels", 0), flush=True)
    return out


if __name__ == "__main__":
    from multiprocessing import Pool
    if not os.path.exists(CK):
        mem, g = watch(int(os.environ.get("WATCH", 9000))); pickle.dump((mem, g), open(CK, "wb"))
        print("episodes", {(spec().program.moves[k].name if not FAIR else k): len(v) for k, v in mem.eps.items()}, "rings", {k: np.round(v, 1).tolist() for k, v in mem.ring_R.items()}, flush=True)
    with Pool(int(os.environ.get("PROCS", 2))) as p:
        res = sum(p.map(series, range(int(os.environ.get("R0", 0)), int(os.environ.get("R0", 0)) + int(os.environ.get("R", 2)))), [])
    print("mirror wins", sum(r[3] == "win" for r in res), "/", len(res), "damage", np.mean([r[4] for r in res]).round(3))
