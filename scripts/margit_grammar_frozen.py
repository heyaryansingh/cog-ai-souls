"""Learn Margit's strings by watching (cheap circling player, infinite HP, noisy 70% identifications at 12
frames), freeze the counts, then plan with them on the same 16 seeds as the perceptual/grammar oracles (B17)."""
import os, sys, pickle
sys.path.insert(0, "/home/user/cog-ai-souls"); os.chdir("/home/user/cog-ai-souls")
exec(open("scripts/arena3d_margit.py").read().split('if __name__ == "__main__"')[0].replace("os.path.dirname(os.path.abspath(__file__))", '"scripts"'))
import arena3d.scripted as SC
from arena3d.env import N_BUTTONS, NONE


def watch(frames):
    sp = spec(); lg = SC.LearnedGrammarPlanner(len(sp.program.moves), seed=7)
    for ep in range(2):
        env = Arena3D(sp, U, clips(), seed=500 + ep, player_strike=pstrike()); env.reset(); lg.reset(env)
        env.hp = 1e9
        if ep == 1: env.boss_hp = 0.45 * sp.hp; env.phase = True
        for t in range(frames):
            lg.observe(env)
            dist = np.linalg.norm(env.boss.root - env.p_xy)
            want = 2.0 + 4.0 * ((t // 400) % 2)          # alternate near / far spacing
            a = (1 if dist > want + 0.5 else 5 if dist < want - 0.5 else 3) * N_BUTTONS + NONE
            env.step(a); env.hp = 1e9
    return lg


def fight(k):
    SC._LGP = pickle.load(open("checkpoints/margit_grammar_counts.pkl", "rb"))
    env = Arena3D(spec(), U, clips(), seed=k, player_strike=pstrike()); env.reset()
    counts = SC._LGP.counts; SC._LGP.learn = True; SC._LGP.reset(env); SC._LGP.counts = counts
    SC._LGP.update = os.environ.get("FROZEN", "1") != "1"
    rng = np.random.default_rng(k); done = False
    while not done:
        _, _, done, info = env.step(SC.learned_grammar_planner(env, rng))
    r = ("learned_frozen", k, info["outcome"], round(env.boss_hp / env.spec.hp, 3), env.t)
    print("FIGHT", *r, flush=True)
    return r


if __name__ == "__main__":
    from multiprocessing import Pool
    if not os.path.exists("checkpoints/margit_grammar_counts.pkl"):
        lg = watch(int(os.environ.get("WATCH", 9000)))
        pickle.dump(lg, open("checkpoints/margit_grammar_counts.pkl", "wb"))
        prog = spec().program
        for c, v in sorted(lg.counts.items(), key=lambda x: str(x[0])):
            top = np.argsort(-v)[:3]
            print(c, int(v.sum()), [(prog.moves[i].name, int(v[i])) for i in top])
    with Pool(4) as p:
        res = p.map(fight, range(8, 24))
    print("learned_frozen damage", np.mean([1 - r[3] for r in res]).round(3))
