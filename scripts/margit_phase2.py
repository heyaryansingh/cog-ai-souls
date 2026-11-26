"""Phase-shift recovery (coordinator ask; cf. Najar 2026 Directed Skill Graphs). Strings learned by watching
phase 1 only; then fights that start in phase 2 (new hammer moves, new strings). Conditions:
  frozen  - phase-1 counts, no updating (phase-2 moves guessed from an empty table = uniform)
  online  - counts keep updating from perception across phase-2 attempts (the 'slow mind')
  grammar - true program (ceiling)
Metric: fraction of the boss's remaining (phase-2) HP removed, per attempt."""
import os, sys, pickle
sys.path.insert(0, "/home/user/cog-ai-souls"); os.chdir("/home/user/cog-ai-souls")
exec(open("scripts/arena3d_margit.py").read().split('if __name__ == "__main__"')[0].replace("os.path.dirname(os.path.abspath(__file__))", '"scripts"'))
import arena3d.scripted as SC
from arena3d.env import N_BUTTONS, NONE
CK = "checkpoints/margit_grammar_phase1.pkl"


def watch(frames):
    sp = spec(); lg = SC.LearnedGrammarPlanner(len(sp.program.moves), seed=7)
    env = Arena3D(sp, U, clips(), seed=500, player_strike=pstrike()); env.reset(); lg.reset(env)
    for t in range(frames):
        lg.observe(env)
        dist = np.linalg.norm(env.boss.root - env.p_xy); want = 2.0 + 4.0 * ((t // 400) % 2)
        env.step((1 if dist > want + 0.5 else 5 if dist < want - 0.5 else 3) * N_BUTTONS + NONE); env.hp = 1e9
        env.boss_hp = env.spec.hp
    return lg


def p2_env(seed):
    env = Arena3D(spec(), U, clips(), seed=seed, player_strike=pstrike()); env.reset()
    env.boss_hp = 0.49 * env.spec.hp; env.phase = True
    return env


def series(args):
    cond, rep = args
    out = []
    lg = pickle.load(open(CK, "rb")); lg.update = cond == "online"
    for a in range(int(os.environ.get("A", 6))):
        seed = 3000 + 100 * rep + a
        env = p2_env(seed); h0 = env.boss_hp; rng = np.random.default_rng(seed); done = False
        if cond == "grammar":
            pol = SC.grammar_oracle
        else:
            counts = lg.counts; lg.reset(env); lg.counts = counts; SC._LGP = lg; pol = SC.learned_grammar_planner
        while not done:
            _, _, done, info = env.step(pol(env, rng))
        r = (cond, rep, a, info["outcome"], round(1 - env.boss_hp / h0, 3), env.t,
             int(sum(v.sum() for k, v in lg.counts.items() if k[2])))
        print("P2", *r, flush=True); out.append(r)
    return out


if __name__ == "__main__":
    from multiprocessing import Pool
    if not os.path.exists(CK):
        pickle.dump(watch(int(os.environ.get("WATCH", 9000))), open(CK, "wb"))
    jobs = [("online", 0), ("online", 1), ("frozen", 0), ("grammar", 0)]
    with Pool(4) as p:
        res = sum(p.map(series, jobs), [])
    for c in ("online", "frozen", "grammar"):
        D = np.array([[x[4] for x in res if x[0] == c and x[1] == r] for r in (0, 1) if any(x[0] == c and x[1] == r for x in res)])
        print(c, "phase-2 damage by attempt", D.mean(0).round(3), "mean", D.mean().round(3))
