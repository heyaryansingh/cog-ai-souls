"""Learning the boss program between deaths with a planner: damage per attempt for learned-grammar planner
(counts kept) vs forget (counts wiped each attempt)."""
import os, sys, json
sys.path.insert(0, "/home/user/cog-ai-souls"); os.chdir("/home/user/cog-ai-souls")
exec(open("scripts/arena3d_margit.py").read().split('if __name__ == "__main__"')[0].replace("os.path.dirname(os.path.abspath(__file__))", '"scripts"'))
import arena3d.scripted as SC


def series(args):
    cond, rep = args
    sp = spec()
    SC._LGP = SC.LearnedGrammarPlanner(len(sp.program.moves), seed=rep, learn=(cond == "learn"))
    out = []
    for a in range(int(os.environ.get("A", 8))):
        env = Arena3D(sp, U, clips(), seed=1000 * rep + a, player_strike=pstrike()); env.reset()
        SC._LGP.reset(env); rng = np.random.default_rng(1000 * rep + a); done = False
        while not done:
            _, _, done, info = env.step(SC.learned_grammar_planner(env, rng))
        out.append((info["outcome"], round(1 - env.boss_hp / env.spec.hp, 3), env.t))
        print(cond, rep, a, *out[-1], "contexts", len(SC._LGP.counts), flush=True)
    return cond, rep, out


if __name__ == "__main__":
    from multiprocessing import Pool
    conds = os.environ.get("CONDS", "learn,forget").split(",")
    with Pool(int(os.environ.get("PROCS", 4))) as p:
        res = p.map(series, [(c, r) for c in conds for r in range(int(os.environ.get("R", 2)))])
    json.dump(res, open("results/margit_grammar.json", "w"))
    for c in conds:
        D = np.array([[x[1] for x in r[2]] for r in res if r[0] == c])
        print(c, "damage by attempt", D.mean(0).round(3), "wins", sum(x[0] == "win" for r in res if r[0] == c for x in r[2]))
