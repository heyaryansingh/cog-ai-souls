"""Episodic reader vs memoryless control on the Margit-like boss: attempts 1..A, learning between deaths."""
import os, sys, json
sys.path.insert(0, "/home/user/cog-ai-souls"); os.chdir("/home/user/cog-ai-souls")
exec(open("scripts/arena3d_margit.py").read().split('if __name__ == "__main__"')[0].replace("os.path.dirname(os.path.abspath(__file__))", '"scripts"'))
from arena3d.reader import EpisodicReader, EpisodicControl, run_attempt


def series(args):
    cond, rep = args
    Ag = EpisodicControl if os.environ.get("AGENT", "reader") == "ec" else EpisodicReader
    ag = Ag(forget=(cond == "forget"), seed=rep)
    out = []
    for a in range(int(os.environ.get("A", 8))):
        env = Arena3D(spec(), U, clips(), seed=1000 * rep + a, player_strike=pstrike())
        o, t, bhp, hp = run_attempt(env, ag)
        out.append((o, round(bhp, 3), t, len(ag.Y), ag.counts))
        print(cond, rep, a, o, round(bhp, 3), t, len(ag.Y), ag.counts, flush=True)
    return cond, rep, out


if __name__ == "__main__":
    from multiprocessing import Pool
    conds = os.environ.get("CONDS", "learn,forget").split(",")
    with Pool(int(os.environ.get("PROCS", 2))) as p:
        res = p.map(series, [(c, r) for c in conds for r in range(int(os.environ.get("R", 3)))])
    json.dump(res, open(os.environ.get("OUT", "results/margit_reader.json"), "w"))
    for c in conds:
        rs = [r[2] for r in res if r[0] == c]
        print(c, "boss hp left by attempt", np.mean([[x[1] for x in r] for r in rs], 0).round(2),
              "wins", sum(x[0] == "win" for r in rs for x in r))
