"""B30 diag: what hits the simulator-free learner on the bosses it never damages? (source x move, risk at the time)"""
import os, sys, collections
__file__ = "scripts/general_learner.py"
exec(open(__file__).read().rsplit("if __name__", 1)[0])
for b in [int(x) for x in os.environ.get("BS", "4,5").split(",")]:
    sp = pboss(b); n = len(sp.program.moves)
    maps = DangerMaps(n, seed=b); g = SC.LearnedGrammarPlanner(n, seed=b)
    scout(sp, int(os.environ.get("WATCH", 1500)), maps, g, 500 + b)
    pl = LearnedModelPlanner(maps, g, seed=b)
    env = Arena3D(sp, U, clips(), seed=100 * b, player_strike=pstrike()); env.reset(); pl.reset(env); done = False
    src = collections.Counter(); dists = []; acts = collections.Counter()
    while not done:
        a = pl.act(env); acts[a % 6] += 1
        _, _, done, info = env.step(a); maps.observe(env, info)
        if info.get("hit"):
            mv = sp.program.moves[env.boss.move_id] if env.boss.kind == "strike" else None
            src[(info["hit"][0] if info["hit"][0] in ("projectile", "ring") else "strike", mv.name if mv else env.boss.kind,
                 round(float(mv.ring), 1) if mv else 0, bool(mv.projectile) if mv else 0)] += 1
            dists.append((round(float(np.linalg.norm(env.boss.root - env.p_xy)), 1), env.t - maps.onset, maps.label is not None, int(pl.hold_n > 0 or not env.player_free())))
    print("boss", b, info["outcome"], "t", env.t, "dmg", round(1 - env.boss_hp / sp.hp, 3), "buttons", dict(acts))
    print("   hits by", dict(src)); print("   (dist, tau since onset, identified?, committed?) at hit", dists)
    print("   moves", [(m.name, round(float(m.ring), 1), bool(m.projectile), m.phase) for m in sp.program.moves])
