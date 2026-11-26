"""B60 diag: how does the attention window (persist) change behaviour and what hits the player on a hand-authored
boss? Runs one 6-fight series from a given world-model checkpoint and tallies hit sources, distances and actions."""
import os, sys, collections, pickle
sys.path.insert(0, "/home/user/cog-ai-souls"); os.chdir("/home/user/cog-ai-souls")
exec(open("scripts/margit_worldmodel.py").read().rsplit("if __name__", 1)[0])
maps, g = pickle.load(open(CK, "rb")); pl = LearnedModelPlanner(maps, g, seed=0)
src = collections.Counter(); nf_shock = 0; acts = collections.Counter(); dist_hist = []; hit_d = []; dealt = 0; frames = 0; dmg = []
for k in range(int(os.environ.get("N", 6))):
    env = Arena3D(spec(), U, clips(), seed=8 + k, player_strike=pstrike()); env.reset(); pl.reset(env); done = False; bhp = env.boss_hp
    while not done:
        free_before = env.player_free(); a = pl.act(env); acts[a % N_BUTTONS] += 1
        _, _, done, info = env.step(a); maps.observe(env, info); frames += 1
        d = float(np.linalg.norm(env.boss.root - env.p_xy)); dist_hist.append(d)
        if env.boss_hp < bhp: dealt += 1
        bhp = env.boss_hp
        if info.get("hit"):
            mv = spec().program.moves[env.boss.move_id] if env.boss.kind == "strike" else None
            src[(info["hit"][0] if info["hit"][0] in ("projectile", "shockwave") else "strike", mv.name if mv else env.boss.kind)] += 1
            hit_d.append(round(d, 1)); nf_shock += (info["hit"][0] == "shockwave" and not free_before)
    dmg.append(round(1 - env.boss_hp / env.spec.hp, 3))
print(CK, "dmg", dmg, "mean", round(float(np.mean(dmg)), 3), "frames", frames, "dealt hits", dealt)
print("  hits by", sorted(src.items(), key=lambda x: -x[1]))
print("  buttons", dict(acts), "dist mean", round(float(np.mean(dist_hist)), 2), "frac<3m", round(float(np.mean(np.array(dist_hist) < 3)), 2))
print("  dist at hit", sorted(hit_d)); print("  ring reflex triggers", getattr(pl, "ring_trig", 0), "rolls", getattr(pl, "ring_rolls", 0))
print("  shock hits while not free", nf_shock, "of", sum(v for k_, v in src.items() if k_[0] == "shockwave"))
