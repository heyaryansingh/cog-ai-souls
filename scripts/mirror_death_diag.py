"""B72 diag: what still hits the mirror-simulation learner? Over NFD fights: each hit's move, time since the move's
onset, what the player was doing, and what the planner had imagined for its chosen action 1-24 frames before."""
import os, sys, pickle, collections
sys.argv = ["x"]; sys.path.insert(0, "/home/user/cog-ai-souls"); os.chdir("/home/user/cog-ai-souls")
exec(open("scripts/mirror_fight.py").read().rsplit("if __name__", 1)[0])
mem, g = pickle.load(open(CK, "rb")); pl = MirrorPlanner(mem, g, seed=0)
names = ["none", "roll", "light", "guard", "heal"]
by_move = collections.Counter(); by_act = collections.Counter(); by_tau = collections.Counter(); kinds = collections.Counter(); dm = []
for k in range(int(os.environ.get("NFD", 4))):
    env = Arena3D(spec(), U, clips(), seed=8 + k, player_strike=pstrike()); env.reset(); pl.reset(env); done = False; last = []
    while not done:
        a = pl.act(env); _, _, done, info = env.step(a); mem.observe(env)
        if info.get("hit"):
            mv = spec().program.moves[env.boss.move_id].name if env.boss.kind == "strike" else env.boss.kind
            tau = env.t - mem.onset
            by_move[mv] += 1; by_act[names[env.act] if env.act < 5 else env.act] += 1; kinds[info["hit"][0] if info["hit"][0] in ("projectile", "shockwave") else "blade"] += 1
            by_tau["<6" if tau < 6 else "6-20" if tau < 20 else "20-40" if tau < 40 else "40+"] += 1
    dm.append(round(1 - env.boss_hp / env.spec.hp, 2))
print("damage", dm); print("hits by move", by_move.most_common()); print("by player action", dict(by_act)); print("by tau", dict(by_tau)); print("by kind", dict(kinds))
