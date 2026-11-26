import os, sys, collections
sys.path.insert(0, "/home/user/cog-ai-souls"); os.chdir("/home/user/cog-ai-souls")
sys.argv = ["x", "run"]
src = open("scripts/arena3d_margit.py").read().split("if __name__ == \"__main__\"")[0].replace("os.path.dirname(os.path.abspath(__file__))", "\"scripts\"")
exec(src)
from arena3d.env import N_BUTTONS, HEAL, ROLL, NONE
def kiter(env, rng):
    # walk away (direction 5 = away from boss), heal often
    if env.t % 90 == 30 and env.flasks > 0: return 5 * N_BUTTONS + HEAL if False else HEAL
    return 5 * N_BUTTONS + NONE
for phase2 in (False, True):
    env = Arena3D(spec(), U, clips(), seed=1, player_strike=pstrike()); env.reset()
    if phase2: env.boss_hp = 0.4 * env.spec.hp
    moves = collections.Counter(); ev = collections.Counter(); rng = np.random.default_rng(0); last = None
    env.hp = 1e9; env.flasks = 99
    for t in range(3000):
        _, _, done, info = env.step(kiter(env, rng))
        s = getattr(env, "boss_serial", -1)
        if s != last and env.boss.cur_move is not None: moves[env.boss.cur_move.name] += 1; last = s
        if info["hit"]: ev[info["hit"][0]] += 1
        if done: break
    print("phase2", phase2, dict(moves), dict(ev), "proj alive", len(env.projectiles), flush=True)
