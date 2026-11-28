"""B84 diag under realism v2 + honest mode: what hits the agent? For each hit: the boss move, frames since its true
onset, what the player was doing (and how far into it), whether it came during hitstun (a chain), and the press
that preceded it."""
import os, sys, pickle, collections
sys.argv = ["x"]; sys.path.insert(0, "/home/user/cog-ai-souls"); os.chdir("/home/user/cog-ai-souls")
exec(open("scripts/mirror_fight.py").read().rsplit("if __name__", 1)[0])
from arena3d.env import HEAVY, BACKSTEP
mem, g = pickle.load(open(CK, "rb")); pl = MirrorPlanner(mem, g, seed=0, n=int(os.environ.get("MIRROR_N", 3)))
names = {0: "none/walk", 1: "roll", 2: "light", 3: "guard", 4: "heal", 5: "heavy", 6: "backstep"}
CNT = collections.Counter(); view = None; dm = []
for k in range(int(os.environ.get("NFD", 4))):
    env = Arena3D(spec(), U, clips(), seed=8 + k, player_strike=pstrike()); env.reset(); pl.reset(env); done = False
    if FAIR: view = view or FairView(env, mem, seed=0); view.attach(env); pl.view = view
    last_hit = -999; last_press = None
    while not done:
        a = pl.act(env)
        if a % N_BUTTONS not in (0,): last_press = (env.t, a % N_BUTTONS)
        act0, actt0, stag0 = env.act, env.act_t, env.stagger
        _, _, done, info = env.step(a)
        if FAIR: view.update()
        mem.observe(view if FAIR else env)
        if info.get("hit"):
            mv = spec().program.moves[env.boss.move_id].name if env.boss.kind == "strike" else env.boss.kind
            tau = env.t - getattr(env, "boss_onset_t", env.t)
            CNT["move:" + mv] += 1
            if stag0 > 0: CNT["during hitstun (chain)"] += 1
            elif act0 == 1: CNT[f"while rolling, frame {actt0} ({'after i-frames' if actt0 > 14 else 'before i-frames'})"] += 1
            else: CNT["while " + names.get(act0, str(act0))] += 1
            CNT["tau " + ("<10" if tau < 10 else "10-20" if tau < 20 else "20-35" if tau < 35 else "35+")] += 1
            if last_press and env.t - last_press[0] < 25: CNT[f"press <25f before: {names[last_press[1]]}"] += 1
            last_hit = env.t
        if info.get("critical"): CNT["CRITICAL dealt"] += 1
        if info.get("stance_break"): CNT["stance breaks dealt"] += 1
        if info.get("blocked"): CNT["blocked"] += 1
    dm.append(round(min(1, 1 - env.boss_hp / env.spec.hp), 2))
print("damage", dm)
for key, v in sorted(CNT.items()): print(f"  {key}: {v}")
