"""Audit: render what the world-model agent sees in the 1 s before it gets hit (lock-on camera), Godrick."""
import os, sys, pickle, collections
sys.path.insert(0, "/home/user/cog-ai-souls"); os.chdir("/home/user/cog-ai-souls")
exec(open("scripts/margit_worldmodel.py").read().rsplit("if __name__", 1)[0])
import mujoco
from PIL import Image
maps, g = pickle.load(open(CK, "rb")); pl = LearnedModelPlanner(maps, g, seed=0)
env = Arena3D(spec(), U, clips(), seed=9, player_strike=pstrike()); env.reset(); pl.reset(env)
R = mujoco.Renderer(env.m, 240, 320); buf = collections.deque(maxlen=40); strips = []; done = False; log = []
names = ["none", "roll", "light", "guard", "heal"]
while not done and len(strips) < int(os.environ.get("NHIT", 4)):
    a = pl.act(env); _, _, done, info = env.step(a)
    mv = spec().program.moves[env.boss.move_id].name if env.boss.kind == "strike" else env.boss.kind
    buf.append((env.render(R), env.t, mv, names[env.act], round(float(np.linalg.norm(env.boss.root - env.p_xy)), 1)))
    if info.get("hit"):
        fr = list(buf)[::5]; strips.append(np.concatenate([f[0] for f in fr], 1)); log.append([(f[1], f[2], f[3], f[4]) for f in fr] + [info["hit"]])
os.makedirs("results/filmstrips", exist_ok=True)
Image.fromarray(np.concatenate(strips, 0)).save("results/filmstrips/godrick_hits.png")
for l in log: print(l)
