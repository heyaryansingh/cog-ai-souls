"""Smoke test + video strip: sample a boss from the open library, run a scripted player, save frames."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from arena3d import library as L, mocap
from arena3d.boss import sample_boss
from arena3d.env import Arena3D, ROLL, LIGHT, GUARD, NONE, N_BUTTONS

U = L.load("checkpoints/arena3d_units.pkl")
train, test, subs = L.split_subjects(U)
idx_train = [i for i, u in enumerate(U) if u.subject not in subs]
clips = mocap.load()
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
spec = sample_boss(rng, U, idx_train)
pstrike = max((u for u in train if u.kind == "strike" and 25 <= u.length <= 40 and u.reach < 1.2),
              key=lambda u: u.peak_speed)
print("boss", spec.scale, spec.weapon, len(spec.strikes), "strikes; hold med", spec.hold_med.round(0))
env = Arena3D(spec, U, clips, seed=1, render=False, width=320, height=240, player_strike=pstrike)
import mujoco
R = mujoco.Renderer(env.m, 240, 320)
obs = env.reset()
frames, t0, hits = [], time.time(), []
done = False
while not done:
    tr = env.truth()
    dist = np.linalg.norm(env.boss.root - env.p_xy)
    a = 1 * N_BUTTONS + NONE if dist > 1.0 + 0.35 * env.spec.scale else (0 * N_BUTTONS + LIGHT if env.rng.random() < 0.3 else 5 * N_BUTTONS + GUARD)
    obs, r, done, info = env.step(a)
    if info["hit"] or info.get("boss_hit") or info["blocked"]:
        hits.append((env.t, info["hit"], info.get("boss_hit"), info["blocked"]))
    if (info["hit"] or info.get("boss_hit") or env.t % 25 == 0) and len(frames) < 24:
        frames.append(env.render(R))
print("outcome", info["outcome"], "frames", env.t, "fps", env.t / (time.time() - t0))
print("events", hits[:12], len(hits))
from PIL import Image
frames = frames[:len(frames) // 6 * 6] or frames[:1]
rows = [np.concatenate(frames[i:i + 6], 1) for i in range(0, min(len(frames), 24), 6)]
Image.fromarray(np.concatenate(rows, 0)).save("results/arena3d_strip.png")
