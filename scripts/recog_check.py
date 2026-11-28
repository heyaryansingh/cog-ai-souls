"""B77 check of segmentation-free recognition (reviewer item): while the boss attacks, does recognition from the
last L perceived poses pick the right move and the right frame in it? Memory = 1.5-min scouting with true labels
(label noise .3 is in the memory itself); perception = true pose, or the honest FairView stream."""
import os, sys, pickle
os.environ["MIRROR_SEGFREE"] = "1"; sys.argv = ["x"]
sys.path.insert(0, "/home/user/cog-ai-souls"); os.chdir("/home/user/cog-ai-souls")
exec(open("scripts/mirror_fight.py").read().rsplit("if __name__", 1)[0])
from arena3d.mirror import recognise
from arena3d.percept import FairView
B = os.environ.get("BOSS", "godrick")
for mode in ["truth", "fair"]:
    mem, g = pickle.load(open(f"checkpoints/mirror_{B}_w1500.pkl", "rb")); mem.update = False
    env = Arena3D(spec(), U, clips(), seed=77, player_strike=pstrike()); env.reset(); mem.reset(env)
    V = FairView(env, None, seed=1) if mode == "fair" else None
    hist = []; rng = np.random.default_rng(0); top1 = []; ferr = []
    for t in range(int(os.environ.get("T", 2500))):
        dist = np.linalg.norm(env.boss.root - env.p_xy); want = [1.5, 3.0, 5.0][(t // 300) % 3]
        a = (1 if dist > want + 0.4 else 5 if dist < want - 0.4 else int(rng.choice([3, 7]))) * N_BUTTONS + NONE
        env.step(a); env.hp = 1e9
        if env.boss_hp < 0.5 * env.spec.hp: env.boss_hp = env.spec.hp
        if V is not None: V.update()
        q = (V.d.qpos if V is not None else env.d.qpos)[:63].copy()
        hist.append((q, env.boss.kind, env.boss.move_id, env.t - getattr(env, "boss_onset_t", env.t) - 1))
        mem.recent = [h[0] for h in hist[-int(os.environ.get("L", 4)):]]
        lag = V.L if V is not None else 0
        if len(hist) > lag + 4 and t % 3 == 0:
            q_, kind, mid, tau = hist[-1 - lag]           # ground truth at the perceived frame
            if kind != "strike": continue
            rec = recognise(mem, None, L=int(os.environ.get("L", 4)), d_now=float(np.linalg.norm(env.p_xy - q_[:2])))
            if not rec: continue
            w, ep, f, lab = rec[int(np.argmax([r[0] for r in rec]))]
            top1.append(lab == mid); ferr.append(abs(f - tau) if lab == mid else np.nan)
    print(B, mode, "top-1 move", round(np.mean(top1), 2), "median frame error (when right)", np.nanmedian(ferr), "n", len(top1), flush=True)
