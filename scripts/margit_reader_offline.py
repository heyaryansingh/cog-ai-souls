"""Offline calibration of the episodic reader's predictor: collect (attempt, context, time-to-danger) with a
mixed scripted player, then leave-one-attempt-out AUC for 'danger lands in the roll window' under feature sets."""
import os, sys, pickle
sys.path.insert(0, "/home/user/cog-ai-souls"); os.chdir("/home/user/cog-ai-souls")
exec(open("scripts/arena3d_margit.py").read().split('if __name__ == "__main__"')[0].replace("os.path.dirname(os.path.abspath(__file__))", '"scripts"'))
from arena3d.reader import EpisodicReader
from arena3d.env import N_BUTTONS, NONE, LIGHT, ROLL, HEAL
from sklearn.metrics import roc_auc_score


def collect(seed):
    env = Arena3D(spec(), U, clips(), seed=seed, player_strike=pstrike()); obs = env.reset()
    ag = EpisodicReader(seed=seed); ag.reset(env); ag.f = ag._feature(env)
    rng = np.random.default_rng(seed); env.hp = 1e9; env.flasks = 999
    rows = []
    ag._store = lambda f, y: rows.append((f, y))
    for t in range(int(os.environ.get("FRAMES", 5000))):
        if env.player_free():
            u = rng.random()
            dist = np.linalg.norm(env.boss.root - env.p_xy)
            if u < 0.02: a = HEAL
            elif u < 0.05: a = int(rng.choice([3, 5, 7])) * N_BUTTONS + ROLL
            elif dist > 0.8 + 0.25 * env.spec.scale + (3 if (t // 600) % 2 else 0): a = N_BUTTONS + NONE
            elif dist < 2 + (3 if (t // 600) % 2 else 0) and (t // 600) % 2: a = 5 * N_BUTTONS + NONE
            else: a = LIGHT
        else: a = 0
        obs, r, done, info = env.step(a)
        ag.observe(obs, env, info)
        if env.boss_hp < 0.1 * env.spec.hp: env.boss_hp = 0.45 * env.spec.hp
    return seed, np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


if __name__ == "__main__":
    from multiprocessing import Pool
    if not os.path.exists("data/margit_ctx.pkl"):
        with Pool(2) as p:
            D = p.map(collect, range(6))
        pickle.dump(D, open("data/margit_ctx.pkl", "wb"))
    D = pickle.load(open("data/margit_ctx.pkl", "rb"))
    print("frames", sum(len(d[2]) for d in D), "base rate roll-window", np.mean(np.concatenate([(d[2] >= 3) & (d[2] <= 10) for d in D])))
    sets = {"all": slice(None), "pose": np.r_[0:78], "ctx": np.r_[78:83], "pose_now": np.r_[0:39], "ctx+vel": np.r_[39:83]}
    for name, sl in sets.items():
        for k in (15, 50):
            aucs = []
            for i in range(len(D)):
                Xtr = np.vstack([D[j][1] for j in range(len(D)) if j != i])[:, sl]
                Ytr = np.concatenate([D[j][2] for j in range(len(D)) if j != i])
                sub = np.random.default_rng(0).choice(len(Ytr), min(20000, len(Ytr)), replace=False); Xtr, Ytr = Xtr[sub], Ytr[sub]
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-3; Xtr = (Xtr - mu) / sd
                te = np.random.default_rng(1).choice(len(D[i][2]), 1500, replace=False)
                Xte = (D[i][1][te][:, sl] - mu) / sd; Yte = D[i][2][te]
                p = []
                for x in Xte:
                    d = ((Xtr - x) ** 2).sum(1); nn = np.argpartition(d, k)[:k]
                    p.append(np.mean((Ytr[nn] >= 3) & (Ytr[nn] <= 10)))
                lab = (Yte >= 3) & (Yte <= 10)
                aucs.append(roc_auc_score(lab, p))
            print(name, "k", k, "AUC", np.round(np.mean(aucs), 3), np.round(aucs, 2), flush=True)
