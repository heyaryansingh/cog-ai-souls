"""Collect experience on training bosses and train the threat network.

    python scripts/arena3d_threat.py collect   -> data/arena3d_threat.npz
    python scripts/arena3d_threat.py train     -> checkpoints/arena3d_threat.pt (+ held-out AUC on test bosses)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from multiprocessing import Pool
sys.argv = sys.argv[:2] + ["none"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "arena3d_agent.py")).read().split("if __name__")[0])
from arena3d import threat as T
from arena3d.env import LIGHT, HEAL

DATA = os.environ.get("THREAT_DATA", "data/arena3d_threat.npz")


def explore_policy(env, rng, state):
    """Mixed exploration: approach, strafe, retreat, random rolls/guards/attacks, held for a while."""
    if state["left"] <= 0:
        dist = np.linalg.norm(env.boss.root - env.p_xy)
        r = rng.random()
        if dist > 4 + env.spec.scale and r < 0.6:
            a = 1 * N_BUTTONS + NONE
        elif r < 0.25:
            a = int(rng.choice([1, 2, 8])) * N_BUTTONS + NONE
        elif r < 0.45:
            a = int(rng.choice([3, 7, 4, 6])) * N_BUTTONS + NONE
        elif r < 0.6:
            a = 5 * N_BUTTONS + NONE
        elif r < 0.72:
            a = int(rng.choice([3, 5, 7, 4, 6])) * N_BUTTONS + ROLL
        elif r < 0.85:
            a = LIGHT
        else:
            a = int(rng.choice([0, 5])) * N_BUTTONS + GUARD
        state["a"], state["left"] = a, int(rng.integers(4, 25))
    state["left"] -= 1
    return state["a"]


def collect(args):
    i, frames, which = args
    spec = train_boss(i) if which == "train" else test_boss(i)
    env = Arena3D(spec, U, clips(), seed=i, player_strike=pstrike())
    env.reset()
    sen = A.Sensor(env, 6, seed=i)
    ph = T.PoseHistory()
    rng = np.random.default_rng(i)
    st = {"left": 0, "a": 0}
    F, LOC, HIT, PROT, EP = [], [], [], [], []
    ep = 0
    frame = (env.boss.root.copy(), 0.0)
    for t in range(frames):
        a = explore_policy(env, rng, st) if env.player_free() else 0
        hp0 = env.hp
        _, _, done, info = env.step(a)
        P, _ = sen()
        if P is not None:
            x, hip, fwd, h = A.centric(P)
            ph.push(x, h)
            frame = (hip[:2], fwd)
        else:
            ph.push(None, None)
        F.append(ph.feat())
        # perceived boss frame (last seen) and my true world position: labels at t+k use the frame seen at t
        LOC.append(np.concatenate([env.p_xy, frame[0], [frame[1], ph.height]]))
        HIT.append(info["hit"] is not None)
        PROT.append(env.in_iframes() or info["blocked"] or info["dodged"] or env.act == GUARD)
        EP.append(ep)
        if done:
            env.reset(); sen = A.Sensor(env, 6, seed=i * 1000 + ep); ph = T.PoseHistory(); ep += 1
            frame = (env.boss.root.copy(), 0.0)
    return np.array(F), np.array(LOC), np.array(HIT), np.array(PROT), np.array(EP) + i * 1000


def make_samples(F, LOC, HIT, PROT, EP, rng, neg_keep=0.25):
    X, Y = [], []
    n = len(F)
    for t in range(n - 30):
        for ki, k in enumerate(T.KS):
            if EP[t + k + 2] != EP[t]:
                continue
            win = slice(t + k, t + k + 3)
            y = HIT[win].any()
            if not y and PROT[win].any():
                continue
            if not y and rng.random() > neg_keep:
                continue
            # my world position at t+k expressed in the boss frame perceived at t
            px, py, bx, by, fwd, h = LOC[t]
            mx, my = LOC[t + k][:2]
            rel = np.array([mx - bx, my - by])
            c, s = np.cos(-fwd), np.sin(-fwd)
            local = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]]) / max(h, 0.5)
            X.append(np.concatenate([F[t], T.query_feats(local, ki)]))
            Y.append(float(y))
    return np.array(X, np.float32), np.array(Y, np.float32)


if __name__ == "__main__":
    stage = sys.argv[1]
    procs = int(os.environ.get("PROCS", 3))
    if stage == "collect":
        t0 = time.time()
        jobs = [(i, int(os.environ.get("FRAMES", 12000)), "train") for i in range(int(os.environ.get("NB", 24)))]
        jobs += [(i, 6000, "test") for i in range(6)]
        with Pool(procs) as p:
            res = p.map(collect, jobs)
        rng = np.random.default_rng(0)
        tr = [make_samples(*r, rng) for r in res[:-6]]
        te = [make_samples(*r, np.random.default_rng(1), neg_keep=1.0) for r in res[-6:]]
        os.makedirs("data", exist_ok=True)
        np.savez_compressed(DATA, Xtr=np.vstack([x for x, _ in tr]), Ytr=np.concatenate([y for _, y in tr]),
                            Xte=np.vstack([x for x, _ in te]), Yte=np.concatenate([y for _, y in te]))
        print("train", sum(len(y) for _, y in tr), "pos", sum(y.sum() for _, y in tr), "test", sum(len(y) for _, y in te),
              "pos", sum(y.sum() for _, y in te), "hits/fight-min", round(time.time() - t0), "s", flush=True)
    elif stage == "train":
        import torch, torch.nn as nn
        from sklearn.metrics import roc_auc_score, average_precision_score
        d = np.load(DATA)
        Xtr, Ytr, Xte, Yte = [torch.from_numpy(d[k]) for k in ("Xtr", "Ytr", "Xte", "Yte")]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-3
        net = T.ThreatNet()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        pw = torch.tensor((1 - Ytr.mean()) / Ytr.mean() * 0.25)   # neg_keep=0.25 already reweights
        lossf = nn.BCEWithLogitsLoss()
        for ep in range(int(os.environ.get("EPOCHS", 12))):
            perm = torch.randperm(len(Xtr))
            for b in range(0, len(Xtr), 512):
                idx = perm[b:b + 512]
                loss = lossf(net((Xtr[idx] - mu) / sd), Ytr[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                p = torch.sigmoid(net((Xte - mu) / sd)).numpy()
            print(f"epoch {ep} loss {loss.item():.4f} test AUC {roc_auc_score(Yte, p):.3f} AP {average_precision_score(Yte, p):.3f} base {Yte.mean():.3f}", flush=True)
        torch.save({"net": net.state_dict(), "mu": mu, "sd": sd}, "checkpoints/arena3d_threat.pt")
