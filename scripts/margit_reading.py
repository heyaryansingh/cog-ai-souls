"""Move-reading study on the Margit-like boss: how early can the move identity and its danger time be read
from perceived (noisy, delayed, dropout) keypoints? Collect per-strike perceived pose windows, then
leave-one-attempt-out k-NN identity decoding vs frames since onset, and danger-time error with/without identity."""
import os, sys, pickle, collections
sys.path.insert(0, "/home/user/cog-ai-souls"); os.chdir("/home/user/cog-ai-souls")
sys.argv = ["x", "run"]
exec(open("scripts/arena3d_margit.py").read().split('if __name__ == "__main__"')[0].replace("os.path.dirname(os.path.abspath(__file__))", '"scripts"'))
from arena3d.agent import Sensor, centric
from multiprocessing import Pool
W = 40


def collect(seed):
    env = Arena3D(spec(), U, clips(), seed=seed, player_strike=pstrike()); env.reset()
    sen = Sensor(env, seed=seed); rng = np.random.default_rng(seed)
    env.hp = 1e9; env.flasks = 0
    hist, rows, cur, last = [], [], None, -1
    for t in range(int(os.environ.get("FRAMES", 4000))):
        P, _ = sen()
        hist.append(None if P is None else centric(P)[0])
        _, _, done, info = env.step(S.aggro(env, rng))
        b = env.boss; s = getattr(env, "boss_serial", -1)
        if s != last and b.kind == "strike" and b.cur_move is not None:
            if cur: rows.append(cur)
            cur = {"seed": seed, "move": b.cur_move.name, "onset": t, "danger": None, "hold": b.hold, "phase": env.phase}
            last = s
        if cur and cur["danger"] is None and b.kind == "strike" and b.f >= b.peak and s == last:
            cur["danger"] = t
        if env.boss_hp < 0.1 * env.spec.hp: env.boss_hp = 0.45 * env.spec.hp
        if done: break
    for r in rows:
        r["obs"] = [hist[i] if i < len(hist) else None for i in range(r["onset"], r["onset"] + W)]
    return [r for r in rows if r["danger"] is not None]


if __name__ == "__main__":
    if not os.path.exists("data/margit_reading.pkl") or os.environ.get("RECOLLECT"):
        with Pool(int(os.environ.get("PROCS", 2))) as p:
            rows = sum(p.map(collect, range(int(os.environ.get("NSEED", 6)))), [])
        pickle.dump(rows, open("data/margit_reading.pkl", "wb"))
    rows = pickle.load(open("data/margit_reading.pkl", "rb"))
    print(len(rows), "strikes", collections.Counter(r["move"] for r in rows))
    names = sorted(set(r["move"] for r in rows)); y = np.array([names.index(r["move"]) for r in rows])
    seeds = np.array([r["seed"] for r in rows]); dz = np.array([r["danger"] - r["onset"] for r in rows], float)

    def feat(r, k):
        o = [x for x in r["obs"][:k] if x is not None]
        if not o: return np.zeros(39 * 2)
        a = np.array(o); return np.concatenate([a[-1], a[-1] - a[0]])
    out = {}
    for k in (1, 4, 8, 12, 16, 20, 24, 30):
        X = np.array([feat(r, k) for r in rows]); acc, err_id, err_base, err_knn = [], [], [], []
        for s in np.unique(seeds):
            tr, te = seeds != s, seeds == s
            d = ((X[te][:, None] - X[tr][None]) ** 2).sum(-1)
            nn = np.argsort(d, 1)[:, :5]
            pred = np.array([np.bincount(y[tr][n], minlength=len(names)).argmax() for n in nn])
            acc += list(pred == y[te])
            med = {c: np.median(dz[tr][y[tr] == c]) if (y[tr] == c).any() else np.median(dz[tr]) for c in range(len(names))}
            err_id += list(np.abs(np.array([med[c] for c in pred]) - dz[te]))
            err_base += list(np.abs(np.median(dz[tr]) - dz[te]))
            err_knn += list(np.abs(np.median(dz[tr][nn], 1) - dz[te]))
        out[k] = dict(acc=float(np.mean(acc)), chance=float(np.bincount(y).max() / len(y)),
                      mae_identity=float(np.mean(err_id)), mae_knn=float(np.mean(err_knn)), mae_none=float(np.mean(err_base)))
        print("k", k, {a: round(b, 3) for a, b in out[k].items()}, flush=True)
    print("danger offset by move (median, IQR frames):", {n: (np.median(dz[y == i]), np.percentile(dz[y == i], 75) - np.percentile(dz[y == i], 25)) for i, n in enumerate(names)})
    import json; json.dump(out, open("results/margit_reading.json", "w"), indent=1)
