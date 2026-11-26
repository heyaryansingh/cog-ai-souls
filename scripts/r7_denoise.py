"""B35: can human-like perceptual denoising rescue within-boss reading under the calibrated sensor noise?
Noisy pose windows -> (a) raw features, (b) projected onto a low-dim pose manifold learned from the agent's OWN noisy
observations (PCA, unsupervised), (c) manifold + temporal smoothing (constant-velocity least-squares fit over the
visible frames). Metric: kNN tip-path error on held-out swings of seen bosses, and same on unseen bosses."""
import os, pickle, numpy as np, torch
rows = pickle.load(open("data/r7_swings.pkl", "rb"))
from sklearn.decomposition import PCA
tr = [r for r in rows if r["b"] >= 100]
allp = np.array([x for r in tr for x in r["obs"] if x is not None])
pcas = {d: PCA(d).fit(allp) for d in (6, 12, 20)}


def feats(r, k, mode, d=12):
    o = [(i, x) for i, x in enumerate(r["obs"][:k]) if x is not None]
    if not o:
        return np.zeros(3 * 39 + 4)
    t = np.array([i for i, _ in o], float); X = np.array([x for _, x in o])
    if mode != "raw":
        X = pcas[d].inverse_transform(pcas[d].transform(X))
    if mode == "manifold+smooth" and len(o) >= 3:
        A = np.c_[np.ones_like(t), t]; coef, *_ = np.linalg.lstsq(A, X, rcond=None)
        a1, v = coef[0] + coef[1] * t[-1], coef[1]; a0 = coef[0] + coef[1] * t[0]
    else:
        a1, a0 = X[-1], X[0]; v = (a1 - a0) / max(1, t[-1] - t[0])
    return np.concatenate([a1, v * 10, a0, [len(o) / k, r["prel"][0], r["prel"][1], r["scale"]]])


def err(P, Y, M):
    d = np.sqrt(((P - Y) ** 2).reshape(len(Y), -1, 2).sum(-1)); m = M.reshape(len(Y), -1, 2)[..., 0]
    return float((d * m).sum() / m.sum())


def knn_err(A, B, k, mode, d=12, nn=10):
    Xa = np.array([feats(r, k, mode, d) for r in A]); Xb = np.array([feats(r, k, mode, d) for r in B])
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-4; Xa, Xb = (Xa - mu) / sd, (Xb - mu) / sd
    Ya = np.array([r["path"] for r in A]).reshape(len(A), -1); Yb = np.array([r["path"] for r in B]).reshape(len(B), -1)
    Ma, Mb = ~np.isnan(Ya), ~np.isnan(Yb); Ya, Yb = np.nan_to_num(Ya), np.nan_to_num(Yb)
    D = ((Xb[:, None] - Xa[None]) ** 2).sum(-1); idx = np.argsort(D, 1)[:, :nn]
    P = np.array([(Ya[i] * Ma[i]).sum(0) / np.maximum(Ma[i].sum(0), 1) for i in idx])
    return err(P, Yb, Mb)


seenA = [r for i, r in enumerate(tr) if i % 5]; seenB = [r for i, r in enumerate(tr) if i % 5 == 0]
te = [r for r in rows if r["b"] < 100]
for k in (12, 20):
    for mode in ("raw", "manifold", "manifold+smooth"):
        ds = (12,) if mode == "raw" else (6, 12, 20)
        for d in ds:
            print(f"k={k} {mode:16s} d={d:2d}  seen-boss kNN {knn_err(seenA, seenB, k, mode, d):.3f}   unseen-boss kNN {knn_err(tr, te, k, mode, d):.3f}", flush=True)
