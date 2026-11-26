"""R1 offline: label-free move discovery from perceived onset-aligned keypoint windows (DP-means / CRP-style
clustering). Data: data/margit_reading.pkl (442 strikes, perceived poses, true labels used only for scoring)."""
import pickle, numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score as nmi

rows = pickle.load(open("data/margit_reading.pkl", "rb"))
names = sorted(set(r["move"] for r in rows)); y = np.array([names.index(r["move"]) for r in rows])


def feat(r, a=4, b=20):
    o = [x for x in r["obs"][a:b] if x is not None]; o0 = [x for x in r["obs"][:a] if x is not None]
    if len(o) < 4: return None
    o = np.array(o); base = np.array(o0).mean(0) if o0 else o[0]
    # trajectory summary: 4 temporal chunks of the window, relative to the onset pose
    ch = np.array_split(o, 4)
    return np.concatenate([c.mean(0) - base for c in ch])


def dpmeans(X, lam, iters=20, seed=0):
    rng = np.random.default_rng(seed); order = rng.permutation(len(X))
    C = [X[order[0]]]; z = np.zeros(len(X), int)
    for _ in range(iters):
        for i in order:
            d = ((np.array(C) - X[i]) ** 2).sum(1)
            if d.min() > lam: C.append(X[i]); z[i] = len(C) - 1
            else: z[i] = int(d.argmin())
        C = [X[z == k].mean(0) for k in range(len(C)) if (z == k).any()]
        _, z = np.unique(z, return_inverse=True)
    return z, np.array(C)


F = [feat(r) for r in rows]; keep = np.array([f is not None for f in F])
X = np.array([f for f in F if f is not None]); yk = y[keep]
for win in [(4, 12), (4, 16), (4, 20), (4, 28)]:
    F = [feat(r, *win) for r in rows]; X = np.array([f for f in F if f is not None]); yk = y[[f is not None for f in F]]
    Z = PCA(8).fit_transform((X - X.mean(0)) / (X.std(0) + 1e-6))
    for q in (0.5, 1.0, 2.0):
        lam = q * np.median(((Z - Z.mean(0)) ** 2).sum(1))
        z, C = dpmeans(Z, lam)
        pur = sum(np.bincount(yk[z == k]).max() for k in np.unique(z)) / len(yk)
        print("window", win, "lam x", q, "clusters", len(C), "NMI", round(nmi(yk, z), 3), "purity", round(pur, 3))

from sklearn.mixture import BayesianGaussianMixture
print("--- larger lambda and DP-GMM")
for win in [(4, 20), (4, 28)]:
    F = [feat(r, *win) for r in rows]; X = np.array([f for f in F if f is not None]); yk = y[[f is not None for f in F]]
    for nc in (4, 6, 8):
        Z = PCA(nc).fit_transform((X - X.mean(0)) / (X.std(0) + 1e-6))
        for q in (4.0, 8.0):
            lam = q * np.median(((Z - Z.mean(0)) ** 2).sum(1)) / 2
            z, C = dpmeans(Z, lam)
            pur = sum(np.bincount(yk[z == k]).max() for k in np.unique(z)) / len(yk)
            print("win", win, "pca", nc, "dpmeans lam", q, "K", len(C), "NMI", round(nmi(yk, z), 3), "purity", round(pur, 3))
        g = BayesianGaussianMixture(n_components=15, weight_concentration_prior=0.1, covariance_type="diag", max_iter=500, random_state=0).fit(Z)
        z = g.predict(Z); K = len(np.unique(z))
        pur = sum(np.bincount(yk[z == k]).max() for k in np.unique(z)) / len(yk)
        print("win", win, "pca", nc, "DP-GMM K", K, "NMI", round(nmi(yk, z), 3), "purity", round(pur, 3))

print("--- retrospective labelling: full window + when the swing was heard (multimodal)")
dz_all = np.array([r["danger"] - r["onset"] for r in rows], float)
for win in [(4, 40)]:
    F = [feat(r, *win) for r in rows]; ok = np.array([f is not None for f in F])
    X = np.array([f for f in F if f is not None]); yk = y[ok]; dz = dz_all[ok]
    for nc in (4, 8):
        Z = PCA(nc).fit_transform((X - X.mean(0)) / (X.std(0) + 1e-6)); Z = Z / Z.std()
        for wt in (0.0, 1.0, 2.0, 4.0):
            Zt = np.hstack([Z, wt * ((dz - dz.mean()) / dz.std())[:, None]])
            for q in (2.0, 4.0):
                lam = q * np.median(((Zt - Zt.mean(0)) ** 2).sum(1)) / 2
                z, C = dpmeans(Zt, lam)
                pur = sum(np.bincount(yk[z == k]).max() for k in np.unique(z)) / len(yk)
                print("pca", nc, "timing wt", wt, "lam", q, "K", len(C), "NMI", round(nmi(yk, z), 3), "purity", round(pur, 3))
