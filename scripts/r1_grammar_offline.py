"""R1 -> grammar without labels. Discover moves by clustering perceived keypoints + swing timing (DP-means), then
learn the boss's strings over the DISCOVERED moves and ask: how much does knowing the previous move tell you about
the next? (leave-one-fight-out information gain, bits). Also: choose the clustering granularity with no labels by
maximising that predictability ("carve the boss at the joints that make it predictable")."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score as nmi
__file__ = "scripts/r1_discovery_offline.py"
src = open(__file__).read().split("F = [feat(r) for r in rows]")[0]
exec(src)
dz_all = np.array([r["danger"] - r["onset"] for r in rows], float)
seeds = np.array([r["seed"] for r in rows]); onset = np.array([r["onset"] for r in rows])
phase = np.array([bool(r["phase"]) for r in rows])


def contexts(ix):
    """prev discovered move in the same fight if it started < 90 frames before, else 'open'."""
    prev = {}
    out = []
    order = sorted(ix, key=lambda i: (seeds[i], onset[i]))
    last = {}
    for i in order:
        s = seeds[i]
        p = last.get(s)
        out.append((i, p if p is not None and onset[i] - onset[p] < 90 else None))
        last[s] = i
    return out


def info_gain(lab, ix, alpha=0.5):
    K = lab.max() + 1; pairs = contexts(ix)
    ce_b, ce_u, n = 0.0, 0.0, 0
    for s in np.unique(seeds[ix]):
        tr = [(i, p) for i, p in pairs if seeds[i] != s]; te = [(i, p) for i, p in pairs if seeds[i] == s]
        cnt, uni = {}, np.zeros(K)
        for i, p in tr:
            c = ("open" if p is None else lab[p], phase[i])
            cnt.setdefault(c, np.zeros(K))[lab[i]] += 1; uni[lab[i]] += 1
        pu = (uni + alpha) / (uni.sum() + alpha * K)
        for i, p in te:
            c = ("open" if p is None else lab[p], phase[i])
            v = cnt.get(c, np.zeros(K)); pb = (v + 2 * pu) / (v.sum() + 2)      # back off to unigram
            ce_b -= np.log2(pb[lab[i]]); ce_u -= np.log2(pu[lab[i]]); n += 1
    return (ce_u - ce_b) / n, ce_b / n


allix = np.arange(len(rows))
print("TRUE labels  K", y.max() + 1, "info gain %.3f bits, CE %.3f" % info_gain(y, allix))
res = []
F = [feat(r, 4, 40) for r in rows]; ok = np.array([f is not None for f in F]); ix = allix[ok]
X = np.array([f for f in F if f is not None]); dz = dz_all[ok]
for nc in (4, 8):
    Z = PCA(nc).fit_transform((X - X.mean(0)) / (X.std(0) + 1e-6)); Z = Z / Z.std()
    for wt in (0.0, 1.0, 2.0, 4.0):
        Zt = np.hstack([Z, wt * ((dz - dz.mean()) / dz.std())[:, None]])
        for q in (1.0, 2.0, 4.0, 8.0):
            lam = q * np.median(((Zt - Zt.mean(0)) ** 2).sum(1)) / 2
            z, C = dpmeans(Zt, lam)
            lab = np.full(len(rows), -1); lab[ix] = z
            g, ce = info_gain(lab, ix)
            # predictability per bit of description: favour gain but penalise splitting (MDL-ish)
            score = g - 0.5 * np.log2(len(C)) / 10
            res.append((score, g, nmi(y[ix], z), len(C), nc, wt, q))
            print("pca %d timing %.0f lam %.0f  K %2d  NMI %.3f  info gain %.3f bits" % (nc, wt, q, len(C), res[-1][2], g))
res.sort(reverse=True)
print("UNSUPERVISED PICK (max predictability score):", "K", res[0][3], "NMI %.3f gain %.3f" % (res[0][2], res[0][1]), "cfg", res[0][4:])
print("best NMI config:", max(res, key=lambda r: r[2]))
print("corr(gain, NMI) over configs: %.2f" % np.corrcoef([r[1] for r in res], [r[2] for r in res])[0, 1])

print("--- expectation-guided discovery: re-assign each swing using what the boss did before and after it")
# hard-EM over a sequence model: score(k) = -|x-mu_k|^2/(2 s^2) + g*[log P(k|prev) + log P(next|k)]
Z = PCA(4).fit_transform((X - X.mean(0)) / (X.std(0) + 1e-6)); Z = Z / Z.std()
Zt = np.hstack([Z, 4.0 * ((dz - dz.mean()) / dz.std())[:, None]])
pos = {i: j for j, i in enumerate(ix)}
pairs = contexts(ix); prevj = np.array([-1] * len(ix)); nextj = np.array([-1] * len(ix))
for i, p in pairs:
    if p is not None and p in pos: prevj[pos[i]] = pos[p]; nextj[pos[p]] = pos[i]
true_prev, true_next = prevj.copy(), nextj.copy()
for q, ctrl in ((4.0, False), (8.0, False), (4.0, True)):
    lam = q * np.median(((Zt - Zt.mean(0)) ** 2).sum(1)) / 2
    z0, C = dpmeans(Zt, lam)
    if ctrl:      # control: same number of links, but each swing's neighbours are random swings (no real sequence)
        print("CONTROL: shuffled sequence links")
        rp = np.random.default_rng(1).permutation(len(ix)); prevj = np.where(true_prev >= 0, rp, -1)
        nextj = np.full(len(ix), -1); nextj[prevj[prevj >= 0]] = np.where(prevj >= 0)[0]
    for g in (0.0, 0.5, 1.0, 2.0):
        z = z0.copy(); K = z.max() + 1
        for it in range(15):
            mu = np.array([Zt[z == k].mean(0) if (z == k).any() else Zt[0] for k in range(K)])
            s2 = np.mean(((Zt - mu[z]) ** 2).sum(1)) / Zt.shape[1]
            T = np.full((K + 1, K), 0.5)                       # row K = 'open'
            for j in range(len(ix)):
                T[z[prevj[j]] if prevj[j] >= 0 else K, z[j]] += 1
            lT = np.log(T / T.sum(1, keepdims=True))
            new = z.copy()
            for j in range(len(ix)):
                sc = -((Zt[j] - mu) ** 2).sum(1) / (2 * s2 * Zt.shape[1])
                sc = sc + g * lT[z[prevj[j]] if prevj[j] >= 0 else K]
                if nextj[j] >= 0: sc = sc + g * lT[:K, z[nextj[j]]]
                new[j] = int(sc.argmax())
            if (new == z).all(): break
            z = new
        lab = np.full(len(rows), -1); lab[ix] = z
        print("lam %.0f  expectation weight %.1f  K %d  NMI %.3f  purity %.3f  info gain %.3f bits (in-sample labels)" % (
            q, g, len(np.unique(z)), nmi(y[ix], z), sum(np.bincount(y[ix][z == k]).max() for k in np.unique(z)) / len(ix), info_gain(lab, ix)[0]))
