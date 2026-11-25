"""Do the learned syllables carve the boss's motion at its true joints? (held-out-subject bosses)

Perceived poses (sensor model) -> library HMM Viterbi path; compared with the true motion unit per frame.
NMI(syllable, true unit), NMI(syllable, strike-vs-move), boundary F1 (+-3 frames)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi
sys.argv = [sys.argv[0], "none"] + sys.argv[1:]
exec(open("/home/user/cog-ai-souls/scripts/arena3d_agent.py").read().split("if __name__")[0])


def viterbi(lib, F):
    T, K = len(F), lib.K
    ll = np.stack([lib.loglik(f) for f in F])
    d = lib.logpi + ll[0]
    bp = np.zeros((T, K), int)
    for t in range(1, T):
        s = d[:, None] + lib.logA
        bp[t] = s.argmax(0)
        d = s.max(0) + ll[t]
    z = np.zeros(T, int); z[-1] = d.argmax()
    for t in range(T - 1, 0, -1):
        z[t - 1] = bp[t, z[t]]
    return z


def bf1(a, b, tol=3):
    A = np.nonzero(np.diff(a))[0]; B = np.nonzero(np.diff(b))[0]
    if len(A) == 0 or len(B) == 0:
        return 0.0
    p = np.mean([np.min(np.abs(B - x)) <= tol for x in A]); r = np.mean([np.min(np.abs(A - x)) <= tol for x in B])
    return 2 * p * r / (p + r + 1e-9)


lib = A.SyllableLibrary.load(LIB)
res = []
for i in range(int(os.environ.get("NB", 6))):
    env = Arena3D(test_boss(i), U, clips(), seed=i, player_strike=pstrike()); env.reset()
    sen = A.Sensor(env, 0, noise=float(os.environ.get("NOISE", 0.06)), p_stay_on=1.0, seed=i)       # no dropout / latency: score segmentation itself
    X, lab, kind = [], [], []
    rng = np.random.default_rng(i); a = 0
    for t in range(1500):
        if t % 20 == 0:
            a = int(rng.choice([1, 5, 3, 7, 0])) * N_BUTTONS + NONE
        _, _, done, _ = env.step(a)
        P, _ = sen()
        tr = env.truth()
        X.append(A.centric(P)[0]); lab.append(hash(tr["unit"]) % 100003 * 7 + int(tr["f"] >= 0)); kind.append(tr["kind"] == "strike")
        if done:
            env.reset()
    X = np.array(X)
    F = [lib.featurize(X[t], X[max(t - 1, 0)]) for t in range(len(X))]
    z = viterbi(lib, F)
    lab = np.array(lab)
    r = {"boss": i, "nmi_unit": float(nmi(lab, z)), "nmi_strike": float(nmi(kind, z)), "boundary_f1": float(bf1(z, lab)),
         "n_units": int(len(np.unique(lab))), "n_syl": int(len(np.unique(z)))}
    print(r, flush=True); res.append(r)
json.dump(res, open("results/arena3d_syllables.json", "w"), indent=1)
print("mean NMI unit", np.mean([r["nmi_unit"] for r in res]), "boundary F1", np.mean([r["boundary_f1"] for r in res]))
