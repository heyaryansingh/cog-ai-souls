"""Can the agent discover the boss's moves from raw pose? Sticky Gaussian HMM syllables vs true
animation segments. python scripts/arena_perception.py"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import normalized_mutual_info_score as nmi
from arena.env import Arena
from arena.moves import sample_boss
from arena.perception import boss_centric, features
from arena.scripted import CheatDodger

def record(boss, seed, ticks=2400):
    env = Arena(boss, seed=seed); obs = env.reset(); P = CheatDodger()
    X, lab_pose, lab_move = [], [], []
    for _ in range(ticks):
        tr = env.truth(); b = env.boss
        if b.stagger: lp, lm = "stagger", "stagger"
        elif b.move < 0: lp, lm = "walk", "walk"
        else:
            m = boss.moves[b.move]; s = m.segments[b.seg]; lp, lm = f"{s.kind}:{s.pose}", m.name
        a = P(obs, env) if env.player_free() else 0
        obs, r, done, _ = env.step(a)
        X.append(obs[:51]); lab_pose.append(lp); lab_move.append(lm)
        if done: break
    return np.array(X), lab_pose, lab_move

def boundary_f1(z, truth, tol=3):
    bz = np.flatnonzero(np.diff(z) != 0); bt = np.flatnonzero(np.array(truth[1:]) != np.array(truth[:-1]))
    if len(bz) == 0 or len(bt) == 0: return 0.0
    hit_p = np.mean([np.min(np.abs(bt - x)) <= tol for x in bz]); hit_r = np.mean([np.min(np.abs(bz - x)) <= tol for x in bt])
    return 2 * hit_p * hit_r / max(hit_p + hit_r, 1e-9)

rng = np.random.default_rng(7)
res = {}
for arch in ("knight", "beast", "caster", "giant"):
    rows = []
    for i in range(4):
        boss = sample_boss(rng, arch)
        seqs = [record(boss, s) for s in range(3)]
        tr_X = np.vstack([boss_centric(s[0]) for s in seqs[:2]]); te = seqs[2]
        F_tr, pca = features(tr_X); F_te, _ = features(boss_centric(te[0]), pca)
        mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-3
        F_tr, F_te = (F_tr - mu) / sd, (F_te - mu) / sd
        n_true = len(set(sum([s[1] for s in seqs], [])))
        K = 25
        hmm = GaussianHMM(n_components=K, covariance_type="diag", n_iter=60, random_state=0, min_covar=1e-2,
                          transmat_prior=1.0 + 50.0 * np.eye(K))     # sticky prior: syllables persist
        lens = [len(boss_centric(s[0])) for s in seqs[:2]]
        hmm.fit(F_tr, lens)
        z = hmm.predict(F_te)
        rows.append(dict(nmi_segment=nmi(te[1], z), nmi_move=nmi(te[2], z), boundary_f1=boundary_f1(z, te[1]),
                         n_true_segments=n_true, n_used=len(set(z)), T=len(z)))
    res[arch] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print(arch, {k: round(v, 3) for k, v in res[arch].items()}, flush=True)
json.dump(res, open("results/arena_perception_hmm.json", "w"), indent=1)
