"""The exact filter must match brute-force enumeration of every latent path.

The enumeration uses the *simulator's* representation (sample the whole wind-up length D
at move start, then step deterministically) rather than the filter's hazard representation,
so the two are independent implementations of the same model.
"""
import numpy as np

from rtb.boss import Boss, Move, D_MAX, dist_bin
from rtb.env import MiniSouls
from rtb.filter import BossFilter
from rtb.primitives import Primitive, N_CUES, RECOVER_CUE, active_cue, windup_cue
from rtb.boss import make_splits


def tiny_boss(seed=0) -> Boss:
    rng = np.random.default_rng(seed)
    prims = [Primitive("idle", "idle", 2, 0, 0, 0, 0),
             Primitive("slam", "overhead", 2, 1, 2, 2.5, 30),
             Primitive("delayed_slam", "overhead", 3, 1, 1, 2.5, 30),
             Primitive("feint", "thrust", 2, 2, 0, 0, 0, feint=True)]
    moves = [Move(p, p.windup_median, 0.3) for p in prims]
    K = len(moves)
    pi = rng.dirichlet(np.ones(K), size=(K, 2))
    b = Boss(moves, pi, phase2_speed=0.8, tell_ticks=1, cue_noise=0.1)
    pmf = np.zeros((2, K, D_MAX))
    for p in range(2):
        for k in range(K):
            w = rng.dirichlet(np.ones(3))
            pmf[p, k, :3] = w          # support on D in {1, 2, 3}
    b._pmf = pmf
    return b


def true_cue(boss, k, stage, j):
    m = boss.moves[k].primitive
    if stage == 0:
        return windup_cue(m, j, boss.tell_ticks)
    if stage == 1:
        return active_cue(m)
    return RECOVER_CUE


def brute_force(boss, cues, dists, phases):
    """Posterior over (k, stage, j) of the state that emitted the last cue."""
    pmf = boss.windup_pmf()
    e, V = boss.cue_noise, N_CUES
    T = len(cues)
    post = {}

    def emit(c, k, s, j):
        return (1 - e + e / V) if c == true_cue(boss, k, s, j) else e / V

    def walk(t, k, s, j, D, prob):
        prob *= emit(cues[t], k, s, j)
        if prob == 0:
            return
        if t == T - 1:
            post[(k, s, j)] = post.get((k, s, j), 0.0) + prob
            return
        m = boss.moves[k]
        # deterministic progression given D
        if s == 0 and j < D:
            return walk(t + 1, k, 0, j + 1, D, prob)
        if s == 0:
            nxt = 1 if m.active > 0 else (2 if m.recovery > 0 else None)
        elif s == 1 and j < m.active:
            return walk(t + 1, k, 1, j + 1, D, prob)
        elif s == 1:
            nxt = 2 if m.recovery > 0 else None
        elif j < m.recovery:
            return walk(t + 1, k, 2, j + 1, D, prob)
        else:
            nxt = None
        if nxt is not None:
            return walk(t + 1, k, nxt, 1, D, prob)
        b, ph = dist_bin(dists[t]), phases[t]
        for k2 in range(boss.K):
            for D2 in range(1, D_MAX + 1):
                w = boss.pi[k, b, k2] * pmf[ph, k2, D2 - 1]
                if w > 0:
                    walk(t + 1, k2, 0, 1, D2, prob * w)

    for D0 in range(1, D_MAX + 1):
        if pmf[0, 0, D0 - 1] > 0:
            walk(0, 0, 0, 1, D0, pmf[0, 0, D0 - 1])
    z = sum(post.values())
    return {key: v / z for key, v in post.items()}, np.log(z)


def filter_marginal(f: BossFilter):
    out = {}
    for p in range(2):
        for k in range(f.K):
            for d in range(D_MAX):
                if f.W[p, k, d] > 0:
                    out[(k, 0, d + 1)] = out.get((k, 0, d + 1), 0) + f.W[p, k, d]
    for k in range(f.K):
        for j in range(f.Amax):
            if f.A[k, j] > 0:
                out[(k, 1, j + 1)] = f.A[k, j]
        for j in range(f.Rmax):
            if f.R[k, j] > 0:
                out[(k, 2, j + 1)] = f.R[k, j]
    return out


def test_filter_matches_brute_force():
    for seed in range(5):
        boss = tiny_boss(seed)
        rng = np.random.default_rng(100 + seed)
        T = 9
        cues = list(rng.integers(0, N_CUES, size=T))
        # bias cues toward plausible ones so the posterior is non-trivial
        for t in range(T):
            if rng.random() < 0.7:
                k = int(rng.integers(boss.K))
                cues[t] = true_cue(boss, k, int(rng.integers(0, 3)) if k else 0, 1)
        dists = list(rng.uniform(1, 5, size=T))
        phases = [0] * (T // 2) + [1] * (T - T // 2)
        bf, logz = brute_force(boss, cues, dists, phases)

        f = BossFilter(boss)
        for t in range(T):
            f.update(cues[t])
            if t < T - 1:
                f.predict(dists[t], phases[t])
        fm = filter_marginal(f)
        keys = set(bf) | set(fm)
        tv = 0.5 * sum(abs(bf.get(k, 0) - fm.get(k, 0)) for k in keys)
        assert tv < 1e-9, (seed, tv)
        assert abs(f.log_evidence - logz) < 1e-9, (seed, f.log_evidence, logz)


def test_filter_tracks_real_fights():
    """On full bosses in the simulator, the filter puts most mass on the true move and stage."""
    splits = make_splits(seed=1, n_train=5, n_test=1)
    hits = []
    for boss in splits["train"][:3]:
        env = MiniSouls(boss, seed=3)
        f = BossFilter(boss)
        rng = np.random.default_rng(0)
        for _ in range(600):
            k, s, j, p0 = env.truth()
            obs, _, done, _ = env.step(int(rng.choice([0, 1, 2, 5])))
            f.update(obs["cue"])
            mass = {0: f.W[:, k].sum(), 1: f.A[k].sum(), 2: f.R[k].sum()}[s]
            hits.append(mass)
            f.predict(obs["dist"], obs["phase"])
            if done:
                break
    assert np.mean(hits) > 0.6, np.mean(hits)
