"""Slow-mind checks: FFBS recovers the true latent path when the program is known, and a few
fights of learning move held-out evidence most of the way from the generic prior to the truth."""
import numpy as np

from rtb.boss import make_splits
from rtb.env import MiniSouls
from rtb.learner import SlowMind, evidence, ffbs, path_stats
from rtb.policies import oracle_continuation


def _trace(boss, seed, with_truth=False):
    env = MiniSouls(boss, seed=seed)
    obs = env.reset()
    tr, truth = [], []
    while True:
        truth.append(env.truth())
        a = oracle_continuation(env) if obs["free"] else 0
        obs, _, done, _ = env.step(a)
        tr.append((obs["cue"], obs["dist"], obs["phase"], obs["ctx"]))
        if done:
            return (tr, truth) if with_truth else tr


def test_ffbs_recovers_true_moves():
    boss = make_splits(0)["train"][0]
    tr, truth = _trace(boss, 3, with_truth=True)
    path, ev = ffbs(boss, tr, np.random.default_rng(0))
    mv = lambda s: s[2] if s[0] == "W" else s[1]
    acc = np.mean([mv(p) == t[0] for p, t in zip(path, truth)])
    assert acc > 0.95
    assert abs(ev - evidence(boss, [tr])) < 1e-6
    windups, trans = path_stats(path, tr)
    assert len(trans) == len(windups) - 1


def test_slow_mind_learns():
    boss = make_splits(0)["train"][0]
    test = [_trace(boss, 100 + i) for i in range(2)]
    sm = SlowMind(boss, n_particles=2, sweeps=2, seed=0)
    ev0 = evidence(sm.particles[0], test)
    for i in range(3):
        sm.update(_trace(boss, i))
    ev_true, ev_learned = evidence(boss, test), evidence(sm.map_particle(), test)
    assert (ev_learned - ev0) > 0.7 * (ev_true - ev0)
