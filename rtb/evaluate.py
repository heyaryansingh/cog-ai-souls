"""Parallel evaluation of agents on boss splits (4 worker processes by default).

Agents are named by a string spec so jobs pickle cleanly:
  "random", "dodge-all", "oracle", "prior-model", "generic-model"   (+ planner kwargs)
"""
from __future__ import annotations

import os
from functools import lru_cache
from multiprocessing import Pool

import numpy as np

from .boss import generic_boss, make_splits, prior_mean_boss, theta_star, SPREAD
from .env import Rules, T4_RULES
from .policies import DodgeEverything, OraclePlanner, RandomPolicy, oracle_continuation
from .run import fight


class StateCheat:
    """Sees the true boss latent state (move, stage, exact remaining wind-up) and plays the
    continuation heuristic. Upper bound that removes ALL uncertainty, not just model uncertainty."""
    def reset(self, env):
        pass

    def act(self, obs, env):
        return oracle_continuation(env)


@lru_cache(maxsize=4)
def _splits(seed: int):
    return make_splits(seed)


def make_agent(name: str, boss, seed: int, **kw):
    if name == "random":
        return RandomPolicy(seed)
    if name == "dodge-all":
        return DodgeEverything()
    if name == "cheat":
        return StateCheat()
    if name == "oracle":
        return OraclePlanner(seed=seed, **kw)
    if name == "prior-model":
        return OraclePlanner(seed=seed, model_boss=prior_mean_boss(boss, theta_star(SPREAD)), **kw)
    if name == "generic-model":
        return OraclePlanner(seed=seed, model_boss=generic_boss(theta_star(SPREAD), boss.tell_ticks,
                                                                 boss.cue_noise, boss.dmg_scale), **kw)
    raise ValueError(name)


def _job(args):
    split_seed, split, idx, agent, kw, seed = args
    kw = dict(kw)
    latency = kw.pop("latency", 0)
    boss = _splits(split_seed)[split][idx]
    rules = T4_RULES if split == "T4" else Rules()
    r = fight(boss, make_agent(agent, boss, seed, **kw), rules=rules, seed=seed, latency=latency)
    r.pop("log", None)
    r.pop("trace", None)
    r.update(split=split, boss=idx, agent=agent, seed=seed)
    return r


def evaluate(agent: str, split: str = "train", n_bosses: int = 10, attempts: int = 2,
             split_seed: int = 0, procs: int | None = None, **kw) -> list[dict]:
    jobs = [(split_seed, split, i, agent, kw, 1000 + 37 * i + a) for i in range(n_bosses) for a in range(attempts)]
    with Pool(procs or min(4, os.cpu_count() or 1)) as pool:
        return pool.map(_job, jobs)


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson interval for a win rate."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def summarize(res: list[dict]) -> str:
    w = np.mean([r["outcome"] == "win" for r in res])
    lo, hi = wilson(int(round(w * len(res))), len(res))
    d = np.mean([r["outcome"] == "death" for r in res])
    return (f"win {w:5.1%} [{lo:.0%}-{hi:.0%}]  death {d:5.1%}  boss-HP-left {np.mean([r['boss_hp_frac'] for r in res]):.2f}  "
            f"hp-left {np.mean([r['hp'] for r in res]):5.1f}  s/fight {np.mean([r['seconds'] for r in res]):.1f}  n={len(res)}")
