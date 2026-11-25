"""Reference players for Arena3D: random, aggressive, and an oracle dodger that peeks into the true future."""
from __future__ import annotations

import numpy as np

from .env import GUARD, HEAL, LIGHT, N_ACTIONS, N_BUTTONS, NONE, ROLL


def random_policy(env, rng):
    return int(rng.integers(N_ACTIONS))


def aggro(env, rng):
    dist = np.linalg.norm(env.boss.root - env.p_xy)
    if dist > 0.8 + 0.25 * env.spec.scale:
        return 1 * N_BUTTONS + NONE
    return LIGHT


def _hit_soon(env, a, horizon, seed):
    sim = env.clone(seed)
    for k in range(horizon):
        _, _, done, info = sim.step(a if k == 0 else (0 if a % N_BUTTONS != NONE else a))
        if info["hit"] is not None or (done and sim.hp <= 0):
            return True
        if done:
            return False
    return False


def oracle_dodger(env, rng, horizon=14):
    """Cheats: simulates the true boss future (same RNG) to decide when and where to roll."""
    if not env.player_free():
        return 0
    seed = None   # same RNG stream as the real boss -> exact future
    if _hit_soon(env, NONE, horizon, seed):
        for d in rng.permutation([5, 3, 7, 4, 6, 2, 8]):
            a = int(d) * N_BUTTONS + ROLL
            if not _hit_soon(env, a, horizon, seed):
                return a
        return 5 * N_BUTTONS + ROLL
    if env.hp < 45 and env.flasks and np.linalg.norm(env.boss.root - env.p_xy) > 4:
        return HEAL
    dist = np.linalg.norm(env.boss.root - env.p_xy)
    if dist > 0.8 + 0.25 * env.spec.scale:
        return 1 * N_BUTTONS + NONE
    if not _hit_soon(env, LIGHT, 24, seed):
        return LIGHT
    return 5 * N_BUTTONS + NONE
