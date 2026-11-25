"""Imagining what you don't know (review R10): a phantom move for open-world boss models.

The slow mind's programs only contain moves it has evidence for (plus the library prior). The
planner therefore never imagines an attack outside its model, and walks into unseen moves (E29: 11%
first-encounter wins on T3). Here every believed program is augmented, *at planning time only*,
with one generic "phantom" move: the Chinese-restaurant-process predictive mass of a move type not
yet in the model. With n move choices observed so far, the boss picks an unseen move type with
probability alpha / (n + alpha) (capped). The phantom is a generic in-reach attack with broad timing;
its cues are wildcards (any cue after the tell is equally likely), so the exact filter attributes
runs of unexplained cues to it, and the planner treats a phantom wind-up as a threat to respect.
"""
from __future__ import annotations

import numpy as np

from .boss import Boss, Move, N_DIST_BINS
from .primitives import CUE_ID, Primitive

PHANTOM = Primitive("phantom", "tell", 15, 3, 15, 2.5, 30.0)
PHANTOM_SIGMA = 0.5


def phantom_mass(n_seen: int, alpha: float, cap: float = 0.3) -> float:
    return float(min(cap, alpha / (n_seen + alpha)))


def with_phantom(boss: Boss, p_new: float) -> Boss:
    if p_new <= 0:
        return boss
    K = boss.K
    pi = np.zeros((K + 1, N_DIST_BINS, K + 1))
    pi[:K, :, :K] = boss.pi / boss.pi.sum(axis=2, keepdims=True) * (1 - p_new)
    pi[:K, :, K] = p_new
    pi[K] = pi[:K].mean(axis=0)            # after a phantom move the boss behaves like an average move
    laws = tuple(boss.laws)
    return Boss(list(boss.moves) + [Move(PHANTOM, PHANTOM.windup_median, PHANTOM_SIGMA)], pi, boss.phase2_speed,
                boss.tell_ticks, boss.cue_noise, boss.max_hp, dmg_scale=boss.dmg_scale, name=boss.name + "+phantom",
                laws=laws)


def count_moves(traces) -> int:
    """Move choices observed: starts of 'tell' runs in the cue streams."""
    tell = CUE_ID["tell"]
    n = 0
    for tr in traces:
        prev = None
        for (c, *_rest) in tr:
            n += c == tell and prev != tell
            prev = c
    return n
