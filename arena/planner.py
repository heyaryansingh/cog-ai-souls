"""Model-predictive control on Arena (the acting ceiling, and later the agent's deliberative system).

For each candidate first action, imagine `n` futures of `horizon` ticks in a cloned world and pick
the best mean return. Futures differ in the parts of the boss's state an observer cannot know: the
remaining length of the current hold, the next move choices, hit timing. The imagined player
continues with a simple reactive policy (`continuation`) that sees the imagined state.
"""
from __future__ import annotations

import numpy as np

from .env import (Arena, GUARD, HEAL, HEAVY, JUMP, LIGHT, N_BUTTONS, NONE, ROLL, BOSS_R)

# candidate first actions: (direction, button)
CANDIDATES = [(0, NONE), (1, NONE), (5, NONE), (3, NONE), (7, NONE),            # stay, in, out, strafe L/R
              (5, ROLL), (3, ROLL), (7, ROLL), (1, ROLL), (4, ROLL), (6, ROLL), # roll back / left / right / fwd / diag
              (0, LIGHT), (0, HEAVY), (0, GUARD), (0, JUMP), (0, HEAL)]
CAND = [d * N_BUTTONS + b for d, b in CANDIDATES]


def _ticks_to_impact(env):
    """Ticks until the current move's next impact (expected for holds), or None."""
    b, spec = env.boss, env.spec
    if b.move < 0 or b.stagger:
        return None
    m = spec.moves[b.move]
    s = m.segments[b.seg]
    if s.kind == "strike" and s.hitbox is not None and s.hitbox.shape != "projectile" and b.t <= s.impact:
        return s.impact - b.t + 1
    left = b.seg_len - b.t
    for ss in m.segments[b.seg + 1:]:
        if ss.kind == "strike" and ss.hitbox is not None and ss.hitbox.shape != "projectile":
            return left + ss.impact + 1
        left += ss.ticks if ss.kind != "hold" else ss.hold_median
    return None


def continuation(env: Arena, rng) -> int:
    p, b = env.player, env.boss
    dist = np.linalg.norm(b.pos - p.pos) - BOSS_R
    tti = _ticks_to_impact(env)
    B = env.body
    if tti is not None and tti <= B.iframe_start + 2 and dist < 6.0 and p.stamina >= B.roll_cost:
        return int(rng.choice([3, 7, 5])) * N_BUTTONS + ROLL
    for pr in env.projectiles:
        if np.linalg.norm(pr.pos - p.pos) < 2.0 and p.stamina >= B.roll_cost:
            return int(rng.choice([3, 7])) * N_BUTTONS + ROLL
    if b.stagger > 0 and dist < B.heavy_reach:
        return HEAVY
    recovering = b.move >= 0 and env.spec.moves[b.move].segments[b.seg].kind == "recover"
    if (recovering or b.move < 0) and dist < B.light_reach and p.stamina > B.light_cost + B.roll_cost:
        return LIGHT
    if p.hp < 45 and p.flasks and dist > 5.0:
        return HEAL
    if dist > 1.8 and (tti is None or tti > 12):
        return 1 * N_BUTTONS + NONE
    return NONE


def imagine(env: Arena, seed) -> Arena:
    """Clone the world and resample what an observer could not know: the remaining hold length."""
    sim = env.clone(seed)
    b = sim.boss
    if b.move >= 0 and not b.stagger:
        s = sim.spec.moves[b.move].segments[b.seg]
        if s.kind == "hold":
            med = s.hold_median * (sim.spec.phase2_speed if sim.phase else 1.0)
            for _ in range(20):
                L = int(np.clip(round(sim.rng.lognormal(np.log(max(med, 1.0)), s.hold_sigma)), 1, 120))
                if L >= b.t:
                    break
            b.seg_len = max(L, b.t)
    return sim


class ArenaPlanner:
    def __init__(self, n=6, horizon=24, seed=0, candidates=CAND):
        self.n, self.h, self.cands = n, horizon, list(candidates)
        self.rng = np.random.default_rng(seed)

    def q_values(self, env: Arena) -> np.ndarray:
        seeds = self.rng.integers(1 << 30, size=self.n)
        q = np.zeros(len(self.cands))
        for ci, a0 in enumerate(self.cands):
            tot = 0.0
            for sd in seeds:
                sim = imagine(env, int(sd))
                crng = np.random.default_rng(int(sd) + 1)
                a = a0
                for _ in range(self.h):
                    _, r, done, _ = sim.step(a)
                    tot += r
                    if done:
                        break
                    a = continuation(sim, crng) if sim.player_free() else 0
            q[ci] = tot / self.n
        return q

    def __call__(self, obs, env: Arena) -> int:
        return self.cands[int(np.argmax(self.q_values(env)))]
