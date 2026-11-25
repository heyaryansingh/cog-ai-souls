"""Reference players for sanity checks."""
import numpy as np
from .env import N_BUTTONS, ROLL, LIGHT, HEAVY, GUARD, HEAL, NONE


def act(d, b):
    return d * N_BUTTONS + b


class Random:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs, env):
        return int(self.rng.integers(63))


class Aggro:
    """Walk in and mash light attacks, heal when low. Ignores the boss."""
    def __call__(self, obs, env):
        p, b = env.player, env.boss
        dist = np.linalg.norm(b.pos - p.pos) - 0.6
        if p.hp < 40 and p.flasks and dist > 4:
            return act(0, HEAL)
        if dist > 1.8:
            return act(1, NONE)
        return act(0, LIGHT)


class CheatDodger:
    """Sees the boss's true state: rolls sideways just before impacts it is inside, punishes recoveries.
    A crude upper reference (it knows the future, but not the best response)."""
    def __call__(self, obs, env):
        p, b, spec = env.player, env.boss, env.spec
        dist = np.linalg.norm(b.pos - p.pos) - 0.6
        threat = False
        if b.move >= 0:
            m = spec.moves[b.move]
            s = m.segments[b.seg]
            # ticks until the next impact in this move
            left = b.seg_len - b.t
            k = b.seg + 1
            if s.kind == "strike" and s.hitbox is not None and b.t <= s.impact:
                threat, tti = True, s.impact - b.t + 1
            while not threat and k < len(m.segments):
                ss = m.segments[k]
                if ss.kind == "strike" and ss.hitbox is not None:
                    threat, tti = True, left + ss.impact + 1
                    break
                left += ss.ticks if ss.kind != "hold" else ss.hold_median
                k += 1
            if threat and tti <= 3 and dist < 6 and p.stamina >= env.body.roll_cost:
                return act(3, ROLL)          # roll left-back... direction 3 = back-left
        if b.stagger > 0 and dist < 2.4:
            return act(0, HEAVY)
        if b.move >= 0 and spec.moves[b.move].segments[b.seg].kind == "recover" and dist < 2.2 and p.stamina > 30:
            return act(0, LIGHT)
        if p.hp < 50 and p.flasks and dist > 5:
            return act(0, HEAL)
        if dist > 2.0:
            return act(1, NONE)
        return act(5, NONE) if p.stamina < 40 else act(0, NONE)
