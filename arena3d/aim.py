"""Attack verification, the way animators tune attacks: keep only strikes that actually connect with a
standing, unguarded player when aimed (3D contact check in the real world model). Real souls attacks are
built to hit you unless you respond; random mocap swings mostly are not."""
from __future__ import annotations

import copy

import numpy as np

from .boss import BossSpec


def verify_strikes(spec: BossSpec, units, clips, candidates, need, rng, dists=(None,), max_checks=120):
    from .env import Arena3D
    test = copy.copy(spec)
    test.strikes = [candidates[0]]
    test.combo = np.ones((2, 1))
    test.p_end = np.ones(1)
    test.speed = np.ones(1)
    test.hold_med = np.zeros(1)
    test.aggression, test.p_recover, test.idles = 1.0, 0.0, []
    env = Arena3D(test, units, clips, seed=0)
    ok = []
    for n, ui in enumerate(candidates[:max_checks]):
        test.strikes = [ui]
        env.spec = test
        env.reset()
        # put the boss at a typical engagement distance and restart its move with the player in view
        env.boss.reset(np.array([0.0, 1.2 + 0.9 * spec.scale]), -np.pi / 2)
        qb = env.boss.step(env.p_xy, False)
        env.prev_qpos[:63] = qb
        env.hit_lock = {}
        hit = False
        for _ in range(3 * units[ui].length + 10):
            _, _, done, info = env.step(0)
            if info["hit"] is not None:
                hit = True
                break
            if env.boss.kind != "strike" and env.t > 5:
                break
        if hit:
            ok.append(ui)
            if len(ok) >= need:
                break
    return ok
