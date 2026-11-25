"""Run fights and attempts; produce a compact text log of each fight."""
from __future__ import annotations

import time

from .env import ACTION_NAMES, MiniSouls, Rules
from .primitives import CUES


def fight(boss, policy, rules: Rules = Rules(), seed=0, log_every: int = 0, max_ticks=3600):
    env = MiniSouls(boss, rules, seed=seed, max_ticks=max_ticks)
    obs = env.reset()
    policy.reset(env)
    lines, t0 = [], time.time()
    decisions = 0
    while True:
        if obs["free"]:
            a = policy.act(obs, env)
            decisions += 1
        else:
            a = 0
        obs, _, done, info = env.step(a)
        if hasattr(policy, "observe"):
            policy.observe(obs)
        if log_every and env.t % log_every == 0:
            lines.append(f"t={env.t:4d} cue={CUES[obs['cue']]:10s} dist={obs['dist']:.2f} "
                         f"hp={obs['hp']:5.1f} st={obs['stamina']:5.1f} boss={obs['boss_hp']:5.1f} "
                         f"act={ACTION_NAMES[env.player.action]}")
        if done:
            break
    return {"outcome": info["outcome"], "ticks": env.t, "boss_hp_frac": max(env.boss_hp, 0) / boss.max_hp,
            "hp": max(env.player.hp, 0), "seconds": time.time() - t0, "decisions": decisions, "log": lines}
