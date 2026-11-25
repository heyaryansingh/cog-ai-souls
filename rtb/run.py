"""Run fights and attempts; produce a compact text log of each fight."""
from __future__ import annotations

import time

from .env import ACTION_NAMES, MiniSouls, Rules
from .primitives import CUES


def fight(boss, policy, rules: Rules = Rules(), seed=0, log_every: int = 0, max_ticks=3600,
          latency: int = 0):
    """latency: the agent perceives boss cues `latency` ticks late (human visual-motor delay is
    ~4-5 ticks at 20 Hz). Its own body state is always current."""
    from collections import deque
    env = MiniSouls(boss, rules, seed=seed, max_ticks=max_ticks)
    obs = env.reset()
    policy.reset(env)
    if hasattr(policy, "latency"):
        policy.latency = latency
    pending = deque()
    lines, t0 = [], time.time()
    decisions = 0
    trace = []      # full (undelayed) boss-observation stream, for learning between fights
    while True:
        if obs["free"]:
            a = policy.act(obs, env)
            decisions += 1
        else:
            a = 0
        obs, _, done, info = env.step(a)
        pending.append(obs)
        trace.append((obs["cue"], obs["dist"], obs["phase"], obs["ctx"]))
        if hasattr(policy, "observe") and len(pending) > latency:
            policy.observe(pending.popleft())
        if latency:   # what the agent sees now: its own state, the boss cue from `latency` ticks ago
            seen = pending[0] if len(pending) >= latency else None
            obs = dict(obs, cue=seen["cue"] if seen else None)
        if log_every and env.t % log_every == 0:
            lines.append(f"t={env.t:4d} cue={CUES[obs['cue']]:10s} dist={obs['dist']:.2f} "
                         f"hp={obs['hp']:5.1f} st={obs['stamina']:5.1f} boss={obs['boss_hp']:5.1f} "
                         f"act={ACTION_NAMES[env.player.action]}")
        if done:
            break
    return {"outcome": info["outcome"], "ticks": env.t, "boss_hp_frac": max(env.boss_hp, 0) / boss.max_hp,
            "hp": max(env.player.hp, 0), "seconds": time.time() - t0, "decisions": decisions, "log": lines, "trace": trace}
