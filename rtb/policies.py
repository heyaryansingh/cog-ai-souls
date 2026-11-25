"""Simple reference policies: random, scripted 'dodge every wind-up', and the oracle planner."""
from __future__ import annotations

import numpy as np

from .env import (ACTIVE, APPROACH, BLOCK, DODGE, HEAL, HEAVY, IDLE, LIGHT, N_ACTIONS, RECOVERY,
                  RETREAT, WINDUP, MiniSouls)
from .filter import BossFilter
from .primitives import CUE_ID, GROUPS

WINDUP_CUES = {CUE_ID[g] for g in GROUPS} | {CUE_ID["tell"]}


class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self, env):
        pass

    def act(self, obs, env):
        return int(self.rng.integers(N_ACTIONS))


class DodgeEverything:
    """Floor baseline: approach, attack when close, dodge as soon as any group cue appears."""

    def reset(self, env):
        pass

    def act(self, obs, env):
        if obs["cue"] in WINDUP_CUES and obs["cue"] != CUE_ID["tell"]:
            return DODGE
        if obs["hp"] < 40 and obs["heals"] > 0 and obs["dist"] > 4:
            return HEAL
        if obs["dist"] > 2.0:
            return APPROACH
        return LIGHT


def oracle_continuation(env: MiniSouls) -> int:
    """Rollout policy inside a *sampled* world where the boss state is known. Used only in
    imagination by the planners (and, with the true state, as a 'cheat' upper bound).

    Priorities: survive the next impact (dodge if affordable and timeable, else retreat out of
    reach, else block), heal when safe and low, punish recovery while keeping a dodge in
    reserve, otherwise close distance when stamina allows."""
    R, bs, boss, p = env.R, env.bs, env.boss, env.player
    m = boss.moves[bs.k]
    prim = m.primitive
    impact = prim.active if prim.lunge else 1
    reach = prim.reach + prim.lunge
    threat = prim.damage > 0 and not prim.feint and not bs.hit_done and env.dist <= reach + 0.3
    can_dodge = p.stamina >= R.dodge_cost
    if bs.stage in (WINDUP, ACTIVE) and threat:
        if bs.stage == WINDUP:
            tti = (bs.D - bs.j + 1) + impact      # dodge tick that would meet the impact
        else:
            tti = impact - bs.j + 1
        if can_dodge and R.iframe_start <= tti <= R.iframe_start + R.iframe_len - 1:
            return DODGE
        # can we walk out of reach before impact?
        need = (reach - env.dist + 0.05) / R.move_speed
        if 0 < need <= tti - 1 and not (can_dodge and tti > R.iframe_start + R.iframe_len - 1):
            return RETREAT
        if tti > R.iframe_start + R.iframe_len - 1:
            return IDLE                            # wait for the dodge window
        return BLOCK if not prim.unblockable else RETREAT
    safe_ticks = 0
    if bs.stage == RECOVERY:
        safe_ticks = m.recovery - bs.j
    elif bs.stage == WINDUP and (prim.damage == 0 or prim.feint or env.dist > reach + 1.0):
        safe_ticks = bs.D - bs.j + (m.active + m.recovery if (prim.feint or prim.damage == 0) else 0)
    if p.hp < 45 and p.heals > 0 and (safe_ticks >= R.heal_ticks or env.dist > 5.0):
        return HEAL
    if env.dist <= R.light_reach and safe_ticks > 0:
        if safe_ticks >= R.heavy_hit_tick + 1 and p.stamina >= R.heavy_cost + R.dodge_cost:
            return HEAVY
        if safe_ticks >= R.light_hit_tick + 1 and p.stamina >= R.light_cost + R.dodge_cost:
            return LIGHT
    if safe_ticks > 0 and env.dist > R.light_reach and p.stamina >= R.light_cost + R.dodge_cost:
        need = (env.dist - R.light_reach + 0.05) / R.move_speed
        if need + R.light_hit_tick + 1 <= safe_ticks:
            return APPROACH
    if p.hp < 45 and p.heals > 0:
        return RETREAT
    if env.dist > 2.0 and p.stamina >= R.dodge_cost + 10:
        return APPROACH
    return IDLE


class OraclePlanner:
    """Model-predictive control with the TRUE boss program (the acting ceiling, M1).

    The current boss state is uncertain (cues are ambiguous and noisy), so it is sampled from
    the exact filter. For each candidate first action, it imagines `n_rollouts` futures of
    `horizon` ticks in which the boss follows its program and the player continues with
    `oracle_continuation`, and picks the action with the best mean return.
    """

    def __init__(self, n_rollouts=12, horizon=30, seed=0, actions=(IDLE, APPROACH, RETREAT, LIGHT, HEAVY, DODGE, BLOCK, HEAL),
                 model_boss=None):
        """model_boss: the boss program the agent BELIEVES in. None = the true one (oracle).
        Passing a wrong program shows how much a learner has to gain by inferring the boss."""
        self.n, self.h = n_rollouts, horizon
        self.model_boss = model_boss
        self.rng = np.random.default_rng(seed)
        self.actions = actions

    def reset(self, env: MiniSouls):
        self.model = self.model_boss or env.boss
        self.filter = BossFilter(self.model)

    def observe(self, obs):
        """Call after every env step (also while the player is busy)."""
        self.filter.update(obs["cue"])
        self.filter.predict(obs["dist"], obs["phase"])

    def _rollout(self, env: MiniSouls, state, first: int, seed: int) -> float:
        sim = env.clone(seed)
        if self.model is not env.boss:          # imagine with the believed program, not the real one
            sim.boss, sim._pmf = self.model, self.model.windup_pmf()
        k, stage, j, p0 = state
        pmf = sim._pmf[p0, k]
        D = j
        if stage == WINDUP:  # sample remaining wind-up given j ticks have elapsed (D >= j)
            tail = pmf[j - 1:]
            D = j + int(sim.rng.choice(len(tail), p=tail / tail.sum())) if tail.sum() > 0 else j
        sim.bs.k, sim.bs.stage, sim.bs.j, sim.bs.D, sim.bs.p0, sim.bs.hit_done = k, stage, j, D, p0, False
        total, a = 0.0, first
        for _ in range(self.h):
            _, r, done, _ = sim.step(a)
            total += r
            if done:
                break
            a = oracle_continuation(sim) if sim.player_free() else IDLE
        return total

    def act(self, obs, env: MiniSouls) -> int:
        # The filter's belief is over the state that will emit the NEXT cue, i.e. env.bs now.
        states = self.filter.sample_states(self.n, self.rng)
        best, best_v = IDLE, -np.inf
        seeds = self.rng.integers(1 << 30, size=self.n)
        for a in self.actions:
            v = np.mean([self._rollout(env, s, a, int(sd)) for s, sd in zip(states, seeds)])
            if v > best_v:
                best, best_v = a, v
        return best
