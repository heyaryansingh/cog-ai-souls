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
                 model_boss=None, utility="linear"):
        """utility: "linear" = env reward (boss damage - 2 x player damage, +-100 win/death);
        "concave" = boss damage + value(hp_end) - value(hp_start), with value concave in HP, so losing
        HP when low is worse and heals are worth more when low.
        model_boss: the boss program the agent BELIEVES in. None = the true one (oracle).
        Passing a wrong program shows how much a learner has to gain by inferring the boss."""
        self.n, self.h = n_rollouts, horizon
        self.model_boss = model_boss
        self.utility = utility
        self.latency = 0          # set by run.fight; belief is then projected `latency` ticks ahead
        self._last = None
        self.rng = np.random.default_rng(seed)
        self.actions = actions

    def reset(self, env: MiniSouls):
        self.model = self.model_boss or env.boss
        self.filter = BossFilter(self.model)

    def observe(self, obs):
        """Call after every env step (also while the player is busy)."""
        self.filter.update(obs["cue"])
        self.filter.predict(obs["dist"], obs["phase"], obs["ctx"])
        self._last = obs

    def _rollout(self, env: MiniSouls, state, first: int, seed: int, model=None) -> float:
        sim = env.clone(seed)
        model = model or self.model
        if model is not env.boss:          # imagine with the believed program, not the real one
            sim.boss, sim._pmf = model, model.windup_pmf()
        if getattr(self, "believed_rules", None) is not None:   # imagine with the believed body (self-model)
            sim.R = self.believed_rules
        k, stage, j, p0 = state
        pmf = sim._pmf[p0, k]
        D = j
        if stage == WINDUP:  # sample remaining wind-up given j ticks have elapsed (D >= j)
            tail = pmf[j - 1:]
            D = j + int(sim.rng.choice(len(tail), p=tail / tail.sum())) if tail.sum() > 0 else j
        sim.bs.k, sim.bs.stage, sim.bs.j, sim.bs.D, sim.bs.p0, sim.bs.hit_done = k, stage, j, D, p0, False
        total, a = 0.0, first
        hp0, boss0 = sim.player.hp, sim.boss_hp
        for _ in range(self.h):
            _, r, done, _ = sim.step(a)
            total += r
            if done:
                break
            a = oracle_continuation(sim) if sim.player_free() else IDLE
        if self.utility == "concave":
            v = lambda h: 60.0 * np.log1p(max(h, 0.0) / 20.0)
            total = (boss0 - max(sim.boss_hp, 0.0)) + v(sim.player.hp) - v(hp0)
            total += 150.0 if sim.outcome == "win" else (-150.0 if sim.outcome == "death" else 0.0)
        return total

    def act(self, obs, env: MiniSouls) -> int:
        # The filter's belief is over the state that will emit the NEXT cue, i.e. env.bs now.
        f = self.filter
        if self.latency and self._last is not None:
            # perceived stream lags reality: push the belief forward without observations
            import copy
            f = copy.copy(f)
            f.W, f.A, f.R = f.W.copy(), f.A.copy(), f.R.copy()
            for _ in range(self.latency):
                f.predict(self._last["dist"], self._last["phase"], self._last["ctx"])
        states = f.sample_states(self.n, self.rng)
        best, best_v = IDLE, -np.inf
        seeds = self.rng.integers(1 << 30, size=self.n)
        for a in self.actions:
            v = np.mean([self._rollout(env, s, a, int(sd)) for s, sd in zip(states, seeds)])
            if v > best_v:
                best, best_v = a, v
        return best


class MixturePlanner(OraclePlanner):
    """MPC over a weighted set of boss programs (the slow mind's particles).

    One exact filter runs per program. At decision time the program posterior is
    prior weight x in-fight evidence; each imagined rollout first samples a program, then a
    state from that program's filter, and imagines the future under that program
    (Bayesian model averaging). mode="map" keeps only the single best program (MAP lesion).
    """

    def __init__(self, bosses, weights, mode="mixture", **kw):
        super().__init__(**kw)
        self.bosses = list(bosses)
        self.prior_logw = np.log(np.maximum(np.asarray(weights, float), 1e-300))
        self.mode = mode

    def reset(self, env: MiniSouls):
        if self.mode == "map":
            i = int(np.argmax(self.prior_logw))
            self.bosses, self.prior_logw = [self.bosses[i]], np.zeros(1)
        self.filters = [BossFilter(b) for b in self.bosses]
        self.model = self.bosses[0]

    def observe(self, obs):
        for f in self.filters:
            f.update(obs["cue"])
            f.predict(obs["dist"], obs["phase"], obs["ctx"])
        self._last = obs

    def act(self, obs, env: MiniSouls) -> int:
        import copy
        lw = self.prior_logw + np.array([f.log_evidence for f in self.filters])
        w = np.exp(lw - lw.max())
        w /= w.sum()
        which = self.rng.choice(len(self.filters), size=self.n, p=w)
        states, models = [], []
        for i in np.unique(which):
            f = self.filters[i]
            if self.latency and self._last is not None:
                f = copy.copy(f)
                f.W, f.A, f.R = f.W.copy(), f.A.copy(), f.R.copy()
                for _ in range(self.latency):
                    f.predict(self._last["dist"], self._last["phase"], self._last["ctx"])
            n_i = int((which == i).sum())
            states += f.sample_states(n_i, self.rng)
            models += [self.bosses[i]] * n_i
        best, best_v = IDLE, -np.inf
        seeds = self.rng.integers(1 << 30, size=self.n)
        for a in self.actions:
            v = np.mean([self._rollout(env, s, a, int(sd), m) for s, m, sd in zip(states, models, seeds)])
            if v > best_v:
                best, best_v = a, v
        return best
