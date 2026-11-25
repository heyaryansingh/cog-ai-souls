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


def _cond_median_D(env, bs):
    """Median remaining wind-up length given j ticks elapsed: what an observer could predict."""
    tail = env._pmf[bs.p0, bs.k][bs.j - 1:]
    if tail.sum() <= 0:
        return bs.j
    c = np.cumsum(tail) / tail.sum()
    return bs.j + int(np.searchsorted(c, 0.5))


def oracle_continuation(env: MiniSouls, humble: bool = False, latency: int = 0) -> int:
    """Rollout policy inside a *sampled* world where the boss state is known. Used only in
    imagination by the planners (and, with the true state, as a 'cheat' upper bound).

    Priorities: survive the next impact (dodge if affordable and timeable, else retreat out of
    reach, else block), heal when safe and low, punish recovery while keeping a dodge in
    reserve, otherwise close distance when stamina allows."""
    R, bs, boss, p = env.R, env.bs, env.boss, env.player
    m = boss.moves[bs.k]
    D = bs.D
    D_dodge = D
    if humble == "dodge" and bs.stage == WINDUP:
        # only the *dodge timing* of the imagined player is limited to prediction: it cannot defer a
        # roll and then time it perfectly, which made waiting look free (Entry 24)
        D_dodge = _cond_median_D(env, bs)
    elif humble and bs.stage == WINDUP:
        # an imagined player with the agent's own limits: it cannot read the move before the tell
        # (plus perception delay) is over, and it predicts the wind-up end, it does not know it
        if bs.j <= boss.tell_ticks + latency and m.primitive.name != "idle":
            return IDLE if env.dist <= 4.0 else APPROACH
        D = D_dodge = _cond_median_D(env, bs)
    prim = m.primitive
    impact = prim.active if prim.lunge else 1
    reach = prim.reach + prim.lunge
    threat = prim.damage > 0 and not prim.feint and not bs.hit_done and env.dist <= reach + 0.3
    can_dodge = p.stamina >= R.dodge_cost
    if bs.stage in (WINDUP, ACTIVE) and threat:
        if bs.stage == WINDUP:
            tti = (D_dodge - bs.j + 1) + impact      # dodge tick that would meet the impact
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
        safe_ticks = D - bs.j + (m.active + m.recovery if (prim.feint or prim.damage == 0) else 0)
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
                 model_boss=None, utility="linear", humble=False, state_map=False):
        """utility: "linear" = env reward (boss damage - 2 x player damage, +-100 win/death);
        "concave" = boss damage + value(hp_end) - value(hp_start), with value concave in HP, so losing
        HP when low is worse and heals are worth more when low.
        model_boss: the boss program the agent BELIEVES in. None = the true one (oracle).
        Passing a wrong program shows how much a learner has to gain by inferring the boss."""
        self.n, self.h = n_rollouts, horizon
        self.humble = humble
        self.state_map = state_map      # lesion: plan from the single most probable state (no state averaging)
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
            a = oracle_continuation(sim, self.humble, self.latency) if sim.player_free() else IDLE
        rb = getattr(self, "rules_belief", None)
        if rb is not None and first == DODGE and self.info_weight > 0:
            # active self-identification: value the information a roll-timed impact gives about i-frames
            for (dt, _hit) in sim.events:
                if dt > 0:
                    total += self.info_weight * rb.info_gain(dt)
                    break
        if self.utility == "concave":
            v = lambda h: 60.0 * np.log1p(max(h, 0.0) / 20.0)
            total = (boss0 - max(sim.boss_hp, 0.0)) + v(sim.player.hp) - v(hp0)
            total += 150.0 if sim.outcome == "win" else (-150.0 if sim.outcome == "death" else 0.0)
        return total

    @staticmethod
    def _state_from_index(f, i):
        nW, nA = f.W.size, f.A.size
        if i < nW:
            p, k, d = np.unravel_index(i, f.W.shape)
            return (int(k), 0, int(d) + 1, int(p))
        if i < nW + nA:
            k, j = np.unravel_index(i - nW, f.A.shape)
            return (int(k), 1, int(j) + 1, 0)
        k, j = np.unravel_index(i - nW - nA, f.R.shape)
        return (int(k), 2, int(j) + 1, 0)

    def projected(self, f):
        """The filter's belief is over the state that will emit the NEXT cue, i.e. env.bs now. With
        perception latency the perceived stream lags reality: push the belief forward blind."""
        if self.latency and self._last is not None:
            import copy
            f = copy.copy(f)
            f.W, f.A, f.R = f.W.copy(), f.A.copy(), f.R.copy()
            for _ in range(self.latency):
                f.predict(self._last["dist"], self._last["phase"], self._last["ctx"])
        return f

    def act(self, obs, env: MiniSouls) -> int:
        q = self.q_values(obs, env)
        return self.actions[int(np.argmax(q))]

    def q_values(self, obs, env: MiniSouls) -> np.ndarray:
        """Mean imagined return of each candidate first action (in self.actions order)."""
        f = self.projected(self.filter)
        states = f.sample_states(self.n, self.rng)
        if self.state_map:
            flat = np.concatenate([f.W.ravel(), f.A.ravel(), f.R.ravel()])
            i = int(np.argmax(flat))
            states = [self._state_from_index(f, i)] * self.n
        seeds = self.rng.integers(1 << 30, size=self.n)
        return np.array([np.mean([self._rollout(env, s, a, int(sd)) for s, sd in zip(states, seeds)])
                         for a in self.actions])


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


class OnlineMixturePlanner(MixturePlanner):
    """MixturePlanner whose programs are re-learned *during* the fight: every `every` ticks the
    slow mind rejuvenates its particles on past fights plus the fight so far, and the filters
    are rebuilt by replaying this fight under the new programs. (Simulated time stops while it
    thinks; the compute is about 1 s per refresh.)"""

    def __init__(self, slow_mind, every=200, sweeps=1, **kw):
        super().__init__(slow_mind.particles, slow_mind.weights, **kw)
        self.sm, self.every, self.sweeps = slow_mind, every, sweeps
        self.partial = []
        self.refreshes = 0

    def observe(self, obs):
        super().observe(obs)
        self.partial.append((obs["cue"], obs["dist"], obs["phase"], obs["ctx"]))
        if len(self.partial) % self.every == 0:
            parts, w = self.sm.posterior_with(list(self.partial), self.sweeps)
            self.bosses = parts
            self.prior_logw = np.log(np.maximum(w, 1e-300))
            self.filters = []
            for b in parts:
                f = BossFilter(b)
                for c, d, p, x in self.partial:
                    f.update(c)
                    f.predict(d, p, x)
                self.filters.append(f)
            self.refreshes += 1
