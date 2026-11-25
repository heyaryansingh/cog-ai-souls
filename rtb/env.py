"""MiniSouls: a one-lane souls-like boss arena, 20 ticks per second.

Per tick, in this order:
  1. the player's current action advances (movement, i-frames, attack hits, heals);
  2. the boss emits its cue and applies its effects (approach, lunge, hit check);
  3. the observation for this tick is produced;
  4. the boss latent state transitions (next move chosen with this tick's observed
     distance bin and phase).

Steps 2 and 4 follow `Boss` exactly, so the boss's latent process given the observed
distance/phase stream is the HSMM the filter assumes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .boss import Boss, ctx_of, dist_bin
from .primitives import N_CUES, RECOVER_CUE, active_cue, windup_cue

# Player actions
IDLE, APPROACH, RETREAT, LIGHT, HEAVY, DODGE, BLOCK, HEAL = range(8)
ACTION_NAMES = ["idle", "approach", "retreat", "light", "heavy", "dodge", "block", "heal"]
N_ACTIONS = 8

# Boss stages
WINDUP, ACTIVE, RECOVERY = 0, 1, 2

MIN_DIST, MAX_DIST = 0.5, 8.0


@dataclass(frozen=True)
class Rules:
    """Player body rules R. Test split T4 changes these."""
    dodge_ticks: int = 10
    iframe_start: int = 2       # first i-frame tick of a dodge (1-based)
    iframe_len: int = 4
    dodge_cost: float = 15.0
    dodge_speed: float = 0.04   # small backward drift per tick while rolling
    light_ticks: int = 8
    light_hit_tick: int = 4
    light_dmg: float = 12.0
    light_cost: float = 12.0
    light_reach: float = 2.2
    heavy_ticks: int = 16
    heavy_hit_tick: int = 10
    heavy_dmg: float = 30.0
    heavy_cost: float = 28.0
    heavy_reach: float = 2.5
    move_ticks: int = 4
    move_speed: float = 0.15
    block_ticks: int = 6
    block_dmg_frac: float = 0.25
    heal_ticks: int = 30
    heal_amount: float = 40.0
    heal_charges: int = 2
    stamina_regen: float = 1.0
    max_stamina: float = 100.0
    max_hp: float = 100.0


T4_RULES = Rules(iframe_len=3, iframe_start=1, dodge_cost=22.0, stamina_regen=0.8, heal_ticks=40)


@dataclass
class BossState:
    k: int = 0
    stage: int = WINDUP
    j: int = 1          # tick index within the stage (1-based)
    D: int = 1          # sampled wind-up length of the current move
    p0: int = 0         # phase at the start of the current move
    hit_done: bool = False


@dataclass
class PlayerState:
    hp: float
    stamina: float
    heals: int
    action: int = IDLE
    t_in_action: int = 0   # ticks elapsed in the current action (0 = free)


def action_length(a: int, R: Rules) -> int:
    return {IDLE: 1, APPROACH: R.move_ticks, RETREAT: R.move_ticks, LIGHT: R.light_ticks,
            HEAVY: R.heavy_ticks, DODGE: R.dodge_ticks, BLOCK: R.block_ticks, HEAL: R.heal_ticks}[a]


def action_cost(a: int, R: Rules) -> float:
    return {LIGHT: R.light_cost, HEAVY: R.heavy_cost, DODGE: R.dodge_cost}.get(a, 0.0)


class MiniSouls:
    def __init__(self, boss: Boss, rules: Rules = Rules(), seed: int | None = None,
                 max_ticks: int = 3600, start_dist: float = 5.0):
        self.boss = boss
        self.R = rules
        self.rng = np.random.default_rng(seed)
        self.max_ticks = max_ticks
        self.start_dist = start_dist
        self._pmf = boss.windup_pmf()
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        R = self.R
        self.t = 0
        self.dist = self.start_dist
        self.boss_hp = self.boss.max_hp
        self.player = PlayerState(R.max_hp, R.max_stamina, R.heal_charges)
        self.bs = BossState(k=0, stage=WINDUP, j=1, D=self._sample_D(0, 0), p0=0)
        self.done = False
        self.outcome = None
        self.last_action = IDLE
        self.last_obs = self._observe(None)   # no cue yet: the first step emits the cue of s_0
        return self.last_obs

    def clone(self, seed: int | None = None) -> "MiniSouls":
        """Cheap copy for planning rollouts (shares the immutable boss and rules)."""
        c = object.__new__(MiniSouls)
        c.boss, c.R, c.max_ticks, c.start_dist, c._pmf = self.boss, self.R, self.max_ticks, self.start_dist, self._pmf
        c.rng = np.random.default_rng(seed)
        c.t, c.dist, c.boss_hp = self.t, self.dist, self.boss_hp
        c.player = replace(self.player)
        c.bs = replace(self.bs)
        c.done, c.outcome, c.last_obs = self.done, self.outcome, self.last_obs
        c.last_action = self.last_action
        return c

    def phase(self) -> int:
        return int(self.boss_hp <= 0.5 * self.boss.max_hp)

    def _sample_D(self, p: int, k: int) -> int:
        return int(self.rng.choice(len(self._pmf[p, k]), p=self._pmf[p, k])) + 1

    def player_free(self) -> bool:
        return self.player.t_in_action == 0

    def in_iframes(self) -> bool:
        p, R = self.player, self.R
        return p.action == DODGE and R.iframe_start <= p.t_in_action < R.iframe_start + R.iframe_len

    # ------------------------------------------------------------------- step
    def step(self, action: int):
        assert not self.done, "episode finished; call reset()"
        R, p = self.R, self.player
        reward = 0.0
        hp_before, boss_before = p.hp, self.boss_hp

        # 1. start a new action if free (unaffordable actions become idle)
        if p.t_in_action == 0:
            if action == HEAL and p.heals <= 0:
                action = IDLE
            if p.stamina < action_cost(action, R):
                action = IDLE
            p.action = action
            p.stamina -= action_cost(action, R)
        p.t_in_action += 1
        a, ti = p.action, p.t_in_action
        self.last_action = a
        if a == APPROACH:
            self.dist -= R.move_speed
        elif a == RETREAT:
            self.dist += R.move_speed
        elif a == DODGE:
            self.dist += R.dodge_speed
        elif a == LIGHT and ti == R.light_hit_tick and self.dist <= R.light_reach:
            self.boss_hp -= R.light_dmg
        elif a == HEAVY and ti == R.heavy_hit_tick and self.dist <= R.heavy_reach:
            self.boss_hp -= R.heavy_dmg
        elif a == HEAL and ti == R.heal_ticks:
            p.hp = min(R.max_hp, p.hp + R.heal_amount)
            p.heals -= 1
        if a not in (LIGHT, HEAVY, DODGE, BLOCK):
            p.stamina = min(R.max_stamina, p.stamina + R.stamina_regen)
        self.dist = float(np.clip(self.dist, MIN_DIST, MAX_DIST))

        # 2. boss cue + effects
        cue = self._emit_cue()
        self._boss_effects()

        # end the player's action after its last tick
        if p.t_in_action >= action_length(a, R):
            p.t_in_action = 0
            p.action = IDLE

        self.t += 1
        # 3. observation, then 4. latent transition (uses this tick's distance/phase)
        obs = self._observe(cue)
        self._boss_transition()

        reward = (boss_before - self.boss_hp) - 2.0 * (hp_before - p.hp)
        if self.boss_hp <= 0:
            self.done, self.outcome = True, "win"
            reward += 100.0
        elif p.hp <= 0:
            self.done, self.outcome = True, "death"
            reward -= 100.0
        elif self.t >= self.max_ticks:
            self.done, self.outcome = True, "timeout"
        self.last_obs = obs
        return obs, reward, self.done, {"outcome": self.outcome}

    def _emit_cue(self) -> int:
        bs, m = self.bs, self.boss.moves[self.bs.k]
        if bs.stage == WINDUP:
            true = windup_cue(m.primitive, bs.j, self.boss.tell_ticks)
        elif bs.stage == ACTIVE:
            true = active_cue(m.primitive)
        else:
            true = RECOVER_CUE
        if self.rng.random() < self.boss.cue_noise:
            return int(self.rng.integers(N_CUES))
        return true

    def _boss_effects(self):
        bs, prim = self.bs, self.boss.moves[self.bs.k].primitive
        if bs.stage == WINDUP and prim.name == "idle" and self.dist > 1.5:
            self.dist = max(MIN_DIST, self.dist - 0.12)
        if bs.stage == ACTIVE:
            if prim.lunge:
                self.dist = max(MIN_DIST, self.dist - prim.lunge / prim.active)
            # One impact tick per move: the first active tick (the last one for lunges, which
            # close distance first). Dodging works iff the impact tick falls inside the i-frames.
            impact = prim.active if prim.lunge else 1
            if bs.j == impact and not prim.feint and prim.damage > 0:
                bs.hit_done = True
                if self.dist <= prim.reach and not self.in_iframes():
                    p, R = self.player, self.R
                    dmg = prim.damage * self.boss.dmg_scale
                    if p.action == BLOCK and not prim.unblockable:
                        p.stamina -= 0.5 * dmg
                        dmg = dmg * R.block_dmg_frac if p.stamina >= 0 else dmg
                        p.stamina = max(p.stamina, 0.0)
                    p.hp -= dmg

    def _boss_transition(self):
        bs, m = self.bs, self.boss.moves[self.bs.k]
        if bs.stage == WINDUP:
            if bs.j < bs.D:
                bs.j += 1
                return
            nxt = ACTIVE if m.active > 0 else (RECOVERY if m.recovery > 0 else None)
        elif bs.stage == ACTIVE:
            if bs.j < m.active:
                bs.j += 1
                return
            nxt = RECOVERY if m.recovery > 0 else None
        else:
            if bs.j < m.recovery:
                bs.j += 1
                return
            nxt = None
        if nxt is not None:
            bs.stage, bs.j = nxt, 1
            return
        # move finished: choose next move with this tick's distance bin and phase
        ph = self.phase()
        k2 = int(self.rng.choice(self.boss.K, p=self.boss.trans(dist_bin(self.dist), ctx_of(self.last_action))[bs.k]))
        self.bs = BossState(k=k2, stage=WINDUP, j=1, D=self._sample_D(ph, k2), p0=ph)

    def _observe(self, cue: int) -> dict:
        p = self.player
        return {"t": self.t, "cue": cue, "dist": self.dist, "phase": self.phase(), "ctx": ctx_of(self.last_action),
                "boss_hp": self.boss_hp, "hp": p.hp, "stamina": p.stamina, "heals": p.heals,
                "free": p.t_in_action == 0}

    def truth(self) -> tuple:
        """Ground-truth boss latent (for tests and diagnostics only)."""
        return (self.bs.k, self.bs.stage, self.bs.j, self.bs.p0)
