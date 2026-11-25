"""Boss generative program.

A boss is an explicit-duration hidden semi-Markov model (HSMM) over moves:

    move k:  windup D ~ pmf_k  ->  active a_k ticks  ->  recovery r_k ticks
    next move ~ pi[k, distance_bin, :]      (chosen at the tick the move ends)

Wind-up pmfs are discretised log-normals around a per-move median; in phase 2
(boss HP <= 50%) every wind-up median is multiplied by `phase2_speed` (< 1).

The same object is used to *generate* behaviour (MiniSouls) and to *score* it
(the exact filter), so the model and the simulator can never disagree.

Hyperpriors:
  Theta   - per-primitive timing hyperparameters + structure knobs. The narrow
            "true" Theta* generates the 40 training bosses and test splits.
  META    - a broad distribution over Theta used to pretrain every learner
            (intuition net, RL baselines), so learning Theta* from 40 bosses is
            a real learning-to-learn problem.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .primitives import PRIMITIVES, ATTACK_PRIMITIVES, HELD_OUT_PRIMITIVE, Primitive

D_MAX = 60            # longest wind-up the model represents (3 s)
N_CTX = 3             # player-action context seen by the boss: 0 neutral, 1 healing, 2 attacking
NEAR_FAR_SPLIT = 2.8  # distance bins: near (< split) / far
N_DIST_BINS = 2


def dist_bin(dist: float) -> int:
    return 0 if dist < NEAR_FAR_SPLIT else 1


def ctx_of(action: int) -> int:
    """Context a reactive boss reads from the player's action on the tick a move ends."""
    if action == 7:          # HEAL
        return 1
    if action in (3, 4):     # LIGHT, HEAVY
        return 2
    return 0


def discretised_lognormal(median: float, sigma: float, d_max: int = D_MAX) -> np.ndarray:
    """pmf over d = 1..d_max (index d-1)."""
    d = np.arange(1, d_max + 1, dtype=float)
    edges_lo, edges_hi = np.log(d - 0.5), np.log(d + 0.5)
    from math import erf, sqrt, log
    mu = log(max(median, 1.0))
    cdf = lambda x: 0.5 * (1.0 + np.vectorize(erf)((x - mu) / (sigma * sqrt(2.0))))
    p = cdf(edges_hi) - cdf(edges_lo)
    p = np.maximum(p, 0.0)
    if p.sum() <= 0:
        p[int(min(max(median, 1), d_max)) - 1] = 1.0
    return p / p.sum()


@dataclass
class Theta:
    """Hyperparameters that bosses are drawn from."""
    log_median: dict          # primitive -> mean of log wind-up median across bosses
    log_median_sd: dict       # primitive -> sd of log wind-up median across bosses
    within_sigma: dict        # primitive -> log-sd of wind-up from one use to the next
    n_attacks: tuple = (3, 6)         # inclusive range of attack moves per boss
    dirichlet_alpha: float = 0.6      # concentration of transition rows
    reach_bias: float = 2.0           # how strongly near/far bins favour in-reach moves
    phase2_speed: tuple = (0.75, 0.9)
    law_rate: float = 0.8              # expected number of reactive laws per boss (Poisson)
    law_boost: tuple = (1.5, 3.0)      # range of log-boosts (x4.5 to x20 more likely)
    allowed: tuple = tuple(p for p in ATTACK_PRIMITIVES if p != HELD_OUT_PRIMITIVE)


def theta_star(spread: float = 0.15) -> Theta:
    """The narrow 'true' distribution the course-world bosses come from.
    `spread` = between-boss sd of log wind-up medians (how idiosyncratic each boss's timing is)."""
    lm = {n: float(np.log(p.windup_median)) for n, p in PRIMITIVES.items()}
    sd = {n: spread for n in PRIMITIVES}
    ws = {n: 0.10 for n in PRIMITIVES}
    ws["delayed_slam"] = 0.18
    ws["idle"] = 0.35
    return Theta(lm, sd, ws)


def sample_meta_theta(rng: np.random.Generator) -> Theta:
    """Broad meta-prior over Theta, used for pretraining. Theta* is one draw-like point in it."""
    base = theta_star()
    lm = {n: v + rng.normal(0, 0.35) for n, v in base.log_median.items()}
    sd = {n: float(rng.uniform(0.1, 0.4)) for n in base.log_median}
    ws = {n: float(rng.uniform(0.06, 0.3)) for n in base.log_median}
    ws["idle"] = float(rng.uniform(0.2, 0.5))
    return Theta(lm, sd, ws,
                 n_attacks=(3, 7),
                 dirichlet_alpha=float(rng.uniform(0.3, 1.5)),
                 reach_bias=float(rng.uniform(0.5, 3.0)),
                 phase2_speed=(0.65, 0.95))


@dataclass
class Move:
    primitive: Primitive
    median: float
    sigma: float

    @property
    def active(self) -> int:
        return self.primitive.active

    @property
    def recovery(self) -> int:
        return self.primitive.recovery


@dataclass
class Boss:
    """One boss = one sample from the generative program. Move 0 is always 'idle'."""
    moves: list
    pi: np.ndarray                   # (K, N_DIST_BINS, K) base next-move probabilities
    phase2_speed: float
    tell_ticks: int = 3              # ticks of generic 'tell' cue before the group is readable
    cue_noise: float = 0.03          # probability a cue token is replaced by a random one
    max_hp: float = 200.0
    dmg_scale: float = 1.0           # multiplies every attack's damage (difficulty knob)
    laws: tuple = ()                 # reactive laws: (ctx, target move k, log-boost)
    name: str = "boss"
    _pmf: np.ndarray = field(default=None, repr=False)
    _trans: dict = field(default=None, repr=False)

    def trans(self, b: int, c: int = 0) -> np.ndarray:
        """(K, K) next-move matrix for distance bin b and player context c (laws applied)."""
        if self._trans is None:
            self._trans = {}
        key = (b, c)
        if key not in self._trans:
            m = self.pi[:, b, :].copy()
            for (lc, tk, boost) in self.laws:
                if lc == c:
                    m[:, tk] *= np.exp(boost)
            self._trans[key] = m / m.sum(axis=1, keepdims=True)
        return self._trans[key]

    @property
    def K(self) -> int:
        return len(self.moves)

    def windup_pmf(self) -> np.ndarray:
        """(2 phases, K, D_MAX) wind-up pmfs, cached."""
        if self._pmf is None:
            pmf = np.zeros((2, self.K, D_MAX))
            for k, m in enumerate(self.moves):
                pmf[0, k] = discretised_lognormal(m.median, m.sigma)
                pmf[1, k] = discretised_lognormal(m.median * self.phase2_speed, m.sigma)
            self._pmf = pmf
        return self._pmf

    def describe(self) -> str:
        lines = [f"{self.name}: {self.K} moves, phase-2 speed x{self.phase2_speed:.2f}"]
        for k, m in enumerate(self.moves):
            lines.append(f"  [{k}] {m.primitive.name:13s} windup~{m.median:5.1f}t (sd {m.sigma:.2f}) "
                         f"active {m.active} rec {m.recovery}")
        for (c, tk, boost) in self.laws:
            lines.append(f"  law: if player {['neutral','heals','attacks'][c]} -> move [{tk}] x{np.exp(boost):.1f}")
        return "\n".join(lines)


def reach_pref(moves, b: int, reach_bias: float) -> np.ndarray:
    """Reach-aware base preference over next moves for distance bin b (normalised)."""
    pref = np.ones(len(moves))
    for j, m in enumerate(moves):
        eff = 2.0 if m.primitive.feint else (m.primitive.reach + m.primitive.lunge)
        in_reach = eff >= (1.5 if b == 0 else 3.5)
        if m.primitive.name == "idle":
            pref[j] = 1.0 if b == 0 else 2.0
        else:
            pref[j] = np.exp(reach_bias * (1.0 if in_reach else -1.0))
    return pref / pref.sum()


def sample_boss(theta: Theta, rng: np.random.Generator, primitives: list | None = None,
                name: str = "boss", tell_ticks: int = 3, cue_noise: float = 0.03,
                dmg_scale: float = 1.0) -> Boss:
    """Draw a boss from Theta. `primitives` forces a specific attack set (used for splits)."""
    if primitives is None:
        n = int(rng.integers(theta.n_attacks[0], theta.n_attacks[1] + 1))
        # Moves may repeat a primitive with different timing (two slams, one quick one slow).
        primitives = list(rng.choice(list(theta.allowed), size=n, replace=True))
    names = ["idle"] + list(primitives)
    moves = []
    for nm in names:
        prim = PRIMITIVES[nm]
        med = float(np.exp(rng.normal(theta.log_median[nm], theta.log_median_sd[nm])))
        moves.append(Move(prim, med, theta.within_sigma[nm]))
    K = len(moves)
    pi = np.zeros((K, N_DIST_BINS, K))
    for k in range(K):
        for b in range(N_DIST_BINS):
            # Reach-aware base preference: near bin likes short-reach moves, far bin likes
            # long-reach moves and idling (approach). Dirichlet noise gives each boss its habits.
            pref = reach_pref(moves, b, theta.reach_bias)
            pi[k, b] = rng.dirichlet(theta.dirichlet_alpha * K * pref + 1e-3)
    speed = float(rng.uniform(*theta.phase2_speed))
    # Reactive laws: the boss reads the player. Heals get punished by a fast in-reach attack,
    # attacks get answered by a quick counter. Each boss has its own (possibly no) laws.
    laws = []
    attack_ks = [k for k, m in enumerate(moves) if m.primitive.damage > 0]
    for _ in range(int(rng.poisson(theta.law_rate))):
        if not attack_ks:
            break
        c = int(rng.integers(1, N_CTX))
        tk = int(rng.choice(attack_ks))
        if any(l[0] == c for l in laws):
            continue
        laws.append((c, tk, float(rng.uniform(*theta.law_boost))))
    return Boss(moves, pi, speed, tell_ticks=tell_ticks, cue_noise=cue_noise, name=name,
                dmg_scale=dmg_scale, laws=tuple(laws))


# Default difficulty, chosen on 25 Nov 2026 so that boss-specific knowledge matters: with these
# settings a planner given the true program wins ~80%, while planners that know only typical
# timings win 20-30% (scripts/learning_gap.py).
SPREAD, TELL_TICKS, CUE_NOISE, DMG_SCALE = 0.35, 6, 0.03, 1.4


def make_splits(seed: int = 0, n_train: int = 40, n_test: int = 10, spread: float = SPREAD,
                tell_ticks: int = TELL_TICKS, cue_noise: float = CUE_NOISE, dmg_scale: float = DMG_SCALE) -> dict:
    """Training bosses and held-out test sets.

    T1  same primitive sets as training, new timings
    T2  primitive combinations never seen together in training
    T3  includes the held-out primitive (grab)
    T4  training-like bosses; the *player rules* change (handled in the env config)
    """
    rng = np.random.default_rng(seed)
    th = theta_star(spread)
    kw = dict(tell_ticks=tell_ticks, cue_noise=cue_noise, dmg_scale=dmg_scale)
    train = [sample_boss(th, rng, name=f"train{i:02d}", **kw) for i in range(n_train)]
    seen_sets = {frozenset(m.primitive.name for m in b.moves[1:]) for b in train}

    t1 = []
    for i in range(n_test):
        src = train[i % n_train]
        t1.append(sample_boss(th, rng, [m.primitive.name for m in src.moves[1:]], name=f"T1-{i}", **kw))

    t2 = []
    while len(t2) < n_test:
        b = sample_boss(th, rng, name=f"T2-{len(t2)}", **kw)
        if frozenset(m.primitive.name for m in b.moves[1:]) not in seen_sets:
            t2.append(b)

    th3 = theta_star(spread)
    th3.allowed = tuple(ATTACK_PRIMITIVES)
    t3 = []
    while len(t3) < n_test:
        b = sample_boss(th3, rng, name=f"T3-{len(t3)}", **kw)
        if any(m.primitive.name == HELD_OUT_PRIMITIVE for m in b.moves):
            t3.append(b)

    t4 = [sample_boss(th, rng, name=f"T4-{i}", **kw) for i in range(n_test)]
    return {"train": train, "T1": t1, "T2": t2, "T3": t3, "T4": t4}


def prior_mean_boss(boss: Boss, theta: Theta | None = None) -> Boss:
    """Same move list as `boss`, but timings at Theta's typical values and uniform transitions:
    what an agent that recognised the moves but learned nothing about THIS boss would believe."""
    theta = theta or theta_star()
    moves = [Move(m.primitive, float(np.exp(theta.log_median[m.primitive.name])), m.sigma) for m in boss.moves]
    K = len(moves)
    pi = np.full((K, N_DIST_BINS, K), 1.0 / K)
    return Boss(moves, pi, float(np.mean(theta.phase2_speed)), boss.tell_ticks, boss.cue_noise,
                boss.max_hp, name=boss.name + "-prior", dmg_scale=boss.dmg_scale)


def generic_boss(theta: Theta | None = None, tell_ticks: int = 3, cue_noise: float = 0.03,
                 dmg_scale: float = 1.0) -> Boss:
    """What an agent knowing only Theta believes before meeting a boss: one move per allowed
    primitive at typical timings, uniform transitions."""
    theta = theta or theta_star()
    names = ["idle"] + list(theta.allowed)
    moves = [Move(PRIMITIVES[n], float(np.exp(theta.log_median[n])), theta.within_sigma[n]) for n in names]
    K = len(moves)
    pi = np.full((K, N_DIST_BINS, K), 1.0 / K)
    return Boss(moves, pi, float(np.mean(theta.phase2_speed)), tell_ticks, cue_noise, name="generic",
                dmg_scale=dmg_scale)
