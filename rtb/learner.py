"""The slow mind (M2): learn a boss program from fight logs, between attempts.

Belief = a weighted particle set of boss programs (resample-move SMC over programs).
Each particle is a full `Boss`: move list (structure), wind-up medians, phase-2 speed,
transition rows and reactive laws.

After every fight:
  1. reweight each particle by the exact log-evidence of the new fight (BossFilter);
  2. resample if the effective sample size drops below N/2;
  3. rejuvenate each particle with a few sweeps of
       a. FFBS: sample the latent boss path of every logged fight given the program
          (the final move of a fight that ended is right-censored);
       b. a structure proposal (delete an unused move, split a bimodal move, merge two
          copies, retype within a visual group, add a novel move for an unexplained cue
          group), accepted by the exact evidence ratio times the structure prior;
       c. Gibbs on parameters given paths: wind-up medians on a grid with the exact
          discretised log-normal likelihood (censoring included), phase-2 speed on a grid,
          Dirichlet transition rows, and reactive laws by enumeration.

The structure step is evidence-guided stochastic search (proposal asymmetry is not
corrected), so the particle set is an approximation to the posterior, not an exact one.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.special import logsumexp, ndtr

from .boss import Boss, D_MAX, Move, N_DIST_BINS, Theta, dist_bin, reach_pref, theta_star, SPREAD
from .filter import BossFilter
from .primitives import CUE_ID, GROUPS, PRIMITIVES

GRID = np.exp(np.linspace(np.log(2.0), np.log(55.0), 120))     # candidate wind-up medians
SPEED_GRID = np.linspace(0.6, 1.0, 41)
LAW_BOOSTS = (1.5, 2.25, 3.0)
P_NO_LAW = float(np.exp(-0.4))                                   # Poisson(0.8) split over 2 contexts


# ----------------------------------------------------------------------------- helpers
def make_boss(template: Boss, moves: list, pi: np.ndarray, speed: float, laws=()) -> Boss:
    return Boss(list(moves), pi, float(speed), template.tell_ticks, template.cue_noise, template.max_hp,
                dmg_scale=template.dmg_scale, name="belief", laws=tuple(laws))


def initial_belief(theta: Theta, tell_ticks: int, cue_noise: float, dmg_scale: float) -> Boss:
    """One move per allowed primitive at typical timing, reach-aware prior-mean transitions."""
    names = ["idle"] + list(theta.allowed)
    moves = [Move(PRIMITIVES[n], float(np.exp(theta.log_median[n])), theta.within_sigma[n]) for n in names]
    K = len(moves)
    pi = np.stack([np.stack([reach_pref(moves, b, theta.reach_bias) for b in range(N_DIST_BINS)])
                   for _ in range(K)])
    return Boss(moves, pi, float(np.mean(theta.phase2_speed)), tell_ticks, cue_noise, name="belief",
                dmg_scale=dmg_scale)


def fight_evidence(boss: Boss, trace) -> float:
    f = BossFilter(boss)
    for c, d, p, x in trace:
        f.update(c)
        f.predict(d, p, x)
    return f.log_evidence


def evidence(boss: Boss, traces) -> float:
    return float(sum(fight_evidence(boss, tr) for tr in traces))


# ----------------------------------------------------------------------------- FFBS
def _exit_candidates(f: BossFilter, W, k):
    """States in wind-up of move k that exit at this tick, with weights."""
    w = W[:, k, :] * f.exit_p[:, k, :]
    q, i = np.nonzero(w > 0)
    return [("W", int(qq), k, int(ii) + 1) for qq, ii in zip(q, i)], w[q, i]


def _back_step(f: BossFilter, alpha, s1, obs_t, rng):
    W, A, R = alpha
    a, r = f.a, f.r
    kind = s1[0]
    if kind == "W":
        _, p, k, d = s1
        if d > 1:
            return ("W", p, k, d - 1)
        col = f.boss.trans(dist_bin(obs_t[1]), obs_t[3])[:, k]
        cands, wts = [], []
        for kp in range(f.K):
            if col[kp] <= 0:
                continue
            if a[kp] == 0 and r[kp] == 0:
                c, w = _exit_candidates(f, W, kp)
                cands += c
                wts.append(w * col[kp])
            elif a[kp] > 0 and r[kp] == 0:
                cands.append(("A", kp, int(a[kp])))
                wts.append(np.array([A[kp, a[kp] - 1] * col[kp]]))
            elif r[kp] > 0:
                cands.append(("R", kp, int(r[kp])))
                wts.append(np.array([R[kp, r[kp] - 1] * col[kp]]))
        wts = np.concatenate(wts) if wts else np.zeros(0)
    elif kind == "A":
        _, k, j = s1
        if j > 1:
            return ("A", k, j - 1)
        cands, wts = _exit_candidates(f, W, k)
    else:
        _, k, j = s1
        if j > 1:
            return ("R", k, j - 1)
        if a[k] > 0:
            return ("A", k, int(a[k]))
        cands, wts = _exit_candidates(f, W, k)
    tot = wts.sum()
    if tot <= 0:
        raise FloatingPointError("FFBS: no predecessor with positive mass")
    return cands[int(rng.choice(len(cands), p=wts / tot))]


def _sample_flat(W, A, R, rng):
    flat = np.concatenate([W.ravel(), A.ravel(), R.ravel()])
    i = int(rng.choice(len(flat), p=flat / flat.sum()))
    if i < W.size:
        p, k, d = np.unravel_index(i, W.shape)
        return ("W", int(p), int(k), int(d) + 1)
    i -= W.size
    if i < A.size:
        k, j = np.unravel_index(i, A.shape)
        return ("A", int(k), int(j) + 1)
    k, j = np.unravel_index(i - A.size, R.shape)
    return ("R", int(k), int(j) + 1)


def ffbs(boss: Boss, trace, rng):
    """Sample one latent path given the program. Returns (path, log evidence)."""
    f = BossFilter(boss)
    alphas = []
    for c, d, p, x in trace:
        f.update(c)
        alphas.append((f.W.copy(), f.A.copy(), f.R.copy()))
        f.predict(d, p, x)
    T = len(trace)
    path = [None] * T
    path[-1] = _sample_flat(*alphas[-1], rng)
    for t in range(T - 2, -1, -1):
        path[t] = _back_step(f, alphas[t], path[t + 1], trace[t], rng)
    return path, f.log_evidence


def _move_of(s):
    return s[2] if s[0] == "W" else s[1]


def path_stats(path, trace):
    """Wind-ups (k, phase, D, censored) and transitions (k_prev, bin, ctx, k_new) on a path."""
    windups, trans = [], []
    T = len(path)
    for t, s in enumerate(path):
        if s[0] != "W":
            continue
        if s[3] == 1 and t > 0:
            trans.append((_move_of(path[t - 1]), dist_bin(trace[t - 1][1]), trace[t - 1][3], s[2]))
        nxt = path[t + 1] if t + 1 < T else None
        if nxt is None:
            windups.append((s[2], s[1], s[3], True))
        elif not (nxt[0] == "W" and nxt[1] == s[1] and nxt[2] == s[2] and nxt[3] == s[3] + 1):
            windups.append((s[2], s[1], s[3], False))
    return windups, trans


# ----------------------------------------------------------------------------- Gibbs pieces
def _ll_grid(D, cens, sigma, scale, grid=GRID):
    """(G,) log-likelihood of wind-up observations for each candidate median in `grid`."""
    if len(D) == 0:
        return np.zeros(len(grid))
    D = np.asarray(D, float)[None, :]
    cens = np.asarray(cens, bool)[None, :]
    mu = np.log(np.maximum(grid * scale, 1.0))[:, None]
    Z = ndtr((np.log(D_MAX + 0.5) - mu) / sigma) - ndtr((np.log(0.5) - mu) / sigma)
    hi = np.where(cens, np.log(D_MAX + 0.5), np.log(D + 0.5))
    p = ndtr((hi - mu) / sigma) - ndtr((np.log(D - 0.5) - mu) / sigma)
    return (np.log(np.maximum(p, 1e-300)) - np.log(Z)).sum(axis=1)


def gibbs_params(boss: Boss, stats, theta: Theta, rng) -> Boss:
    windups = [w for s in stats for w in s[0]]
    trans = [t for s in stats for t in s[1]]
    K = boss.K
    moves = list(boss.moves)
    speed = boss.phase2_speed
    # wind-up medians
    for k, m in enumerate(moves):
        name = m.primitive.name
        lp = -0.5 * ((np.log(GRID) - theta.log_median[name]) / theta.log_median_sd[name]) ** 2
        for ph, sc in ((0, 1.0), (1, speed)):
            obs = [(D, c) for (kk, p, D, c) in windups if kk == k and p == ph]
            if obs:
                lp = lp + _ll_grid([o[0] for o in obs], [o[1] for o in obs], m.sigma, sc)
        g = rng.choice(len(GRID), p=np.exp(lp - logsumexp(lp)))
        moves[k] = Move(m.primitive, float(GRID[g]), m.sigma)
    # phase-2 speed
    lp = np.where((SPEED_GRID >= 0.65) & (SPEED_GRID <= 0.95), 0.0, -np.inf)
    for (k, p, D, c) in windups:
        if p == 1:
            lp = lp + _ll_grid([D], [c], moves[k].sigma, 1.0, grid=moves[k].median * SPEED_GRID)
    speed = float(SPEED_GRID[rng.choice(len(SPEED_GRID), p=np.exp(lp - logsumexp(lp)))])
    # transition rows (contexts carrying a law are left to the law step)
    law_ctx = {l[0] for l in boss.laws}
    counts = np.zeros((K, N_DIST_BINS, K))
    for (kp, b, c, kn) in trans:
        if c not in law_ctx:
            counts[kp, b, kn] += 1
    pi = np.zeros_like(counts)
    for b in range(N_DIST_BINS):
        prior = theta.dirichlet_alpha * K * reach_pref(moves, b, theta.reach_bias) + 1e-3
        for k in range(K):
            pi[k, b] = rng.dirichlet(prior + counts[k, b])
    nb = make_boss(boss, moves, pi, speed, ())
    # reactive laws, one context at a time, by enumeration
    laws = []
    attack = [k for k, m in enumerate(moves) if m.primitive.damage > 0]
    for c in (1, 2):
        tc = [(kp, b, kn) for (kp, b, cc, kn) in trans if cc == c]
        opts, lps = [None], [np.log(P_NO_LAW) + sum(np.log(nb.trans(b, 0)[kp, kn]) for kp, b, kn in tc)]
        for tk in attack:
            for boost in LAW_BOOSTS:
                trial = make_boss(boss, moves, pi, speed, ((c, tk, boost),))
                opts.append((c, tk, boost))
                lps.append(np.log((1 - P_NO_LAW) / (len(attack) * len(LAW_BOOSTS)))
                           + sum(np.log(trial.trans(b, c)[kp, kn]) for kp, b, kn in tc))
        lps = np.array(lps)
        pick = opts[rng.choice(len(opts), p=np.exp(lps - logsumexp(lps)))]
        if pick is not None:
            laws.append(pick)
    return make_boss(boss, moves, pi, speed, laws)


# ----------------------------------------------------------------------------- structure
def _drop(boss: Boss, k: int) -> Boss:
    keep = [i for i in range(boss.K) if i != k]
    pi = boss.pi[np.ix_(keep, range(N_DIST_BINS), keep)]
    pi = pi / pi.sum(axis=2, keepdims=True)
    remap = {old: new for new, old in enumerate(keep)}
    laws = [(c, remap[tk], b) for (c, tk, b) in boss.laws if tk in remap]
    return make_boss(boss, [boss.moves[i] for i in keep], pi, boss.phase2_speed, laws)


def _add(boss: Boss, move: Move, copy_of: int | None = None) -> Boss:
    """Append a move. If copy_of is given the new move shares its row and half its column mass."""
    K = boss.K
    pi = np.zeros((K + 1, N_DIST_BINS, K + 1))
    pi[:K, :, :K] = boss.pi
    if copy_of is not None:
        pi[:K, :, K] = pi[:K, :, copy_of] / 2
        pi[:K, :, copy_of] /= 2
        pi[K] = pi[copy_of]
    else:
        pi[:K, :, K] = 1.0 / (K + 1)
        pi[:K] /= pi[:K].sum(axis=2, keepdims=True)
        pi[K, :, :] = 1.0 / (K + 1)
    return make_boss(boss, list(boss.moves) + [move], pi, boss.phase2_speed, boss.laws)


def _phase_free_logD(windups, k, speed):
    return [np.log(D / (speed if p == 1 else 1.0)) for (kk, p, D, c) in windups if kk == k and not c]


SIBLINGS = {"slam": "delayed_slam", "delayed_slam": "slam", "lunge": "feint", "feint": "lunge"}


def propose_structure(boss: Boss, stats, traces, theta: Theta, rng):
    """Return (proposal, log prior ratio, label) or None."""
    windups = [w for s in stats for w in s[0]]
    uses = np.zeros(boss.K, int)
    for (k, p, D, c) in windups:
        uses[k] += 1
    n_att = boss.K - 1
    kinds = []
    unused = [k for k in range(1, boss.K) if uses[k] == 0]
    if unused and n_att > 1:
        kinds.append("delete")
    splittable = [k for k in range(1, boss.K) if len(_phase_free_logD(windups, k, boss.phase2_speed)) >= 4]
    if splittable and n_att < 8:
        kinds.append("split")
    names = [m.primitive.name for m in boss.moves]
    pairs = [(i, j) for i in range(1, boss.K) for j in range(i + 1, boss.K) if names[i] == names[j]]
    if pairs:
        kinds.append("merge")
    retypable = [k for k in range(1, boss.K) if names[k] in SIBLINGS]
    if retypable:
        kinds.append("retype")
    # novel groups: cue groups seen in the logs that no move in the program can produce
    have = {m.primitive.group for m in boss.moves}
    seen = {}
    for tr in traces:
        for (c, *_r) in tr:
            for g in GROUPS:
                if c == CUE_ID[g]:
                    seen[g] = seen.get(g, 0) + 1
    novel = [g for g, n in seen.items() if g not in have and n >= 3]
    if novel and n_att < 8:
        kinds.append("add")
    if not kinds:
        return None
    kind = kinds[rng.integers(len(kinds))]
    lp_move = np.log(len(theta.allowed))          # one fewer move of 1-in-|allowed| type
    if kind == "delete":
        k = unused[rng.integers(len(unused))]
        return _drop(boss, k), lp_move, f"delete {names[k]}"
    if kind == "split":
        k = splittable[rng.integers(len(splittable))]
        ld = np.sort(_phase_free_logD(windups, k, boss.phase2_speed))
        lo, hi = float(np.exp(ld[: len(ld) // 2].mean())), float(np.exp(ld[len(ld) // 2:].mean()))
        m = boss.moves[k]
        nb = _add(boss, Move(m.primitive, hi, m.sigma), copy_of=k)
        nb.moves[k] = Move(m.primitive, lo, m.sigma)
        nb._pmf = None
        return nb, -lp_move, f"split {names[k]}"
    if kind == "merge":
        i, j = pairs[rng.integers(len(pairs))]
        li, lj = _phase_free_logD(windups, i, boss.phase2_speed), _phase_free_logD(windups, j, boss.phase2_speed)
        allv = li + lj
        med = float(np.exp(np.mean(allv))) if allv else float(np.sqrt(boss.moves[i].median * boss.moves[j].median))
        pi = boss.pi.copy()
        pi[:, :, i] += pi[:, :, j]
        nb = make_boss(boss, boss.moves, pi, boss.phase2_speed, boss.laws)
        nb.moves[i] = Move(boss.moves[i].primitive, med, boss.moves[i].sigma)
        return _drop(nb, j), lp_move, f"merge {names[i]}"
    if kind == "retype":
        k = retypable[rng.integers(len(retypable))]
        m = boss.moves[k]
        sib = PRIMITIVES[SIBLINGS[m.primitive.name]]
        nb = make_boss(boss, boss.moves, boss.pi.copy(), boss.phase2_speed, boss.laws)
        nb.moves[k] = Move(sib, m.median, theta.within_sigma.get(sib.name, m.sigma))
        return nb, 0.0, f"retype {m.primitive.name}->{sib.name}"
    g = novel[rng.integers(len(novel))]
    prim = next(p for p in PRIMITIVES.values() if p.group == g)
    mv = Move(prim, float(np.exp(theta.log_median[prim.name])), theta.within_sigma[prim.name])
    return _add(boss, mv), -lp_move + np.log(0.05), f"add novel {prim.name}"


# ----------------------------------------------------------------------------- SMC
class SlowMind:
    def __init__(self, template: Boss, theta: Theta | None = None, n_particles: int = 4, sweeps: int = 3,
                 seed: int = 0):
        self.theta = theta or theta_star(SPREAD)
        self.rng = np.random.default_rng(seed)
        b0 = initial_belief(self.theta, template.tell_ticks, template.cue_noise, template.dmg_scale)
        self.particles = [b0 for _ in range(n_particles)]
        self.logw = np.zeros(n_particles)
        self.traces = []
        self.sweeps = sweeps
        self.history = []          # per-update diagnostics

    @property
    def weights(self) -> np.ndarray:
        return np.exp(self.logw - logsumexp(self.logw))

    def ess(self) -> float:
        w = self.weights
        return float(1.0 / np.sum(w ** 2))

    def update(self, trace):
        self.traces.append(trace)
        # 1. reweight by the new fight's evidence
        self.logw = self.logw + np.array([fight_evidence(b, trace) for b in self.particles])
        # 2. resample
        resampled = False
        if self.ess() < len(self.particles) / 2:
            idx = self.rng.choice(len(self.particles), size=len(self.particles), p=self.weights)
            self.particles = [self.particles[i] for i in idx]
            self.logw = np.zeros(len(self.particles))
            resampled = True
        # 3. rejuvenate
        accepted = []
        for i in range(len(self.particles)):
            b, acc = self._rejuvenate(self.particles[i])
            self.particles[i] = b
            accepted += acc
        self.history.append({"n_fights": len(self.traces), "resampled": resampled, "ess": self.ess(),
                             "accepted": accepted, "K": [b.K for b in self.particles]})

    def _rejuvenate(self, boss: Boss):
        accepted = []
        for _ in range(self.sweeps):
            paths = [ffbs(boss, tr, self.rng) for tr in self.traces]
            ev = sum(p[1] for p in paths)
            stats = [path_stats(p, tr) for (p, _), tr in zip(paths, self.traces)]
            prop = propose_structure(boss, stats, self.traces, self.theta, self.rng)
            if prop is not None:
                nb, lprior, label = prop
                ev_new = evidence(nb, self.traces)
                if np.log(self.rng.random()) < ev_new - ev + lprior:
                    boss = nb
                    accepted.append(label)
                    paths = [ffbs(boss, tr, self.rng) for tr in self.traces]
                    stats = [path_stats(p, tr) for (p, _), tr in zip(paths, self.traces)]
            boss = gibbs_params(boss, stats, self.theta, self.rng)
        return boss, accepted

    def map_particle(self) -> Boss:
        return self.particles[int(np.argmax(self.logw))]


# ----------------------------------------------------------------------------- self-model (T4)
class RulesBelief:
    """Belief about the agent's own body rules.

    Stamina cost, regen and heal duration are read off proprioception (the stamina bar and
    the heal animation) after one fight, so they are set directly. The i-frame window
    (start, length) is not directly visible: it is inferred on a grid from every in-reach
    impact the agent experienced while rolling: a hit at roll tick tau rules tau out of
    the window, a clean dodge rules it in. The prior favours the rules of the training world.
    """

    def __init__(self, base, p_base: float = 0.5, noise: float = 0.02):
        self.base = base
        self.starts = np.arange(0, 6)
        self.lens = np.arange(1, 9)
        S, L = np.meshgrid(self.starts, self.lens, indexing="ij")
        self.S, self.L = S, L
        n = S.size
        prior = np.full(S.shape, (1 - p_base) / (n - 1))
        prior[(S == base.iframe_start) & (L == base.iframe_len)] = p_base
        self.logp = np.log(prior)
        self.noise = noise
        self.current = base

    def update(self, events, true_rules=None):
        for (tau, hit) in events:
            if tau <= 0:
                continue
            inside = (self.S <= tau) & (tau < self.S + self.L)
            p_hit = np.where(inside, self.noise, 1 - self.noise)
            self.logp += np.log(p_hit if hit else 1 - p_hit)
        i = np.unravel_index(np.argmax(self.logp), self.S.shape)
        kw = dict(iframe_start=int(self.S[i]), iframe_len=int(self.L[i]))
        if true_rules is not None:   # proprioceptive quantities: observed directly
            kw.update(dodge_cost=true_rules.dodge_cost, stamina_regen=true_rules.stamina_regen,
                      heal_ticks=true_rules.heal_ticks)
        self.current = replace(self.base, **kw)
        return self.current
