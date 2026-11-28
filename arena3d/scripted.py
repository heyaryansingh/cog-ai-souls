"""Reference players for Arena3D: random, aggressive, and an oracle dodger that peeks into the true future."""
from __future__ import annotations

import numpy as np
import os

from .env import GUARD, HEAL, LIGHT, N_ACTIONS, N_BUTTONS, NONE, ROLL


def random_policy(env, rng):
    return int(rng.integers(N_ACTIONS))


def aggro(env, rng):
    dist = np.linalg.norm(env.boss.root - env.p_xy)
    if dist > 0.8 + 0.25 * env.spec.scale:
        return 1 * N_BUTTONS + NONE
    return LIGHT


def _hit_soon(env, a, horizon, seed):
    sim = env.clone(seed)
    for k in range(horizon):
        _, _, done, info = sim.step(a if k == 0 else (0 if a % N_BUTTONS != NONE else a))
        if info["hit"] is not None or (done and sim.hp <= 0):
            return True
        if done:
            return False
    return False


def oracle_dodger(env, rng, horizon=14):
    """Cheats: simulates the true boss future (same RNG) to decide when and where to roll."""
    if not env.player_free():
        return 0
    seed = None   # same RNG stream as the real boss -> exact future
    if _hit_soon(env, NONE, horizon, seed):
        for d in rng.permutation([5, 3, 7, 4, 6, 2, 8]):
            a = int(d) * N_BUTTONS + ROLL
            if not _hit_soon(env, a, horizon, seed):
                return a
        return 5 * N_BUTTONS + ROLL
    if env.hp < 45 and env.flasks and np.linalg.norm(env.boss.root - env.p_xy) > 4:
        return HEAL
    dist = np.linalg.norm(env.boss.root - env.p_xy)
    if dist > 0.8 + 0.25 * env.spec.scale:
        return 1 * N_BUTTONS + NONE
    if not _hit_soon(env, LIGHT, 24, seed):
        return LIGHT
    return 5 * N_BUTTONS + NONE


class CueReflex:
    """Human-like reactive player on *perceived* cues only (sensor model: noise, dropout, 180 ms delay):
    roll sideways when a hand or foot closes on me fast and near; attack when the boss's whole body is
    still (recovery window) and in reach; otherwise close distance or wait. Thresholds in boss heights."""

    def __init__(self, v_th=0.035, r_th=0.75, still_th=0.25, seed=0):
        self.v_th, self.r_th, self.still_th = v_th, r_th, still_th
        self.rng = np.random.default_rng(seed)

    def reset(self, env):
        from .agent import Sensor
        self.env = env
        self.sensor = Sensor(env, 6, seed=int(self.rng.integers(1 << 30)))
        self.hist = []
        self.frame = (env.boss.root.copy(), 0.0, 1.75 * 1.5)
        self.side = 3

    def observe(self, obs, env, info=None):
        from .agent import centric
        P, _ = self.sensor()
        if P is not None:
            x, hip, fwd, h = centric(P)
            self.frame = (hip[:2].copy(), fwd, h)
            self.hist.append(x.reshape(-1, 3))
        elif self.hist:
            self.hist.append(self.hist[-1])
        self.hist = self.hist[-8:]

    def act(self, obs, env):
        if len(self.hist) < 4:
            return 1 * N_BUTTONS + NONE
        bxy, fwd, h = self.frame
        rel = env.p_xy - bxy
        c, s = np.cos(-fwd), np.sin(-fwd)
        me = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]]) / max(h, 0.5)
        P0, P3 = self.hist[-1], self.hist[-4]
        threat = 0.0
        for j in (5, 6, 3, 4, 11, 12):
            v = (P0[j] - P3[j]) / 3
            d = me - P0[j, :2]
            r = np.linalg.norm(d) + 1e-6
            closing = v[:2] @ d / r
            if r < self.r_th and (np.linalg.norm(v) > self.v_th or closing > 0.6 * self.v_th):
                threat = max(threat, np.linalg.norm(v))
        body = np.abs(P0 - P3).sum()
        dist = np.linalg.norm(me)
        if threat > 0 and env.stamina >= 18:
            self.side = 7 if self.side == 3 else 3
            return int(self.rng.choice([self.side, 4 if self.side == 3 else 6])) * N_BUTTONS + ROLL
        if env.hp < 45 and env.flasks and dist > 1.2:
            return HEAL
        if dist < 0.45 and body < self.still_th and env.stamina > 30:
            return LIGHT
        if dist > 0.4:
            return 1 * N_BUTTONS + NONE
        return 5 * N_BUTTONS + GUARD if env.stamina < 40 else 0 * N_BUTTONS + GUARD


def cue_reflex(env, rng, _state={}):
    """Stateless wrapper for scripts/arena3d_baselines.py (keeps one CueReflex per env)."""
    key = id(env)
    if _state.get("key") != key or env.t <= 1:
        _state["key"] = key
        _state["pol"] = CueReflex(seed=int(rng.integers(1 << 30)))
        _state["pol"].reset(env)
        _state["t"] = -1
    pol = _state["pol"]
    if env.t != _state["t"]:
        pol.observe(None, env)
        _state["t"] = env.t
    return pol.act(env, env) if env.player_free() else 0


def state_oracle(env, rng, horizon=14, n=5):
    """Perfect perception of the boss's *current* state, but no clairvoyance: imagines n futures with fresh
    randomness (hold lengths, next-move choices) and rolls if most of them hit. The information ceiling
    for any perception-based agent."""
    if not env.player_free():
        return 0
    seeds = [int(s) for s in rng.integers(1 << 30, size=n)]
    risk = lambda a, H=horizon: np.mean([_hit_soon(env, a, H, s) for s in seeds])
    if risk(NONE) >= 0.4:
        best, bv = 5 * N_BUTTONS + ROLL, 2.0
        for d in rng.permutation([5, 3, 7, 4, 6, 2, 8]):
            a = int(d) * N_BUTTONS + ROLL
            r = risk(a)
            if r < bv:
                best, bv = a, r
            if r == 0:
                break
        return best
    if env.hp < 45 and env.flasks and np.linalg.norm(env.boss.root - env.p_xy) > 4:
        return HEAL
    dist = np.linalg.norm(env.boss.root - env.p_xy)
    if dist > 0.8 + 0.25 * env.spec.scale:
        return 1 * N_BUTTONS + NONE
    if risk(LIGHT, 30) < 0.2:          # the attack commits the body for ~30 frames
        return LIGHT
    return 5 * N_BUTTONS + NONE


def grammar_prior(env, rng):
    """Sample the current move as an observer who knows the boss program and has seen the string so far would:
    mid-string moves are known (except at branch points); openers are drawn from the strings valid at this
    distance and phase, weighted as the program weights them."""
    b, prog = env.boss, env.spec.program
    st, pos = getattr(b, "cur_string", None), getattr(b, "string_pos", 1)
    if st is not None and pos > 1:
        if pos - 1 < len(st.moves):
            return prog.idx[st.moves[pos - 1]]
        opts = [o for o in (st.branch.get(len(st.moves)) or []) if o[1]]   # an empty continuation = string ends here
        if opts:
            ps = np.array([q for q, _ in opts])
            return prog.idx[opts[int(rng.choice(len(opts), p=ps / ps.sum()))][1][0]]
    dist = np.linalg.norm(env.p_xy - b.origin)
    ph = 2 if env.phase else 1
    ok = [x for x in prog.strings if x.min_d <= dist <= x.max_d and x.phase <= ph] or prog.strings
    w = np.array([x.weight * (1.5 if x.phase == 2 else 1.0) for x in ok])
    return prog.idx[ok[int(rng.choice(len(ok), p=w / w.sum()))].moves[0]]


def _blind(sim, env, rng, read_k=12, grammar=False):
    """Remove from an imagined copy what perception cannot know (B11): the remaining delay of a held windup
    (resampled from the boss's delay distribution) and, before the move is readable (read_k frames after
    onset), the move's identity (resampled from the boss's moves)."""
    from .boss import danger_frame, heading
    b, s = sim.boss, sim.spec
    if b.kind != "strike" or b.unit is None:
        return
    since = env.t - getattr(env, "boss_onset_t", env.t)
    j = b.move_id
    if since < read_k:
        prog = getattr(s, "program", None)
        cands = [i for i, m in enumerate(prog.moves) if m.phase <= (2 if env.phase else 1)] if prog else list(range(len(s.strikes)))
        if grammar == "learned":
            j = _LGP.sample(env, rng)
        elif grammar and prog:
            j = grammar_prior(env, rng)
        else:
            j = int(rng.choice(cands))
        u = sim.U[s.strikes[j]]
        b.unit, b.move_id, b.combo_last = u, j, j
        b.c = sim.C[u.clip]
        b.q0yaw = heading(b.c.quat[u.start]); b.p0 = b.c.pos[u.start].copy()
        k = danger_frame(u, s.scale); b.peak = k
        p = u.tip_rel[k, :2] * s.scale
        b.yaw_off = b.want0 - b.q0yaw - np.arctan2(p[1], p[0])
        dist = np.linalg.norm(env.p_xy - b.origin)
        b.lunge = float(np.clip(dist - np.linalg.norm(p) + 0.15, -1.5, 2.5 + s.scale))
        b.speed = s.speed[j] * (1.2 if env.phase else 1.0)
        b.f = min(b.f, u.length - 2)
        if prog:
            mv = prog.moves[j]; b.cur_move = mv; b.lunge += mv.lunge_boost
    if s.hold_med[j] > 0 and (b.hold > 0 or int(b.f) < max(0, b.peak - 5)):
        b.hold_at = max(0, b.peak - 5)
        b.hold = int(np.clip(round(rng.lognormal(np.log(s.hold_med[j]), s.hold_sig)), 1, 90))
        prog = getattr(s, "program", None)
        if prog and prog.moves[j].delay_med > 0 and rng.random() > prog.moves[j].p_delay:
            b.hold = 0


def _hit_soon_blind(env, a, horizon, seed, read_k, grammar=False):
    sim = env.clone(seed)
    _blind(sim, env, np.random.default_rng(seed), read_k, grammar)
    for k in range(horizon):
        _, _, done, info = sim.step(a if k == 0 else (0 if a % N_BUTTONS != NONE else a))
        if info["hit"] is not None or (done and sim.hp <= 0):
            return True
        if done:
            return False
    return False


def perceptual_oracle(env, rng, horizon=14, n=5, read_k=None, grammar=False):
    """State oracle limited to what perception can deliver: exact boss position and pose, but hidden delays and
    move identity unknown for the first READ_K frames (env var, default 12; B11). Same decisions as state_oracle."""
    import os
    rk = int(os.environ.get("READ_K", 12)) if read_k is None else read_k
    if not env.player_free():
        return 0
    seeds = [int(s) for s in rng.integers(1 << 30, size=n)]
    risk = lambda a, H=horizon: np.mean([_hit_soon_blind(env, a, H, s, rk, grammar) for s in seeds])
    if risk(NONE) >= 0.4:
        best, bv = 5 * N_BUTTONS + ROLL, 2.0
        for d in rng.permutation([5, 3, 7, 4, 6, 2, 8]):
            a = int(d) * N_BUTTONS + ROLL
            r = risk(a)
            if r < bv:
                best, bv = a, r
            if r == 0:
                break
        return best
    if env.hp < 45 and env.flasks and np.linalg.norm(env.boss.root - env.p_xy) > 4:
        return HEAL
    dist = np.linalg.norm(env.boss.root - env.p_xy)
    if dist > 0.8 + 0.25 * env.spec.scale:
        return 1 * N_BUTTONS + NONE
    if risk(LIGHT, 30) < 0.2:
        return LIGHT
    return 5 * N_BUTTONS + NONE


def perceptual_oracle_delay_only(env, rng):
    """Ablation: only the remaining delay is hidden; move identity known from the first frame."""
    return perceptual_oracle(env, rng, read_k=0)


def grammar_oracle(env, rng):
    """Perceptual oracle that also knows the boss program (string grammar): unreadable moves are anticipated
    from the string so far and the distance instead of guessed uniformly. Upper bound for a learned program model."""
    return perceptual_oracle(env, rng, grammar=True)


import os as _os
TOM = _os.environ.get("MIRROR_TOM") == "1"


def _uclass(act):
    """what the player is doing, as the boss can see it: 0 neutral, 1 healing, 2 attacking/recovering, 3 dodging, 4 guarding"""
    return {4: 1, 2: 2, 5: 2, 1: 3, 6: 3, 3: 4}.get(int(act), 0)


class LearnedGrammarPlanner:
    """Planner with realistic perception that LEARNS the boss program between deaths. Moves are identified
    READ_K frames after onset with READ_ACC accuracy (B11: ~.7 at 12-16 frames); each identification updates
    Dirichlet counts over the current move given (previous move in the string or 'opener', near/far, phase).
    Imagined futures draw unreadable moves from these counts (instead of uniform, or the true grammar).
    Counts persist across attempts. Physics/hitboxes of each move are assumed known (its own simulator)."""

    def __init__(self, n_moves, read_k=12, acc=0.7, alpha=0.5, seed=0, learn=True):
        self.n, self.read_k, self.acc, self.alpha, self.learn = n_moves, read_k, acc, alpha, learn
        self.counts = {}
        self.rng = np.random.default_rng(seed)

    def reset(self, env):
        self.prev, self.prev_end, self.seen_serial, self.ctx_cur = None, -999, -1, None
        if not self.learn:
            self.counts = {}

    def _ctx(self, env):
        b = env.boss
        opener = self.prev is None or env.t - self.prev_end > 45
        dist = np.linalg.norm(env.p_xy - b.origin)
        if TOM:     # H2: the boss's choice also depends on what I am doing (theory of mind: its policy is about me)
            return ("open" if opener else self.prev, dist > 4.0, bool(env.phase), _uclass(getattr(env, "act", 0)))
        return ("open" if opener else self.prev, dist > 4.0, bool(env.phase))

    def observe(self, env):
        b = env.boss
        serial = getattr(env, "boss_serial", -1)
        if b.kind == "strike" and serial != self.seen_serial:
            since = env.t - getattr(env, "boss_onset_t", env.t)
            if serial != getattr(self, "ctx_serial", None):
                self.ctx_cur, self.ctx_serial = self._ctx(env), serial
            if since >= self.read_k:
                self.seen_serial = serial
                lab = b.move_id if self.rng.random() < self.acc else int(self.rng.integers(self.n))
                if self.ctx_cur is not None and getattr(self, "update", True):
                    c = self.counts.setdefault(self.ctx_cur, np.zeros(self.n))
                    c[lab] += 1
                self.prev, self.prev_end = lab, env.t + 30
        elif b.kind == "strike":
            self.prev_end = env.t

    def sample(self, env, rng, next_move=False):
        ctx = self._ctx(env) if next_move else (self.ctx_cur or self._ctx(env))
        c = self.counts.get(ctx, np.zeros(self.n)) + self.alpha
        if TOM:     # hierarchical back-off: pool over my-action classes, weighted as a Dirichlet prior of strength 2
            pool = sum((v for k, v in self.counts.items() if k[:3] == ctx[:3]), np.zeros(self.n))
            c = c + 2.0 * pool / max(pool.sum(), 1.0)
        prog = env.spec.program
        ok = np.array([m.phase <= (2 if env.phase else 1) for m in prog.moves], float)
        p = c * ok
        return int(rng.choice(self.n, p=p / p.sum()))


_LGP = None


def learned_grammar_planner(env, rng):
    global _LGP
    _LGP.observe(env)
    return perceptual_oracle(env, rng, grammar="learned")


def _rollout_value(env, a, H, seed, read_k, grammar, hold=6):
    """Imagined outcome of committing to action a (held for movement): damage dealt - 2.5 x damage taken."""
    sim = env.clone(seed)
    _blind(sim, env, np.random.default_rng(seed), read_k, grammar)
    hp0, b0 = sim.hp, sim.boss_hp
    for k in range(H):
        mv = a % N_BUTTONS == NONE
        _, _, done, info = sim.step(a if (k == 0 or (mv and k < hold)) else 0)
        if done:
            break
    v = (b0 - sim.boss_hp) - 2.5 * (hp0 - sim.hp) - (100 if sim.hp <= 0 else 0)
    w = getattr(env, "_close_w", 0.0)
    if w:     # drive to engage: imagined end distance beyond own reach costs w per metre (B28)
        reach = 0.8 + 0.25 * env.spec.scale
        v -= w * max(0.0, float(np.linalg.norm(sim.boss.root - sim.p_xy)) - reach)
    return v


def mpc_planner(env, rng, H=24, n=3, grammar="learned"):
    """Search over own macro-actions with imagined futures (realistic perception limits + boss program prior)
    instead of hand-set thresholds. Replans every free frame, movement held for 6 frames."""
    import os
    if not env.player_free():
        return 0
    st = getattr(env, "_mpc", None)
    if st and st[1] > 0:
        env._mpc = (st[0], st[1] - 1)
        return st[0]
    g = grammar
    acts = [NONE, 1 * N_BUTTONS + NONE, 5 * N_BUTTONS + NONE, 3 * N_BUTTONS + NONE, 7 * N_BUTTONS + NONE,
            LIGHT, 3 * N_BUTTONS + ROLL, 7 * N_BUTTONS + ROLL, 5 * N_BUTTONS + ROLL]
    if env.flasks and env.hp < 60:
        acts.append(HEAL)
    # Engagement drive (B28): a short horizon makes standing off look safe and free, so a boss that is always
    # swinging is never approached. MPC_CLOSE sets a per-metre cost on ending out of reach; MPC_HUNGER makes it
    # grow with time since the agent last dealt damage (frustration -> pressure), reset by each hit landed.
    w0 = float(os.environ.get("MPC_CLOSE", 0))
    if w0:
        if getattr(env, "_last_bhp", None) != env.boss_hp:
            env._last_bhp, env._last_dmg_t = env.boss_hp, env.t
        hung = float(os.environ.get("MPC_HUNGER", 0))
        env._close_w = w0 * (1 + hung * (env.t - env._last_dmg_t) / 300.0)
    seeds = [int(s) for s in rng.integers(1 << 30, size=n)]
    vals = [np.mean([_rollout_value(env, a, H, s, int(os.environ.get("READ_K", 12)), g) for s in seeds]) for a in acts]
    a = acts[int(np.argmax(vals))]
    env._mpc = (a, 5 if a % N_BUTTONS == NONE and a != NONE else 0)
    return a


def mpc_grammar(env, rng):
    return mpc_planner(env, rng, grammar=True)


def mpc_uniform(env, rng):
    return mpc_planner(env, rng, grammar=False)


def mpc_learned(env, rng, ck="checkpoints/margit_grammar_counts.pkl", update=False):
    """MPC with Margit's strings learned by watching (B18 checkpoint), frozen (or updating within the fight)."""
    global _LGP
    if _LGP is None or getattr(_LGP, "_env", None) is not env:
        import pickle
        lg = pickle.load(open(ck, "rb"))
        counts = lg.counts; lg.reset(env); lg.counts = counts; lg.update = update; lg._env = env
        _LGP = lg
    _LGP.observe(env)
    return mpc_planner(env, rng, grammar="learned")


def mpc_learned_online(env, rng):
    return mpc_learned(env, rng, update=True)


def mpc_learned_clean(env, rng):
    """Strings learned by watching with perfect identification (separates label noise from context structure)."""
    return mpc_learned(env, rng, ck="checkpoints/margit_grammar_counts_clean.pkl")


class HierGrammar(LearnedGrammarPlanner):
    """Learned strings with hierarchical back-off (a Chinese-restaurant-style prior): the full context
    (prev, far, phase) borrows strength from the coarser context (prev) and from the overall move frequency,
    so a boss's habits generalise across distance/phase before each fine context has data."""
    beta = float(os.environ.get("HIER_BETA", 2.0))

    def sample(self, env, rng, next_move=False):
        ctx = self._ctx(env) if next_move else (self.ctx_cur or self._ctx(env))
        z = np.zeros(self.n)
        coarse = sum((c for k, c in self.counts.items() if k[0] == ctx[0]), z)
        glob = sum(self.counts.values(), z)
        pg = (glob + self.alpha) / (glob.sum() + self.alpha * self.n)
        pc = (coarse + self.beta * pg) / (coarse.sum() + self.beta)
        c = self.counts.get(ctx, z) + self.beta * pc
        ok = np.array([m.phase <= (2 if env.phase else 1) for m in env.spec.program.moves], float)
        p = c * ok + 1e-9
        return int(rng.choice(self.n, p=p / p.sum()))


def _mpc_scratch(env, rng, cls):
    """General learner: knows nothing about this boss's strings; learns them from scratch during the fight."""
    global _LGP
    if _LGP is None or getattr(_LGP, "_env", None) is not env:
        lg = cls(len(env.spec.program.moves), acc=float(os.environ.get("READ_ACC", 0.7)), seed=env.seed if hasattr(env, "seed") else 0)
        lg.reset(env); lg.update = True; lg._env = env
        _LGP = lg
    _LGP.observe(env)
    return mpc_planner(env, rng, grammar="learned")


def mpc_scratch(env, rng):
    return _mpc_scratch(env, rng, LearnedGrammarPlanner)


def mpc_scratch_hier(env, rng):
    return _mpc_scratch(env, rng, HierGrammar)
