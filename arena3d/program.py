"""Boss programs: authored structure on top of mocap animation, the way real souls bosses are built.

A real boss is a set of named moves (each with a windup, possible delay, one or more hits, a recovery)
arranged in strings with branch points, distance-dependent openers, string enders that leave a punish
window, special rules (punish healing with a throw) and a phase-2 move set. Margit (Elden Ring) is the
first template, from his published move list (Fextralife): phase 1 staff strings with a branch into a
thrust or a delayed overhead, dagger throws that punish healing, a leaping stab; phase 2 adds hammer moves
that always end the string, with a shockwave. Frame timings are placeholders until measured from footage.

Animation for each move slot is a verified mocap unit of the right *shape* (sweep / overhead / thrust),
classified from the weapon-tip velocity at the danger frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .boss import BossRunner, BossSpec, danger_frame


def unit_shape(u, scale=1.7):
    k = danger_frame(u, scale)
    if k < 0:
        return None
    v = (u.tip_rel[min(k + 1, u.length - 1)] - u.tip_rel[k - 1]) / 2
    a = np.abs(v)
    if v[2] < 0 and a[2] >= max(a[0], a[1]):
        return "overhead"
    return "sweep" if a[1] >= a[0] else "thrust"


@dataclass
class MoveDef:
    name: str
    shape: str                  # sweep | overhead | thrust | throw
    delay_med: float = 0.0      # windup hold median (frames); 0 = none
    p_delay: float = 1.0        # chance the delay is used (mix-ups)
    speed: float = 1.0
    ender: bool = False
    ring: float = 0.0           # shockwave radius at impact (m), 0 = none
    lunge_boost: float = 0.0    # extra gap-closing (leaps)
    projectile: bool = False
    phase: int = 1
    recover: int = 0            # extra frozen recovery after the move (frames), the punish window of enders


@dataclass
class StringDef:
    moves: list                 # move names
    branch: dict = field(default_factory=dict)   # index -> [(prob, [continuation names])]
    min_d: float = 0.0
    max_d: float = 99.0
    weight: float = 1.0
    phase: int = 1


MARGIT_MOVES = [
    MoveDef("staff_sweep", "sweep", speed=1.1),
    MoveDef("staff_sweep_b", "sweep", speed=1.0),
    MoveDef("staff_thrust", "thrust", speed=1.1),
    MoveDef("delayed_overhead", "overhead", delay_med=24, p_delay=0.8, speed=0.9, ender=True),
    MoveDef("dagger_throw", "throw", projectile=True),
    MoveDef("leaping_stab", "overhead", delay_med=10, lunge_boost=2.5, ender=True),
    MoveDef("hammer_slam", "overhead", delay_med=18, speed=0.85, ender=True, ring=4.0, phase=2),
    MoveDef("sword_spin", "sweep", speed=1.25, phase=2),
    MoveDef("hammer_leap", "overhead", delay_med=8, lunge_boost=4.0, ender=True, ring=5.0, phase=2),
]
MARGIT_STRINGS = [
    StringDef(["staff_sweep", "staff_sweep_b"], branch={2: [(0.5, ["staff_thrust"]), (0.5, ["delayed_overhead"])]},
              max_d=4.0, weight=3.0),
    StringDef(["staff_thrust", "staff_sweep"], max_d=4.0, weight=2.0),
    StringDef(["delayed_overhead"], max_d=4.5, weight=1.5),
    StringDef(["dagger_throw"], min_d=4.0, weight=1.5),
    StringDef(["dagger_throw", "dagger_throw"], min_d=5.0, weight=1.0),
    StringDef(["leaping_stab"], min_d=3.5, max_d=9.0, weight=1.5),
    StringDef(["staff_sweep", "sword_spin", "sword_spin"], max_d=4.0, weight=2.0, phase=2),
    StringDef(["staff_sweep", "hammer_slam"], max_d=4.0, weight=2.0, phase=2),
    StringDef(["hammer_leap"], min_d=4.0, weight=2.0, phase=2),
]


# Footage calibration (log B22, Margit moveset video, frame-annotated at 30 fps, +-4 frames): leap slam
# takeoff->impact ~1.3 s (sim 0.63 s) and ~1.5 s recovery after big hammer moves (sim ~0.5-0.9 s) were off by
# more than 2x; staff swings (0.55-0.65 s real vs 0.45 s sim) and the summoned-hammer sweep (~1.2 s both) were not.
import dataclasses as _dc
MARGIT_MOVES_CAL = [
    _dc.replace(m, delay_med=30) if m.name == "hammer_leap" else m for m in MARGIT_MOVES]
MARGIT_MOVES_CAL = [_dc.replace(m, recover=25) if m.name in ("hammer_slam", "hammer_leap") else
                    (_dc.replace(m, recover=12) if m.ender else m) for m in MARGIT_MOVES_CAL]


def build_program_boss(rng, units, allowed, clips, moves=MARGIT_MOVES, strings=MARGIT_STRINGS, scale=1.9,
                       weapon="greatsword", name="margit_like", verify=True, hp=450.0):
    """Pick a verified mocap unit of the right shape for each move slot; returns (spec, program)."""
    from .aim import verify_strikes
    by_shape = {"sweep": [], "overhead": [], "thrust": []}
    for i in allowed:
        u = units[i]
        if u.kind != "strike" or u.length >= 75 or danger_frame(u, scale) < 12:
            continue
        sh = unit_shape(u, scale)
        if sh:
            by_shape[sh].append(i)
    pool_w = [i for i in allowed if units[i].kind == "move" and units[i].travel / (units[i].length * 0.03) > 0.6]
    pool_i = [i for i in allowed if units[i].kind == "move" and units[i].peak_speed < 3.0
              and units[i].travel / (units[i].length * 0.03) < 0.3 and units[i].length >= 15]
    walks = [int(i) for i in rng.choice(pool_w, 6, replace=False)]
    idles = [int(i) for i in rng.choice(pool_i, 6, replace=False)]
    proto = BossSpec(name=name, scale=scale, weapon=weapon, strikes=[], walks=walks, combo=None, p_end=None,
                     speed=None, hold_med=None, hold_sig=0.45, track=0.05, aggression=1.0, idles=idles)
    strikes, used = [], set()
    for mv in moves:
        sh = "sweep" if mv.shape == "throw" else mv.shape
        cand = [int(c) for c in rng.permutation(by_shape[sh]) if c not in used]
        got = verify_strikes(proto, units, clips, cand, 1, rng) if verify else cand[:1]
        ui = got[0] if got else cand[0]
        used.add(ui)
        strikes.append(ui)
    n = len(moves)
    spec = BossSpec(name=name, scale=scale, weapon=weapon, strikes=strikes, walks=walks,
                    combo=np.vstack([np.ones(n) / n, np.ones((n, n)) / n]), p_end=np.ones(n),
                    speed=np.array([m.speed for m in moves]) / np.sqrt(scale / 1.6),
                    hold_med=np.array([m.delay_med for m in moves], float), hold_sig=0.35, track=0.05,
                    aggression=0.85, hp=hp, idles=idles, p_recover=1.0)
    spec.program = Program(moves, strings)
    return spec


class Program:
    def __init__(self, moves, strings):
        self.moves, self.strings = moves, strings
        self.idx = {m.name: i for i, m in enumerate(moves)}


class ProgramRunner(BossRunner):
    """BossRunner driven by a Program: strings with branches, enders -> recovery, distance-based openers,
    heal punish with a throw, phase-2 strings."""

    def reset(self, pos, yaw):
        super().reset(pos, yaw)
        self.queue, self.string_pos, self.cur_string = [], 0, None
        self.player_healing = False

    def _choose(self, player_xy, phase):
        prog: Program = self.s.program
        rng = self.rng
        dist = np.linalg.norm(player_xy - self.root)
        last = prog.moves[self.combo_last] if self.combo_last >= 0 else None
        if self.queue:
            name = self.queue.pop(0)
            self.string_pos += 1
            self._maybe_branch()
            return "strike", prog.idx[name]
        if last is not None:
            self.combo_last = -1
            if last.recover > 0:
                self.stagger = int(last.recover)              # frozen recovery pose: the punish window
            return "idle", int(rng.integers(len(self.s.idles)))      # every string ends in a recovery
        punish = [i for i, m in enumerate(prog.moves) if m.projectile and m.phase <= (2 if phase else 1)]
        if self.player_healing and punish and rng.random() < 0.7:     # input reading: punish heals
            return "strike", punish[0]
        ph = 2 if phase else 1
        ok = [s for s in prog.strings if s.min_d <= dist <= s.max_d and s.phase <= ph]
        if not ok or rng.random() > self.s.aggression:
            return "walk", int(rng.integers(len(self.s.walks)))
        w = np.array([s.weight * (1.5 if s.phase == 2 else 1.0) for s in ok])
        st = ok[int(rng.choice(len(ok), p=w / w.sum()))]
        self.cur_string, self.string_pos = st, 1
        self.queue = list(st.moves[1:])
        self._maybe_branch()
        return "strike", prog.idx[st.moves[0]]

    def _maybe_branch(self):
        st = self.cur_string
        if st is None or self.queue:
            return
        opts = st.branch.get(self.string_pos)
        if opts:
            ps = np.array([p for p, _ in opts])
            k = int(self.rng.choice(len(opts), p=ps / ps.sum()))
            self.queue = list(opts[k][1])

    def _start(self, player_xy, phase):
        super()._start(player_xy, phase)
        if self.kind == "strike":
            mv = self.s.program.moves[self.move_id]
            if mv.delay_med > 0 and self.rng.random() > mv.p_delay:
                self.hold = 0                                     # mix-up: same move, no delay
            self.lunge += mv.lunge_boost
            self.cur_move = mv
        else:
            self.cur_move = None


def random_program(rng, n_moves=None):
    """A procedural boss program for the generalisation suite: random named moves (shape, delay mix-ups,
    speed, enders with recovery, rings, projectiles, leaps), random strings with branch points and distance
    ranges, and a phase-2 moveset switch (2-3 new moves, new strings)."""
    n1 = int(n_moves or rng.integers(4, 8)); n2 = int(rng.integers(2, 4))
    moves = []
    for i in range(n1 + n2):
        ph = 1 if i < n1 else 2
        shape = str(rng.choice(["sweep", "sweep", "overhead", "thrust"]))
        r = rng.random()
        mv = MoveDef(f"m{i}", shape, speed=float(rng.uniform(0.8, 1.3)), phase=ph)
        if rng.random() < 0.35:
            mv = _dc.replace(mv, delay_med=float(rng.integers(8, 36)), p_delay=float(rng.uniform(0.5, 1.0)))
        if r < 0.12:
            mv = _dc.replace(mv, shape="throw", projectile=True)
        elif r < 0.25:
            mv = _dc.replace(mv, lunge_boost=float(rng.uniform(1.5, 4.0)))
        if rng.random() < 0.3 and not mv.projectile:
            mv = _dc.replace(mv, ender=True, recover=int(rng.integers(8, 30)))
            if rng.random() < 0.4:
                mv = _dc.replace(mv, ring=float(rng.uniform(3.0, 5.0)))
        moves.append(mv)
    strings = []
    for ph, pool in ((1, list(range(n1))), (2, list(range(n1 + n2)))):
        for _ in range(int(rng.integers(3, 6))):
            L = int(rng.integers(1, 4))
            seq = [moves[int(rng.choice(pool))].name for _ in range(L)]
            if ph == 2 and not any(moves[int(m[1:])].phase == 2 for m in seq):
                seq[-1] = moves[int(rng.integers(n1, n1 + n2))].name
            br = {}
            if L >= 2 and rng.random() < 0.4:
                a, b = (moves[int(rng.choice(pool))].name for _ in range(2))
                br = {L: [(0.5, [a]), (0.5, [b])]}
            far = rng.random() < 0.3 or any(moves[int(m[1:])].projectile or moves[int(m[1:])].lunge_boost > 0 for m in seq[:1])
            strings.append(StringDef(seq, branch=br, min_d=4.0 if far else 0.0, max_d=99.0 if far else 4.5,
                                     weight=float(rng.uniform(0.5, 3.0)), phase=ph))
    # every distance band needs an option in each phase
    for ph in (1, 2):
        pool = [m.name for m in moves if m.phase <= ph]
        strings.append(StringDef([str(rng.choice(pool))], max_d=4.5, weight=0.5, phase=ph))
        strings.append(StringDef([str(rng.choice(pool))], min_d=4.0, weight=0.5, phase=ph))
    return moves, strings


# B55: a second hand-authored boss, Godrick-like (heavy axe, long multi-hit strings, wind shockwaves, a dragon arm in
# phase 2). Timings are from general knowledge of the fight (slower, longer wind-ups and strings than Margit, many
# area rings), NOT frame-annotated footage: treat as "Godrick-flavoured", uncalibrated.
GODRICK_MOVES = [
    MoveDef("axe_sweep", "sweep", speed=0.9),
    MoveDef("axe_sweep_b", "sweep", speed=0.9),
    MoveDef("axe_overhead", "overhead", delay_med=14, p_delay=0.6, speed=0.85, ender=True, ring=2.5),
    MoveDef("stomp", "overhead", delay_med=20, speed=0.8, ender=True, ring=4.5, recover=15),
    MoveDef("axe_charge", "thrust", speed=1.0, lunge_boost=2.0),
    MoveDef("jump_slam", "overhead", delay_med=20, lunge_boost=4.0, ender=True, ring=4.0, recover=20),
    MoveDef("whirlwind", "sweep", delay_med=30, speed=0.7, ender=True, ring=5.5, recover=25),
    MoveDef("dragon_breath", "throw", projectile=True, phase=2),
    MoveDef("dragon_swipe", "sweep", speed=1.0, phase=2),
    MoveDef("dragon_slam", "overhead", delay_med=16, speed=0.85, ender=True, ring=4.5, phase=2),
]
GODRICK_STRINGS = [
    StringDef(["axe_sweep", "axe_sweep_b", "axe_sweep"], branch={3: [(0.5, ["axe_overhead"]), (0.3, ["stomp"]), (0.2, [])]},
              max_d=4.0, weight=3.0),
    StringDef(["axe_charge", "axe_sweep"], min_d=2.5, max_d=7.0, weight=1.5),
    StringDef(["stomp"], max_d=4.0, weight=1.5),
    StringDef(["axe_overhead"], max_d=4.5, weight=1.5),
    StringDef(["jump_slam"], min_d=4.0, max_d=10.0, weight=2.0),
    StringDef(["whirlwind"], max_d=5.0, weight=1.0),
    StringDef(["dragon_breath"], min_d=3.0, weight=2.0, phase=2),
    StringDef(["axe_sweep", "dragon_swipe", "dragon_slam"], max_d=4.0, weight=2.5, phase=2),
    StringDef(["dragon_slam"], max_d=5.0, weight=1.5, phase=2),
]
