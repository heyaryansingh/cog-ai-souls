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


def build_program_boss(rng, units, allowed, clips, moves=MARGIT_MOVES, strings=MARGIT_STRINGS, scale=1.9,
                       weapon="greatsword", name="margit_like", verify=True):
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
                    aggression=0.85, hp=1100.0, idles=idles, p_recover=1.0)
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
            return "idle", int(rng.integers(len(self.s.idles)))      # every string ends in a recovery
        if self.player_healing and rng.random() < 0.7:
            return "strike", prog.idx["dagger_throw"]
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
