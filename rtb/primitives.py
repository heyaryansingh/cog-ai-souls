"""Move-primitive library and cue vocabulary for MiniSouls.

Times are in ticks (20 ticks = 1 second). Each boss move instantiates one primitive:
windup (variable duration, the "telegraph") -> active (hit window) -> recovery (punish window).
"""
from dataclasses import dataclass

TICKS_PER_SECOND = 20


@dataclass(frozen=True)
class Primitive:
    name: str
    group: str            # visual family of the wind-up: moves in a group look alike
    windup_median: float  # typical wind-up length in ticks (Theta* is centred near this)
    active: int           # ticks the hit window lasts (0 = no hit window)
    recovery: int         # ticks of recovery after the active window
    reach: float          # max distance at which the hit connects
    damage: float
    unblockable: bool = False
    lunge: float = 0.0    # distance the boss closes during the active window
    feint: bool = False   # active window shows a feint cue and never hits


PRIMITIVES = {
    "idle":         Primitive("idle", "idle", 12, 0, 0, 0.0, 0.0),
    "slam":         Primitive("slam", "overhead", 14, 3, 18, 2.5, 30.0),
    "delayed_slam": Primitive("delayed_slam", "overhead", 26, 3, 20, 2.5, 35.0),
    "sweep":        Primitive("sweep", "sweep", 10, 4, 12, 3.5, 20.0),
    "lunge":        Primitive("lunge", "thrust", 12, 4, 16, 2.0, 25.0, lunge=2.5),
    "feint":        Primitive("feint", "thrust", 10, 2, 6, 0.0, 0.0, feint=True),
    "jab":          Primitive("jab", "jab", 6, 2, 6, 2.0, 12.0),
    "aoe":          Primitive("aoe", "burst", 20, 4, 24, 5.0, 30.0),
    "grab":         Primitive("grab", "grab", 16, 3, 25, 1.5, 40.0, unblockable=True),  # held out for T3
}

ATTACK_PRIMITIVES = [n for n in PRIMITIVES if n != "idle"]
HELD_OUT_PRIMITIVE = "grab"

GROUPS = ["overhead", "sweep", "thrust", "jab", "burst", "grab"]
# Cue vocabulary. "tell" is the generic first ticks of any wind-up (before the group is readable).
CUES = (["idle", "tell"] + GROUPS + [f"a_{g}" for g in GROUPS] + ["a_feint", "recover"])
CUE_ID = {c: i for i, c in enumerate(CUES)}
N_CUES = len(CUES)


def windup_cue(prim: Primitive, elapsed: int, tell_ticks: int) -> int:
    if prim.name == "idle":
        return CUE_ID["idle"]
    if elapsed <= tell_ticks:
        return CUE_ID["tell"]
    return CUE_ID[prim.group]


def active_cue(prim: Primitive) -> int:
    if prim.feint:
        return CUE_ID["a_feint"]
    if prim.group == "idle":       # idle has no active window; never emitted
        return CUE_ID["idle"]
    return CUE_ID[f"a_{prim.group}"]


RECOVER_CUE = CUE_ID["recover"]
