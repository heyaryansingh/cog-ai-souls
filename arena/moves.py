"""Boss moves as motion clips, and the procedural boss generator (archetypes -> movesets -> parameters).

A Move is a list of Segments. Each segment eases the skeleton from its current pose to a target pose
over `ticks` ticks (30 Hz). A "hold" segment keeps the pose for a random, log-normal duration: the
delay that baits early rolls, and the timing a learner must infer. A "strike" segment carries a
hitbox, applied once at its impact tick. Moves chain into combos through continuation probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

HZ = 30


@dataclass(frozen=True)
class Hitbox:
    shape: str                 # "arc" | "capsule" | "disc" | "projectile"
    damage: float
    r: float = 0.0             # arc / disc radius, projectile radius
    a0: float = 0.0            # arc angle range (rad, boss-local; + = left)
    a1: float = 0.0
    x0: float = 0.0            # capsule from x0 to x1 along the boss's forward axis
    x1: float = 0.0
    w: float = 0.0             # capsule half-width
    cx: float = 0.0            # disc centre (boss-local)
    cy: float = 0.0
    r_in: float = 0.0          # disc inner safe radius (rings)
    low: bool = False          # can be jumped over
    unblockable: bool = False
    poise: float = 30.0        # poise / guard damage
    speed: float = 0.0         # projectile speed (m / tick)
    homing: float = 0.0        # projectile turn rate toward the player (rad / tick)
    life: int = 0              # projectile lifetime (ticks)


@dataclass(frozen=True)
class Segment:
    kind: str                  # "pose" | "hold" | "strike" | "recover"
    pose: str
    ticks: int = 1             # fixed duration (pose / strike / recover)
    hold_median: float = 0.0   # hold: log-normal duration
    hold_sigma: float = 0.0
    track: float = 0.0         # turn rate toward the player (rad / tick) during the segment
    travel: float = 0.0        # forward travel over the segment (m); with `leap` it is a max
    leap: bool = False         # travel aims at the player's position at segment start
    hitbox: Hitbox | None = None
    impact: int = 0            # tick within a strike when the hitbox is applied (0-based)


@dataclass(frozen=True)
class Move:
    name: str
    family: str                # visual family: moves of a family share their first segment
    segments: tuple
    ideal: tuple = (0.0, 3.0)  # preferred player-distance band for choosing the move
    phase: int = 1             # 1 = available from the start, 2 = unlocked at half HP
    rear: bool = False         # usable against a player behind the boss (spins, stomps)


def _hit(seg_pose, ticks, hb, impact=None, travel=0.0, track=0.0):
    return Segment("strike", seg_pose, ticks=ticks, hitbox=hb, impact=ticks - 1 if impact is None else impact,
                   travel=travel, track=track)


# --------------------------------------------------------------------------------- move templates
def knight_moves(k):
    """k: dict of per-boss parameters (speed multiplier `s`, hold medians, damage scale `d`, tracking `tr`)."""
    s, d, tr = k["s"], k["d"], k["tr"]
    T = lambda x: max(1, int(round(x * s)))
    arc_r = Hitbox("arc", 22 * d, r=3.0, a0=-1.3, a1=1.3, poise=35)
    arc_l = Hitbox("arc", 20 * d, r=3.0, a0=-1.3, a1=1.3, poise=30)
    return [
        Move("slash_r", "windback_r", (Segment("pose", "windback_r", T(9), track=tr),
                                         Segment("hold", "windback_r", hold_median=k["h_slash"], hold_sigma=0.25, track=tr),
                                         _hit("slash_end_l", T(4), arc_r, travel=0.6),
                                         Segment("recover", "rest", T(16))), ideal=(0.0, 3.0)),
        Move("slash_l", "windback_l", (Segment("pose", "windback_l", T(7), track=tr),
                                         Segment("hold", "windback_l", hold_median=k["h_slash"] * 0.6, hold_sigma=0.25, track=tr),
                                         _hit("slash_end_r", T(4), arc_l, travel=0.6),
                                         Segment("recover", "rest", T(14))), ideal=(0.0, 3.0)),
        Move("thrust", "draw_back", (Segment("pose", "draw_back", T(10), track=tr * 1.5),
                                       Segment("hold", "draw_back", hold_median=k["h_thrust"], hold_sigma=0.3, track=tr * 1.5),
                                       _hit("thrust_ext", T(3), Hitbox("capsule", 26 * d, x0=0.0, x1=3.6, w=0.55, poise=40), travel=1.6),
                                       Segment("recover", "rest", T(20))), ideal=(2.0, 5.0)),
        Move("overhead", "raise_both", (Segment("pose", "raise_both", T(12), track=tr),
                                          Segment("hold", "raise_both", hold_median=k["h_over"], hold_sigma=0.45, track=tr * 0.7),
                                          _hit("slam_end", T(3), Hitbox("disc", 34 * d, r=1.3, cx=1.9, poise=60)),
                                          Segment("recover", "rest", T(26))), ideal=(0.5, 2.8)),
        Move("jump_slam", "crouch_deep", (Segment("pose", "crouch_deep", T(10), track=tr * 2),
                                            Segment("hold", "airborne", hold_median=14, hold_sigma=0.15, travel=7.0, leap=True),
                                            _hit("slam_end", T(3), Hitbox("disc", 30 * d, r=2.2, cx=1.0, poise=60)),
                                            Segment("recover", "rest", T(30))), ideal=(4.0, 9.0)),
        Move("low_sweep", "low_windback", (Segment("pose", "low_windback", T(11), track=tr),
                                             Segment("hold", "low_windback", hold_median=k["h_slash"], hold_sigma=0.3, track=tr),
                                             _hit("low_sweep_end", T(5), Hitbox("arc", 18 * d, r=3.4, a0=-1.6, a1=1.6, low=True, poise=30), travel=0.3),
                                             Segment("recover", "rest", T(18))), ideal=(0.0, 3.2), phase=2),
        Move("spin", "windback_r", (Segment("pose", "windback_r", T(9), track=tr),        # shares slash_r's wind-up
                                      Segment("hold", "spin_wind", hold_median=k["h_slash"] * 1.4, hold_sigma=0.3),
                                      _hit("spin_end", T(6), Hitbox("arc", 24 * d, r=3.2, a0=-np.pi, a1=np.pi, poise=45)),
                                      Segment("recover", "rest", T(22))), ideal=(0.0, 3.0), rear=True),
    ]


def beast_moves(k):
    s, d, tr = k["s"], k["d"], k["tr"]
    T = lambda x: max(1, int(round(x * s)))
    bite = lambda i: Move(f"bite{i}", "bite_open", (Segment("pose", "bite_open", T(5 + 2 * i), track=tr * 2.5),
                                                      Segment("hold", "bite_open", hold_median=k["h_bite"], hold_sigma=0.3, track=tr * 2.5),
                                                      _hit("bite_close", T(3), Hitbox("arc", 14 * d, r=2.3, a0=-0.7, a1=0.7, poise=20), travel=0.9),
                                                      Segment("recover", "rest", T(8 + 4 * i))), ideal=(0.0, 2.6))
    return [
        bite(1), bite(2), bite(3),
        Move("pounce", "pounce_crouch", (Segment("pose", "pounce_crouch", T(10), track=tr * 2),
                                           Segment("hold", "pounce_crouch", hold_median=k["h_pounce"], hold_sigma=0.4, track=tr * 2),
                                           Segment("pose", "airborne", T(10), travel=6.0, leap=True),
                                           _hit("bite_close", T(3), Hitbox("arc", 28 * d, r=2.4, a0=-0.9, a1=0.9, poise=55)),
                                           Segment("recover", "rest", T(24))), ideal=(3.5, 9.0)),
        Move("tail_spin", "spin_wind", (Segment("pose", "spin_wind", T(8), track=0.0),
                                          Segment("hold", "spin_wind", hold_median=k["h_bite"] * 1.5, hold_sigma=0.3),
                                          _hit("spin_end", T(6), Hitbox("arc", 20 * d, r=3.6, a0=-np.pi, a1=np.pi, low=True, poise=35)),
                                          Segment("recover", "rest", T(20))), ideal=(0.0, 3.5), rear=True),
        Move("charge", "pounce_crouch", (Segment("pose", "pounce_crouch", T(10), track=tr * 2),
                                           Segment("hold", "pounce_crouch", hold_median=k["h_pounce"] * 0.6, hold_sigma=0.3, track=tr * 3),
                                           _hit("bite_open", T(14), Hitbox("capsule", 25 * d, x0=0.0, x1=1.6, w=0.9, poise=50), impact=7, travel=7.0),
                                           Segment("recover", "rest", T(26))), ideal=(3.0, 9.0), phase=2),
        Move("roar", "ring_charge", (Segment("pose", "ring_charge", T(14)),
                                       Segment("hold", "ring_charge", hold_median=12, hold_sigma=0.2),
                                       _hit("ring_burst", T(4), Hitbox("disc", 10 * d, r=3.5, poise=80)),
                                       Segment("recover", "rest", T(20))), ideal=(0.0, 3.0), rear=True, phase=2),
    ]


def caster_moves(k):
    s, d, tr = k["s"], k["d"], k["tr"]
    T = lambda x: max(1, int(round(x * s)))
    bolt = Hitbox("projectile", 18 * d, r=0.45, speed=0.45, life=60, poise=25)
    orb = Hitbox("projectile", 26 * d, r=0.7, speed=0.16, homing=0.035, life=180, poise=40)
    return [
        Move("bolt", "cast_raise", (Segment("pose", "cast_raise", T(9), track=tr * 2),
                                      Segment("hold", "cast_raise", hold_median=k["h_cast"], hold_sigma=0.3, track=tr * 2),
                                      _hit("cast_throw", T(3), bolt, impact=1),
                                      Segment("recover", "rest", T(12))), ideal=(3.0, 12.0)),
        Move("triple_bolt", "cast_raise", (Segment("pose", "cast_raise", T(9), track=tr * 2),
                                             Segment("hold", "cast_raise", hold_median=k["h_cast"] * 1.6, hold_sigma=0.3, track=tr * 2),
                                             _hit("cast_throw", T(3), bolt, impact=1),
                                             Segment("pose", "cast_raise", T(6), track=tr * 3), _hit("cast_throw", T(3), bolt, impact=1),
                                             Segment("pose", "cast_raise", T(6), track=tr * 3), _hit("cast_throw", T(3), bolt, impact=1),
                                             Segment("recover", "rest", T(18))), ideal=(4.0, 12.0), phase=2),
        Move("ring", "ring_charge", (Segment("pose", "ring_charge", T(14)),
                                       Segment("hold", "ring_charge", hold_median=k["h_ring"], hold_sigma=0.45),
                                       _hit("ring_burst", T(4), Hitbox("disc", 30 * d, r=4.0, r_in=1.2, poise=60)),
                                       Segment("recover", "rest", T(24))), ideal=(0.0, 4.5), rear=True),
        Move("orb", "cast_raise", (Segment("pose", "cast_raise", T(12), track=tr),
                                     Segment("hold", "cast_raise", hold_median=k["h_cast"] * 2.2, hold_sigma=0.35, track=tr),
                                     _hit("cast_throw", T(3), orb, impact=1),
                                     Segment("recover", "rest", T(16))), ideal=(2.0, 12.0)),
        Move("staff_swipe", "windback_r", (Segment("pose", "windback_r", T(8), track=tr),
                                             Segment("hold", "windback_r", hold_median=6, hold_sigma=0.25, track=tr),
                                             _hit("slash_end_l", T(4), Hitbox("arc", 16 * d, r=2.6, a0=-1.2, a1=1.2, poise=25)),
                                             Segment("recover", "rest", T(14))), ideal=(0.0, 2.6)),
        Move("blink", "crouch_deep", (Segment("pose", "crouch_deep", T(8)),
                                        Segment("hold", "crouch_deep", hold_median=6, hold_sigma=0.2, travel=-6.0),
                                        Segment("recover", "rest", T(8))), ideal=(0.0, 2.2)),
    ]


def giant_moves(k):
    s, d, tr = k["s"], k["d"], k["tr"]
    T = lambda x: max(1, int(round(x * s)))
    return [
        Move("stomp", "stomp_lift", (Segment("pose", "stomp_lift", T(14), track=tr * 0.5),
                                       Segment("hold", "stomp_lift", hold_median=k["h_stomp"], hold_sigma=0.35),
                                       _hit("stomp_down", T(3), Hitbox("disc", 30 * d, r=2.6, cx=0.5, low=True, poise=70)),
                                       Segment("recover", "rest", T(28))), ideal=(0.0, 3.0), rear=True),
        Move("giant_sweep", "low_windback", (Segment("pose", "low_windback", T(16), track=tr * 0.6),
                                               Segment("hold", "low_windback", hold_median=k["h_stomp"] * 0.7, hold_sigma=0.3, track=tr * 0.6),
                                               _hit("low_sweep_end", T(7), Hitbox("arc", 26 * d, r=4.6, a0=-1.8, a1=1.8, low=True, poise=60)),
                                               Segment("recover", "rest", T(26))), ideal=(1.0, 4.5)),
        Move("grab", "grab_reach", (Segment("pose", "grab_reach", T(12), track=tr * 1.2),
                                      Segment("hold", "grab_reach", hold_median=k["h_grab"], hold_sigma=0.3, track=tr * 1.2),
                                      _hit("grab_close", T(4), Hitbox("capsule", 45 * d, x0=0.3, x1=2.4, w=0.7, unblockable=True, poise=100), travel=0.8),
                                      Segment("recover", "rest", T(30))), ideal=(0.5, 2.6)),
        Move("body_slam", "raise_both", (Segment("pose", "raise_both", T(16), track=tr),
                                           Segment("hold", "raise_both", hold_median=k["h_stomp"] * 1.4, hold_sigma=0.45, track=tr * 0.6),
                                           _hit("slam_end", T(4), Hitbox("disc", 38 * d, r=2.4, cx=1.6, poise=90), travel=1.0),
                                           Segment("recover", "rest", T(36))), ideal=(0.5, 3.5)),
        Move("rock_throw", "cast_raise", (Segment("pose", "cast_raise", T(16), track=tr * 1.5),
                                            Segment("hold", "cast_raise", hold_median=k["h_grab"], hold_sigma=0.3, track=tr * 1.5),
                                            _hit("cast_throw", T(4), Hitbox("projectile", 28 * d, r=0.9, speed=0.35, life=60, poise=60), impact=2),
                                            Segment("recover", "rest", T(22))), ideal=(4.0, 12.0), phase=2),
        Move("double_stomp", "stomp_lift", (Segment("pose", "stomp_lift", T(14), track=tr * 0.5),       # shares stomp's wind-up
                                              Segment("hold", "stomp_lift", hold_median=k["h_stomp"], hold_sigma=0.35),
                                              _hit("stomp_down", T(3), Hitbox("disc", 22 * d, r=2.6, cx=0.5, low=True, poise=60)),
                                              Segment("pose", "stomp_lift", T(8)),
                                              Segment("hold", "stomp_lift", hold_median=k["h_stomp"] * 0.8, hold_sigma=0.5),
                                              _hit("stomp_down", T(3), Hitbox("disc", 30 * d, r=3.4, cx=0.5, low=True, poise=70)),
                                              Segment("recover", "rest", T(30))), ideal=(0.0, 3.5), rear=True, phase=2),
    ]


ARCHETYPES = {"knight": knight_moves, "beast": beast_moves, "caster": caster_moves, "giant": giant_moves}
ARCH_STATS = {   # hp, walk speed (m/tick), turn rate, posture max, preferred range
    "knight": dict(hp=700, walk=0.07, turn=0.06, posture=120, range=2.4),
    "beast": dict(hp=600, walk=0.11, turn=0.09, posture=100, range=2.0),
    "caster": dict(hp=500, walk=0.05, turn=0.05, posture=80, range=6.0),
    "giant": dict(hp=900, walk=0.05, turn=0.035, posture=160, range=2.5),
}


@dataclass
class BossSpec:
    name: str
    archetype: str
    moves: list
    cont: np.ndarray            # (K, K+1) combo continuation probabilities; last column = end the string
    stats: dict
    params: dict
    laws: tuple = ()            # input reading: (player action name, move index, log-boost)
    phase2_speed: float = 0.85


def sample_boss(rng: np.random.Generator, archetype: str | None = None, name: str = "boss",
                n_moves: tuple = (4, 6), exclude: tuple = ()) -> BossSpec:
    arch = archetype or rng.choice(list(ARCHETYPES))
    k = dict(s=float(np.exp(rng.normal(0, 0.12))), d=float(np.exp(rng.normal(0, 0.1))),
             tr=float(np.exp(rng.normal(np.log(0.03), 0.4))),
             h_slash=float(np.exp(rng.normal(np.log(6), 0.35))), h_thrust=float(np.exp(rng.normal(np.log(8), 0.35))),
             h_over=float(np.exp(rng.normal(np.log(16), 0.4))), h_bite=float(np.exp(rng.normal(np.log(4), 0.35))),
             h_pounce=float(np.exp(rng.normal(np.log(10), 0.35))), h_cast=float(np.exp(rng.normal(np.log(8), 0.35))),
             h_ring=float(np.exp(rng.normal(np.log(18), 0.4))), h_stomp=float(np.exp(rng.normal(np.log(10), 0.35))),
             h_grab=float(np.exp(rng.normal(np.log(9), 0.35))))
    pool = [m for m in ARCHETYPES[arch](k) if m.name not in exclude]
    p1 = [m for m in pool if m.phase == 1]
    p2 = [m for m in pool if m.phase == 2]
    n1 = int(min(len(p1), rng.integers(n_moves[0], n_moves[1] + 1)))
    chosen = [p1[i] for i in sorted(rng.choice(len(p1), size=n1, replace=False))]
    if p2:
        chosen += [p2[i] for i in sorted(rng.choice(len(p2), size=int(min(len(p2), rng.integers(1, 3))), replace=False))]
    K = len(chosen)
    # combo strings: each move continues into another with total prob ~ Beta, else ends the string
    cont = np.zeros((K, K + 1))
    for i in range(K):
        p_cont = float(rng.beta(2, 3))
        w = rng.dirichlet(np.full(K, 0.5))
        cont[i, :K] = p_cont * w
        cont[i, K] = 1 - p_cont
    laws = []
    if rng.random() < 0.6:   # punish heals with a fast or gap-closing move
        laws.append(("heal", int(rng.integers(K)), float(rng.uniform(1.5, 3.0))))
    if rng.random() < 0.4:   # react to rolls
        laws.append(("roll", int(rng.integers(K)), float(rng.uniform(1.0, 2.5))))
    st = dict(ARCH_STATS[arch])
    st["hp"] = float(st["hp"] * np.exp(rng.normal(0, 0.1)))
    return BossSpec(name, arch, chosen, cont, st, k, tuple(laws), float(rng.uniform(0.78, 0.92)))
