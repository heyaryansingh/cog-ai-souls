"""Boss skeleton: 17 keypoints in 3D, boss-local frame (x forward, y left, z up, metres).

A pose is a (17, 3) array. Named poses are built from a rest pose plus per-joint overrides and
whole-body transforms (crouch, lean, twist). Motion between poses is eased interpolation, so a move's
tell is a *trajectory* of keypoints, and moves that share a wind-up share their first frames.
"""
from __future__ import annotations

import numpy as np

JOINTS = ["pelvis", "spine", "neck", "head",
          "r_shoulder", "r_elbow", "r_hand", "l_shoulder", "l_elbow", "l_hand",
          "r_hip", "r_knee", "r_foot", "l_hip", "l_knee", "l_foot", "weapon_tip"]
J = {n: i for i, n in enumerate(JOINTS)}
N_JOINTS = len(JOINTS)

REST = np.array([
    [0.0, 0.0, 1.0],    # pelvis
    [0.0, 0.0, 1.35],   # spine
    [0.0, 0.0, 1.65],   # neck
    [0.05, 0.0, 1.85],  # head
    [0.0, -0.22, 1.6], [0.05, -0.32, 1.3], [0.25, -0.3, 1.1],     # right arm (y negative = right)
    [0.0, 0.22, 1.6], [0.05, 0.32, 1.3], [0.2, 0.3, 1.05],        # left arm
    [0.0, -0.12, 0.95], [0.05, -0.15, 0.5], [0.0, -0.15, 0.0],    # right leg
    [0.0, 0.12, 0.95], [0.05, 0.15, 0.5], [0.0, 0.15, 0.0],       # left leg
    [0.9, -0.3, 1.3],   # weapon tip (held forward-up in the right hand)
])


def _rot_z(p, ang, about=np.zeros(3)):
    c, s = np.cos(ang), np.sin(ang)
    q = p - about
    return np.stack([c * q[:, 0] - s * q[:, 1], s * q[:, 0] + c * q[:, 1], q[:, 2]], 1) + about


def make_pose(arm_r=None, arm_l=None, weapon=None, crouch=0.0, lean=0.0, twist=0.0, legs=None):
    """arm_r / arm_l: (elbow, hand) positions; weapon: tip position; crouch: lowers the body (m);
    lean: forward lean of the upper body (m at the head); twist: upper-body yaw (rad, + = left)."""
    p = REST.copy()
    if arm_r is not None:
        p[J["r_elbow"]], p[J["r_hand"]] = arm_r
    if arm_l is not None:
        p[J["l_elbow"]], p[J["l_hand"]] = arm_l
    if weapon is not None:
        p[J["weapon_tip"]] = weapon
    if legs is not None:
        for name, pos in legs.items():
            p[J[name]] = pos
    upper = [J[n] for n in ["spine", "neck", "head", "r_shoulder", "r_elbow", "r_hand", "l_shoulder",
                            "l_elbow", "l_hand", "weapon_tip"]]
    if twist:
        p[upper] = _rot_z(p[upper], twist, about=np.array([0.0, 0.0, 0.0]))
    if lean:
        h = (p[upper, 2] - 1.0) / 0.85
        p[upper, 0] += lean * np.clip(h, 0, None)
    if crouch:
        p[:, 2] = np.where(np.arange(N_JOINTS) == J["r_foot"], p[:, 2],
                           np.where(np.arange(N_JOINTS) == J["l_foot"], p[:, 2], p[:, 2] - crouch))
    return p


P = np.array
POSES = {
    "rest": make_pose(),
    "guard": make_pose(arm_r=(P([0.2, -0.3, 1.3]), P([0.4, -0.15, 1.4])), weapon=P([0.9, 0.0, 1.9]), crouch=0.08),
    # right-hand horizontal swings
    "windback_r": make_pose(arm_r=(P([-0.2, -0.45, 1.45]), P([-0.45, -0.5, 1.4])), weapon=P([-1.2, -0.6, 1.5]), twist=-0.6, crouch=0.1),
    "slash_end_l": make_pose(arm_r=(P([0.3, 0.1, 1.35]), P([0.45, 0.4, 1.3])), weapon=P([0.8, 1.2, 1.2]), twist=0.6, lean=0.15),
    "windback_l": make_pose(arm_r=(P([0.25, 0.25, 1.4]), P([0.2, 0.55, 1.45])), weapon=P([0.0, 1.4, 1.6]), twist=0.7, crouch=0.1),
    "slash_end_r": make_pose(arm_r=(P([0.3, -0.45, 1.3]), P([0.5, -0.7, 1.2])), weapon=P([0.6, -1.5, 1.1]), twist=-0.6, lean=0.15),
    # low sweep (jumpable)
    "low_windback": make_pose(arm_r=(P([-0.2, -0.45, 1.0]), P([-0.4, -0.5, 0.6])), weapon=P([-1.1, -0.8, 0.3]), twist=-0.8, crouch=0.35),
    "low_sweep_end": make_pose(arm_r=(P([0.3, 0.2, 0.9]), P([0.5, 0.45, 0.6])), weapon=P([0.9, 1.3, 0.25]), twist=0.8, crouch=0.35),
    # thrust
    "draw_back": make_pose(arm_r=(P([-0.3, -0.3, 1.2]), P([-0.25, -0.2, 1.15])), weapon=P([0.6, -0.15, 1.2]), crouch=0.15, lean=-0.1),
    "thrust_ext": make_pose(arm_r=(P([0.35, -0.2, 1.35]), P([0.75, -0.1, 1.35])), weapon=P([1.95, -0.05, 1.35]), lean=0.35),
    # overhead
    "raise_both": make_pose(arm_r=(P([0.05, -0.2, 2.0]), P([0.0, -0.05, 2.35])), arm_l=(P([0.05, 0.2, 2.0]), P([0.0, 0.05, 2.35])),
                            weapon=P([-0.6, 0.0, 3.0]), lean=-0.15),
    "slam_end": make_pose(arm_r=(P([0.45, -0.15, 1.1]), P([0.8, -0.05, 0.7])), arm_l=(P([0.45, 0.15, 1.1]), P([0.8, 0.05, 0.7])),
                          weapon=P([1.9, 0.0, 0.05]), lean=0.4, crouch=0.2),
    # jumps / leaps
    "crouch_deep": make_pose(crouch=0.45, lean=0.2, weapon=P([0.5, -0.4, 0.9])),
    "airborne": make_pose(arm_r=(P([0.0, -0.2, 2.0]), P([-0.1, -0.1, 2.3])), weapon=P([-0.7, 0.0, 2.9]), lean=-0.1),
    # beast
    "pounce_crouch": make_pose(crouch=0.5, lean=0.5, arm_r=(P([0.4, -0.3, 0.6]), P([0.6, -0.3, 0.2])), arm_l=(P([0.4, 0.3, 0.6]), P([0.6, 0.3, 0.2])),
                               weapon=P([0.7, -0.3, 0.3])),
    "bite_open": make_pose(lean=0.45, crouch=0.25, arm_r=(P([0.45, -0.35, 1.0]), P([0.8, -0.35, 1.2])), arm_l=(P([0.45, 0.35, 1.0]), P([0.8, 0.35, 1.2])),
                           weapon=P([1.0, -0.3, 1.3])),
    "bite_close": make_pose(lean=0.65, crouch=0.3, arm_r=(P([0.55, -0.15, 0.9]), P([0.95, -0.05, 1.0])), arm_l=(P([0.55, 0.15, 0.9]), P([0.95, 0.05, 1.0])),
                            weapon=P([1.2, -0.05, 1.0])),
    "spin_wind": make_pose(twist=1.4, crouch=0.2, arm_r=(P([0.0, 0.45, 1.2]), P([-0.2, 0.7, 1.1])), weapon=P([-0.6, 1.4, 0.9])),
    "spin_end": make_pose(twist=-1.4, crouch=0.2, arm_r=(P([0.1, -0.5, 1.2]), P([0.0, -0.8, 1.1])), weapon=P([-0.2, -1.6, 0.9])),
    # caster
    "cast_raise": make_pose(arm_l=(P([0.25, 0.3, 1.6]), P([0.45, 0.25, 1.95])), weapon=P([0.3, -0.3, 1.0]), lean=-0.05),
    "cast_throw": make_pose(arm_l=(P([0.4, 0.2, 1.45]), P([0.85, 0.1, 1.5])), weapon=P([0.3, -0.3, 1.0]), lean=0.2),
    "ring_charge": make_pose(arm_r=(P([0.1, -0.5, 1.7]), P([0.1, -0.8, 2.1])), arm_l=(P([0.1, 0.5, 1.7]), P([0.1, 0.8, 2.1])),
                             weapon=P([0.0, -1.0, 2.6]), crouch=0.05),
    "ring_burst": make_pose(arm_r=(P([0.1, -0.55, 0.9]), P([0.1, -0.85, 0.6])), arm_l=(P([0.1, 0.55, 0.9]), P([0.1, 0.85, 0.6])),
                            weapon=P([0.2, -1.2, 0.3]), crouch=0.35),
    # giant
    "stomp_lift": make_pose(legs={"r_knee": P([0.35, -0.15, 1.0]), "r_foot": P([0.45, -0.15, 0.7])}, lean=-0.1),
    "stomp_down": make_pose(legs={"r_knee": P([0.35, -0.15, 0.55]), "r_foot": P([0.55, -0.15, 0.0])}, crouch=0.2, lean=0.2),
    "grab_reach": make_pose(arm_r=(P([0.45, -0.25, 1.3]), P([0.9, -0.15, 1.1])), arm_l=(P([0.45, 0.25, 1.3]), P([0.9, 0.15, 1.1])),
                            weapon=P([0.4, -0.5, 0.9]), lean=0.3),
    "grab_close": make_pose(arm_r=(P([0.6, -0.1, 1.25]), P([1.2, 0.0, 1.15])), arm_l=(P([0.6, 0.1, 1.25]), P([1.2, 0.0, 1.15])),
                            weapon=P([0.4, -0.5, 0.9]), lean=0.5),
    "step_a": make_pose(legs={"r_knee": P([0.2, -0.15, 0.5]), "r_foot": P([0.3, -0.15, 0.0]),
                              "l_knee": P([-0.1, 0.15, 0.5]), "l_foot": P([-0.25, 0.15, 0.0])}),
    "step_b": make_pose(legs={"r_knee": P([-0.1, -0.15, 0.5]), "r_foot": P([-0.25, -0.15, 0.0]),
                              "l_knee": P([0.2, 0.15, 0.5]), "l_foot": P([0.3, 0.15, 0.0])}),
    "stagger": make_pose(crouch=0.4, lean=-0.3, arm_r=(P([-0.1, -0.4, 1.0]), P([-0.1, -0.5, 0.7])), weapon=P([-0.3, -1.0, 0.1])),
}


def ease(t):
    """Smoothstep in [0, 1]."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def lerp_pose(a, b, t):
    return a + (b - a) * ease(t)
