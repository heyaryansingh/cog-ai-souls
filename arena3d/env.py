"""Arena3D environment: 3D mocap boss vs. a humanoid player; harm by per-substep 3D contact.

Frame = 30 ms (mocap rate). Each frame is checked at `substeps` interpolated poses, so a fast blade
cannot tunnel through a limb. A hit needs the blade (or a fast-moving boss limb) to touch a player body
capsule while moving; damage scales with contact speed and the body part struck. Rolling gives i-frames,
a raised shield blocks contact geometrically (the blade must touch the shield first).

Observations are what a player has: rendered pixels from a lock-on camera behind the player, three sound
events (move-start grunt, swing whoosh, impact), and proprioception. Ground-truth boss keypoints are in
`info` for scoring perception, not for the agent.
"""
from __future__ import annotations

import os
import mujoco
import numpy as np

from . import mocap
from .boss import BossRunner, BossSpec, _qz, heading
from .world import build_world

NONE, ROLL, LIGHT, GUARD, HEAL = range(5)
GUARD_REAL = os.environ.get("ARENA_GUARD_REAL") == "1"   # chip damage + guard break (default off: old runs unchanged)
N_BUTTONS = 5
N_ACTIONS = 9 * N_BUTTONS          # 9 directions (0 = none, 1..8 lock-on frame, 1 = toward boss) x buttons
KP_BODIES = ["root", "lfemur", "ltibia", "lfoot", "rfemur", "rtibia", "rfoot", "lowerback", "thorax", "head",
             "lhumerus", "lradius", "lhand", "rhumerus", "rradius", "rhand"]
PART_MULT = {"head": 1.5, "upperneck": 1.4, "lowerneck": 1.4, "thorax": 1.1, "upperback": 1.1, "lowerback": 1.0,
             "root": 1.0}

# player clips (found automatically in the library, see scripts/arena3d_demo.py)
WALK = ("CMU_143_40", 39, 129)
IDLE = ("CMU_143_40", 39)
CROUCH = ("CMU_139_17", 121)


class Arena3D:
    def __init__(self, spec: BossSpec, units, clips, seed=0, substeps=4, render=False, width=96, height=96,
                 max_frames=None, player_strike=None):
        import os
        max_frames = max_frames or int(os.environ.get("ARENA_MAX_FRAMES", 6000))
        self.spec, self.U, self.C = spec, units, clips
        self.m, _ = build_world(spec.scale, spec.weapon, width, height)
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        self.sub, self.max_frames = substeps, max_frames
        self.renderer = mujoco.Renderer(self.m, height, width) if render else None
        self.cam = mujoco.MjvCamera()
        m = self.m
        self.blade = m.geom("b_blade").id if spec.weapon != "fist" else -1
        self.pblade = m.geom("p_blade").id
        self.shield = m.geom("p_shield").id
        self.p_geoms = {i for i in range(m.ngeom) if m.geom(i).name.startswith("p_") and i not in (self.pblade, self.shield)}
        self.b_geoms = {i for i in range(m.ngeom) if m.geom(i).name.startswith("b_") and i != self.blade}
        self.tip = m.site("b_tip").id if self.blade >= 0 else m.body("b_rhand").id
        self.kp_b = [m.body("b_" + n).id for n in KP_BODIES]
        self.player_strike = player_strike
        self.walk = clips[WALK[0]]
        self.reset()

    # ------------------------------------------------------------------------------------------------
    def reset(self):
        if getattr(self.spec, "program", None) is not None:
            from .program import ProgramRunner
            self.boss = ProgramRunner(self.spec, self.U, self.C, self.rng)
        else:
            self.boss = BossRunner(self.spec, self.U, self.C, self.rng)
        self.projectiles, self.rings, self.swing_id, self.fired = [], [], 0, -1
        self.boss.reset(np.array([0.0, 5.0 + self.spec.scale]), -np.pi / 2)
        self.p_xy, self.p_yaw = np.array([0.0, 0.0]), np.pi / 2
        self.hp, self.stamina, self.flasks = 100.0, 100.0, 4
        self.stagger = 0
        self.boss_hp, self.phase = self.spec.hp, False
        self.t, self.walk_phase = 0, 0.0
        self.act, self.act_t, self.act_dir = NONE, 0, np.zeros(2)
        self.hit_lock = {}                       # (source, move instance) -> frames
        self.move_serial = 0
        self.p_serial = 0
        self.near = []                           # per boss strike: min blade-player distance (diagnostic)
        self.prev_unit = None
        self.audio = np.zeros(3, np.float32)
        self.prev_tip = None
        qb = self.boss.step(self.p_xy, False)
        self.d.qpos[:63] = qb
        self.d.qpos[63:] = self._player_q()
        mujoco.mj_fwdPosition(self.m, self.d)
        self.prev_qpos = self.d.qpos.copy()
        return self.observe()

    # --- player body -----------------------------------------------------------------------------
    def _player_q(self):
        if self.act == ROLL:
            c, t = self.C[CROUCH[0]], CROUCH[1]
        elif self.act == LIGHT and self.player_strike is not None:
            u = self.player_strike
            c, t = self.C[u.clip], u.start + min(self.act_t, u.length - 1)
        elif np.linalg.norm(self.act_dir) > 0 and self.act in (NONE, GUARD):
            c = self.walk
            n = WALK[2] - WALK[1]
            t = WALK[1] + int(self.walk_phase) % n
        else:
            c, t = self.C[IDLE[0]], IDLE[1]
        j = c.joints[t].copy()
        # yaw: undo the clip's own heading, face p_yaw
        q = _qmul_np(_qz(self.p_yaw - heading(c.quat[t])), c.quat[t])
        z = c.pos[t, 2]
        if self.act == GUARD:
            j = self._guard_pose(j)
        return np.concatenate([[self.p_xy[0], self.p_xy[1], z], q / np.linalg.norm(q), j])

    def _guard_pose(self, j):
        # raise the left forearm across the chest (shield in front): set left shoulder / elbow joints
        names = [self.m.joint(i).name for i in range(63 // 63 * 0, 0)]
        jj = j.copy()
        for nm, v in (("lhumerusrz", 0.9), ("lhumerusry", -0.4), ("lhumerusrx", -1.2), ("lradiusrx", 1.7)):
            k = self._jidx(nm)
            if k is not None:
                jj[k] = v
        return jj

    def _jidx(self, name):
        if not hasattr(self, "_jmap"):
            self._jmap = {self.m.joint(i).name[2:]: self.m.jnt_qposadr[i] - 63 - 7
                          for i in range(self.m.njnt) if self.m.joint(i).name.startswith("p_") and
                          self.m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE}
        return self._jmap.get(name)

    def in_iframes(self):
        return self.act == ROLL and 2 <= self.act_t <= 11

    def player_free(self):
        if getattr(self, "stagger", 0) > 0:
            return False
        return self.act in (NONE, GUARD) or (self.act == ROLL and self.act_t >= 16) or \
            (self.act == LIGHT and self.act_t >= 22) or (self.act == HEAL and self.act_t >= 40)

    # --- step --------------------------------------------------------------------------------------
    def step(self, a):
        d_i, btn = divmod(int(a), N_BUTTONS)
        to_b = self.boss.root - self.p_xy
        face = np.arctan2(to_b[1], to_b[0])
        if self.player_free():
            if btn == ROLL and self.stamina >= 18:
                self.act, self.act_t = ROLL, 0
                self.stamina -= 18
                dd = d_i if d_i else 5
                ang = face + (dd - 1) * np.pi / 4
                self.act_dir = np.array([np.cos(ang), np.sin(ang)])
            elif btn == LIGHT and self.stamina >= 12:
                self.act, self.act_t, self.stamina = LIGHT, 0, self.stamina - 12
                self.act_dir = np.zeros(2)
                self.p_serial += 1
            elif btn == HEAL and self.flasks > 0:
                self.act, self.act_t, self.flasks = HEAL, 0, self.flasks - 1
                self.act_dir = np.zeros(2)
            else:
                self.act = GUARD if btn == GUARD else NONE
                self.act_t = 0
                if d_i:
                    ang = face + (d_i - 1) * np.pi / 4
                    self.act_dir = np.array([np.cos(ang), np.sin(ang)])
                else:
                    self.act_dir = np.zeros(2)
        # player motion
        spd = {NONE: 4.0, GUARD: 2.0, ROLL: 6.5 if self.act_t < 14 else 0.0, LIGHT: 0.0, HEAL: 0.8}[self.act] * 0.03
        self.p_xy = self.p_xy + self.act_dir * spd
        self.p_xy = np.clip(self.p_xy, -15, 15)
        self.walk_phase += 1.2 if np.linalg.norm(self.act_dir) > 0 else 0
        self.p_yaw = face
        if self.act == HEAL and self.act_t == 25:
            self.hp = min(100.0, self.hp + 40)
        if self.act in (NONE, GUARD):
            self.stamina = min(100.0, self.stamina + (0.6 if self.act == GUARD else 1.5))
        # boss
        unit_before = self.boss.unit
        self.boss.player_healing = self.act == HEAL
        qb = self.boss.step(self.p_xy, self.phase)
        self.audio[:] = 0
        if self.boss.unit is not unit_before and self.boss.kind == "strike":
            self.move_serial += 1
            self.boss_serial = self.move_serial
            self.boss_onset_t = self.t
            if self.rng.random() < 0.6:
                self.audio[0] = 1
        q_new = np.concatenate([qb, self._player_q()])
        reward, info = self._resolve(q_new)
        reward += self._effects(info)
        self.act_t += 1
        if getattr(self, "stagger", 0) > 0:
            self.stagger -= 1
        if self.act in (ROLL, LIGHT, HEAL) and self.player_free() and self.act != GUARD:
            pass
        if self.act == ROLL and self.act_t >= 20 or self.act == LIGHT and self.act_t >= 30 or \
                self.act == HEAL and self.act_t >= 50:
            self.act, self.act_t = NONE, 0
        self.t += 1
        if not self.phase and self.boss_hp < 0.5 * self.spec.hp and self.spec.phase2:
            self.phase = True
        done = self.hp <= 0 or self.boss_hp <= 0 or self.t >= self.max_frames
        if done:
            info["outcome"] = "win" if self.boss_hp <= 0 else ("death" if self.hp <= 0 else "timeout")
            reward += 100.0 if self.boss_hp <= 0 else (-100.0 if self.hp <= 0 else -50.0)
        return self.observe(), reward, done, info

    def _resolve(self, q_new):
        """Interpolate poses over substeps; detect blade/limb contacts; apply damage once per move."""
        m, d = self.m, self.d
        q_old = self.prev_qpos
        reward, info = 0.0, {"hit": None, "blocked": False, "dodged": False}
        tip_old = d.site_xpos[self.tip].copy() if self.blade >= 0 else d.xpos[self.tip].copy()
        boss_serial = getattr(self, "boss_serial", -1)
        for k in range(1, self.sub + 1):
            w = k / self.sub
            q = (1 - w) * q_old + w * q_new
            for adr in (3, 66):                      # renormalise root quaternions
                q[adr:adr + 4] /= np.linalg.norm(q[adr:adr + 4])
            d.qpos[:] = q
            mujoco.mj_fwdPosition(m, d)
            tip = d.site_xpos[self.tip].copy() if self.blade >= 0 else d.xpos[self.tip].copy()
            tip_v = min(np.linalg.norm(tip - tip_old) / (0.03 / self.sub), 25.0)
            tip_old = tip
            # whoosh: onset of a fast swing (crossing 5 m/s upward), once per crossing
            fast = tip_v > 5.0 and self.boss.kind == "strike"
            if fast and not getattr(self, "_was_fast", False):
                self.audio[1] = 1
                self.swing_id += 1          # a new swing inside the same move can hit again (multi-hit)
            self._was_fast = fast
            shield_touch, body_touch, pbl_touch = False, None, None
            for c in d.contact[:d.ncon]:
                g1, g2 = c.geom1, c.geom2
                pair = {g1, g2}
                if self.blade in pair:
                    other = g2 if g1 == self.blade else g1
                    if other == self.shield:
                        shield_touch = True
                    elif other in self.p_geoms:
                        body_touch = other
                if self.pblade in pair:
                    other = g2 if g1 == self.pblade else g1
                    if other in self.b_geoms:
                        pbl_touch = other
            mv = getattr(self.boss, "cur_move", None)
            melee = not (mv is not None and mv.projectile)
            key = ("b", boss_serial, self.swing_id)
            if melee and (shield_touch or body_touch is not None) and tip_v > 2.5 and key not in self.hit_lock:
                self.hit_lock[key] = 1
                self.audio[2] = 1
                # souls rule: a raised guard blocks any hit from the front half (angle check, as in
                # Elden Ring), not only blade-on-shield contact
                to_b = self.boss.root - self.p_xy
                front = np.cos(np.arctan2(to_b[1], to_b[0]) - self.p_yaw) > 0.0
                if self.act == GUARD and (shield_touch or front):
                    self.stamina -= 25 + 2 * tip_v
                    info["blocked"] = True
                    if GUARD_REAL:
                        # Elden Ring-like: a medium shield lets ~25% through; a guard break staggers
                        chip = 0.25 * (8 + 0.9 * tip_v) * (self.spec.scale / 1.6) ** 0.5
                        if self.stamina < 0:
                            chip *= 2; self.stagger = 40; self.act, self.act_t = NONE, 0
                            self.act_dir = np.zeros(2); info["guard_break"] = True
                        self.hp -= chip; reward -= chip
                        self.stamina = max(self.stamina, 0.0)
                    elif self.stamina < 0:
                        self.stamina = 0
                        self.hp -= 10
                elif self.in_iframes():
                    info["dodged"] = True
                elif body_touch is not None:
                    part = m.body(m.geom_bodyid[body_touch]).name[2:]
                    dmg = (8 + 0.9 * tip_v) * PART_MULT.get(part, 0.8) * (self.spec.scale / 1.6) ** 0.5
                    self.hp -= dmg
                    reward -= dmg
                    info["hit"] = (part, float(dmg))
                    if self.act in (LIGHT, HEAL):
                        self.act, self.act_t = NONE, 0       # interrupted
            if pbl_touch is not None and self.act == LIGHT and ("p", self.p_serial) not in self.hit_lock:
                self.hit_lock[("p", self.p_serial)] = 1
                part = m.body(m.geom_bodyid[pbl_touch]).name[2:]
                dmg = 50.0 * PART_MULT.get(part, 0.8)
                self.boss_hp -= dmg
                reward += dmg * 100.0 / self.spec.hp
                info["boss_hit"] = (part, dmg)
        self.prev_qpos = q_new.copy()
        return reward, info

    def _guarded_front(self):
        to_b = self.boss.root - self.p_xy
        return self.act == GUARD and np.cos(np.arctan2(to_b[1], to_b[0]) - self.p_yaw) > 0.0

    def _effects(self, info):
        """Projectiles (dagger throws) and shockwave rings, spawned at a program move's danger frame."""
        b, reward = self.boss, 0.0
        mv = getattr(b, "cur_move", None)
        serial = getattr(self, "boss_serial", -1)
        if mv is not None and b.kind == "strike" and b.f >= b.peak and self.fired != serial:
            self.fired = serial
            hand = self.d.xpos[self.m.body("b_rhand").id].copy()
            if mv.projectile:
                tgt = np.array([self.p_xy[0], self.p_xy[1], 1.1])
                v = tgt - hand
                self.projectiles.append({"pos": hand, "vel": v / (np.linalg.norm(v) + 1e-6) * 14.0, "life": 50})
            if mv.ring > 0:
                tip = self.d.site_xpos[self.tip] if self.blade >= 0 else hand
                self.rings.append({"c": tip[:2].copy(), "r": 0.3, "R": mv.ring, "hit": False})
                self.audio[2] = 1
        prot = self.in_iframes()
        for pr in self.projectiles:
            pr["pos"] = pr["pos"] + pr["vel"] * 0.03
            pr["life"] -= 1
            d_xy = np.linalg.norm(pr["pos"][:2] - self.p_xy)
            if d_xy < 0.35 and 0.1 < pr["pos"][2] < 1.8:
                pr["life"] = 0
                if prot:
                    info["dodged"] = True
                elif self._guarded_front():
                    self.stamina = max(0.0, self.stamina - 15); info["blocked"] = True
                else:
                    self.hp -= 14; reward -= 14; info["hit"] = ("projectile", 14.0)
        self.projectiles = [p for p in self.projectiles if p["life"] > 0]
        for rg in self.rings:
            rg["r"] += (rg["R"] - 0.3) / 12
            if not rg["hit"] and abs(np.linalg.norm(self.p_xy - rg["c"]) - rg["r"]) < 0.45:
                rg["hit"] = True
                if prot:
                    info["dodged"] = True
                elif self._guarded_front():
                    self.stamina = max(0.0, self.stamina - 30); info["blocked"] = True
                else:
                    self.hp -= 20; reward -= 20; info["hit"] = ("shockwave", 20.0)
        self.rings = [r for r in self.rings if r["r"] < r["R"]]
        return reward

    # --- observation -----------------------------------------------------------------------------
    def observe(self):
        body = np.array([self.hp / 100, self.stamina / 100, self.flasks / 3, self.act, self.act_t / 30,
                         self.in_iframes()], np.float32)
        obs = {"audio": self.audio.copy(), "body": body}
        if self.renderer is not None:
            obs["pixels"] = self.render()
        return obs

    def camera(self):
        to_b = self.boss.root - self.p_xy
        dist = np.linalg.norm(to_b)
        az = np.degrees(np.arctan2(to_b[1], to_b[0]))
        cam = self.cam
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        mid = self.p_xy + to_b * 0.6
        cam.lookat[:] = [mid[0], mid[1], 0.9 + 0.35 * self.spec.scale]
        cam.distance = 4.2 + 0.6 * dist
        cam.azimuth = az + 8          # slightly over the right shoulder, as in souls games
        cam.elevation = -16
        return cam

    def render(self, renderer=None):
        r = renderer or self.renderer
        r.update_scene(self.d, self.camera())
        return r.render()

    def clone(self, seed=None):
        """Independent copy for imagination (shares the model, copies data, boss state and RNG)."""
        import copy
        c = copy.copy(self)
        c.d = copy.deepcopy(self.d)
        c.renderer = None
        c.boss = copy.copy(self.boss)
        c.boss.rng = np.random.default_rng(seed) if seed is not None else copy.deepcopy(self.boss.rng)
        c.rng = np.random.default_rng(None if seed is None else seed + 1)
        for k in ("p_xy", "act_dir", "audio", "prev_qpos"):
            setattr(c, k, np.array(getattr(self, k), copy=True))
        c.boss.root = self.boss.root.copy()
        c.boss.origin = getattr(self.boss, "origin", self.boss.root).copy()
        c.boss.lunge_dir = getattr(self.boss, "lunge_dir", np.zeros(2)).copy()
        c.hit_lock = dict(self.hit_lock)
        # program bosses keep mutable plans (string queue); imagination must not consume the real boss's queue
        if hasattr(self.boss, "queue"):
            c.boss.queue = list(self.boss.queue)
        c.projectiles = [dict(p, pos=p["pos"].copy(), vel=p["vel"].copy()) for p in self.projectiles]
        c.rings = [dict(r, c=r["c"].copy()) for r in self.rings]
        return c

    def truth(self):
        """Ground truth for scoring: boss 3D keypoints, current unit, strike id, frame in unit."""
        b = self.boss
        return {"kp": self.d.xpos[self.kp_b].copy(), "tip": self.d.site_xpos[self.tip].copy() if self.blade >= 0 else None,
                "unit": None if b.unit is None else (b.unit.clip, b.unit.start), "move": b.move_id,
                "kind": b.kind if b.unit is not None else None, "f": getattr(b, "f", 0.0),
                "hold": b.hold if b.unit is not None else 0}


def _qmul_np(a, b):
    return mocap._qmul(a, b)
