"""Arena: a continuous 2D souls-like boss fight at 30 Hz.

The player fights locked on (facing the boss). The boss is an animated 17-joint skeleton driven by
its moveset (arena/moves.py): wind-up poses, randomly long holds, strikes with geometric hitboxes
(arcs, lines, discs, rings, projectiles), combo strings, tracking, leaps, phase-2 moves, input reading,
and a posture bar that staggers it when full.

Player actions: a = move_dir * 7 + button (63 total)
  move_dir 0 none, 1..8 = forward, fwd-left, left, back-left, back, back-right, right, fwd-right
            (directions in the lock-on frame; forward = toward the boss)
  button   0 none, 1 roll, 2 light, 3 heavy, 4 guard (first ticks parry), 5 jump, 6 heal

Observation (88 floats, egocentric lock-on frame): boss keypoints (17 x 3, noisy, occasional
dropout), boss facing, 4 nearest projectiles, boss bars, 3 audio events (grunt, whoosh, impact),
own body state. Boss-related inputs can be
delayed by `latency` ticks (perception delay); the body state is always current.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

from .moves import BossSpec, Hitbox
from .skeleton import N_JOINTS, POSES, lerp_pose

NONE, ROLL, LIGHT, HEAVY, GUARD, JUMP, HEAL = range(7)
BUTTONS = ["none", "roll", "light", "heavy", "guard", "jump", "heal"]
N_DIRS, N_BUTTONS = 9, 7
N_ACTIONS = N_DIRS * N_BUTTONS
DIR_ANGLES = np.array([0.0] + [i * np.pi / 4 for i in range(8)])   # 0 = forward (toward boss), + = left
ARENA_R = 15.0
PLAYER_R = 0.4
BOSS_R = 0.6


@dataclass(frozen=True)
class Body:
    """Player body rules (a 'build'). Different builds = different embodiments to learn."""
    walk: float = 0.12
    roll_ticks: int = 14
    iframe_start: int = 2
    iframe_len: int = 8
    roll_dist: float = 3.0
    roll_cost: float = 18.0
    light_ticks: int = 13
    light_hit: int = 6
    light_dmg: float = 18.0
    light_reach: float = 2.2
    light_cost: float = 12.0
    light_posture: float = 10.0
    heavy_ticks: int = 26
    heavy_hit: int = 16
    heavy_dmg: float = 45.0
    heavy_reach: float = 2.6
    heavy_cost: float = 26.0
    heavy_posture: float = 28.0
    guard_ticks: int = 8
    parry_ticks: int = 4
    jump_ticks: int = 16
    air_start: int = 3
    air_len: int = 10
    heal_ticks: int = 40
    heal_at: int = 28
    heal_amount: float = 40.0
    flasks: int = 3
    stamina_regen: float = 1.2
    max_stamina: float = 100.0
    max_hp: float = 100.0
    max_poise: float = 50.0


BUILDS = {"medium": Body(),
          "light": Body(walk=0.14, roll_dist=3.4, iframe_len=9, roll_cost=15.0, max_poise=30.0, max_hp=85.0),
          "heavy": Body(walk=0.10, roll_ticks=18, roll_dist=2.2, iframe_start=3, iframe_len=5, roll_cost=24.0,
                        max_poise=90.0, max_hp=120.0, heavy_dmg=55.0)}


def _ang(v):
    return np.arctan2(v[1], v[0])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _rot(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


@dataclass
class Player:
    pos: np.ndarray
    hp: float
    stamina: float
    poise: float
    flasks: int
    button: int = NONE
    t: int = 0                 # ticks into the current busy action (0 = free)
    length: int = 0
    dir: float = 0.0           # world angle of the roll / jump direction
    stagger: int = 0


@dataclass
class BossState:
    pos: np.ndarray
    facing: float
    hp: float
    posture: float = 0.0
    move: int = -1             # -1 = neutral (walking / strafing)
    seg: int = 0
    t: int = 0
    seg_len: int = 1
    pose0: np.ndarray = None
    pose: np.ndarray = None
    neutral_left: int = 20
    strafe: float = 1.0
    leap_target: np.ndarray = None
    stagger: int = 0
    since_hit: int = 0
    hit_done: bool = False


@dataclass
class Projectile:
    pos: np.ndarray
    ang: float
    hb: Hitbox
    life: int


class Arena:
    def __init__(self, boss: BossSpec, body: Body = Body(), seed: int | None = None, latency: int = 0,
                 kp_noise: float = 0.03, dropout: float = 0.02, max_ticks: int = 30 * 180):
        self.spec, self.body, self.latency = boss, body, latency
        self.kp_noise, self.dropout, self.max_ticks = kp_noise, dropout, max_ticks
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self):
        B = self.body
        self.t = 0
        self.player = Player(np.array([-4.0, 0.0]), B.max_hp, B.max_stamina, B.max_poise, B.flasks)
        self.boss = BossState(np.array([4.0, 0.0]), np.pi, self.spec.stats["hp"])
        self.boss.pose = POSES["rest"].copy()
        self.boss.pose0 = self.boss.pose
        self.projectiles: list = []
        self.done, self.outcome = False, None
        self.recent = deque(maxlen=20)        # player's recent buttons (for boss input reading)
        self.events = []                      # (kind, details) for learners: hits taken/landed, parries...
        self._obs_q = deque()
        self.audio = np.zeros(3)
        obs = self._observe()
        return obs

    def clone(self, seed=None) -> "Arena":
        c = object.__new__(Arena)
        c.spec, c.body, c.latency, c.kp_noise, c.dropout, c.max_ticks = (self.spec, self.body, 0, 0.0, 0.0, self.max_ticks)
        c.rng = np.random.default_rng(seed)
        c.t, c.done, c.outcome = self.t, self.done, self.outcome
        c.player = replace(self.player, pos=self.player.pos.copy())
        b = self.boss
        c.boss = replace(b, pos=b.pos.copy(), leap_target=None if b.leap_target is None else b.leap_target.copy())
        c.projectiles = [Projectile(p.pos.copy(), p.ang, p.hb, p.life) for p in self.projectiles]
        c.recent = deque(self.recent, maxlen=20)
        c.events = []
        c._obs_q = deque()
        c.audio = np.zeros(3)
        return c

    @property
    def phase(self) -> int:
        return int(self.boss.hp <= 0.5 * self.spec.stats["hp"])

    def player_free(self) -> bool:
        return self.player.t == 0 and self.player.stagger == 0

    # ------------------------------------------------------------------ step
    def step(self, action: int):
        assert not self.done
        d, btn = divmod(int(action), N_BUTTONS)
        p, B, b = self.player, self.body, self.boss
        hp0, bhp0 = p.hp, b.hp
        self.audio = np.zeros(3)        # grunt (move start, sometimes), whoosh (strike start), impact
        to_boss = _ang(b.pos - p.pos)
        # ---- player
        if p.stagger > 0:
            p.stagger -= 1
        elif p.t == 0:
            cost = {ROLL: B.roll_cost, LIGHT: B.light_cost, HEAVY: B.heavy_cost}.get(btn, 0.0)
            if btn == HEAL and p.flasks <= 0:
                btn = NONE
            if btn != NONE and p.stamina < max(cost, 1.0 if btn == GUARD else 0.0):
                btn = NONE
            if btn != NONE:
                p.button, p.t, p.stamina = btn, 0, p.stamina - cost
                p.length = {ROLL: B.roll_ticks, LIGHT: B.light_ticks, HEAVY: B.heavy_ticks, GUARD: B.guard_ticks,
                            JUMP: B.jump_ticks, HEAL: B.heal_ticks}[btn]
                if btn == ROLL and d == 0:          # no direction: backstep
                    p.dir = to_boss + np.pi
                else:
                    p.dir = to_boss + DIR_ANGLES[d]
                self.recent.append(btn)
            elif d > 0:
                p.pos = p.pos + B.walk * np.array([np.cos(to_boss + DIR_ANGLES[d]), np.sin(to_boss + DIR_ANGLES[d])])
                self.recent.append(NONE)
        if p.button != NONE and p.stagger == 0:
            p.t += 1
            self._player_action_tick()
        if p.button in (ROLL, LIGHT, HEAVY):
            pass
        else:
            p.stamina = min(B.max_stamina, p.stamina + (0.3 if p.button == GUARD else B.stamina_regen))
        p.poise = min(B.max_poise, p.poise + 0.4)
        # ---- boss
        self._boss_tick()
        self._projectiles_tick()
        # ---- bounds, contact
        for pos in (p.pos, b.pos):
            r = np.linalg.norm(pos)
            if r > ARENA_R:
                pos *= ARENA_R / r
        sep = p.pos - b.pos
        dist = np.linalg.norm(sep)
        if dist < PLAYER_R + BOSS_R:
            p.pos = b.pos + sep / max(dist, 1e-6) * (PLAYER_R + BOSS_R)
        if p.button != NONE and p.t >= p.length:
            p.t, p.button, p.length = 0, NONE, 0
        self.t += 1
        reward = (bhp0 - b.hp) - 2.0 * (hp0 - p.hp)
        if b.hp <= 0:
            self.done, self.outcome = True, "win"
            reward += 100.0
        elif p.hp <= 0:
            self.done, self.outcome = True, "death"
            reward -= 100.0
        elif self.t >= self.max_ticks:
            self.done, self.outcome = True, "timeout"
        return self._observe(), reward, self.done, {"outcome": self.outcome}

    def _player_action_tick(self):
        p, B, b = self.player, self.body, self.boss
        dist = np.linalg.norm(b.pos - p.pos) - BOSS_R
        if p.button == ROLL:
            if p.t <= B.roll_ticks - 3:
                p.pos = p.pos + (B.roll_dist / (B.roll_ticks - 3)) * np.array([np.cos(p.dir), np.sin(p.dir)])
        elif p.button == JUMP:
            p.pos = p.pos + (1.0 / B.jump_ticks) * np.array([np.cos(p.dir), np.sin(p.dir)])
        elif p.button in (LIGHT, HEAVY):
            hit_t, reach, dmg, post = ((B.light_hit, B.light_reach, B.light_dmg, B.light_posture) if p.button == LIGHT
                                       else (B.heavy_hit, B.heavy_reach, B.heavy_dmg, B.heavy_posture))
            if p.t == hit_t and dist <= reach:
                crit = b.stagger > 0
                b.hp -= dmg * (2.5 if crit else 1.0)
                b.since_hit = 0
                if crit:
                    b.stagger, b.posture = 0, 0.0
                else:
                    b.posture += post
                self.events.append(("landed", self.t, p.button, crit))
        elif p.button == HEAL and p.t == B.heal_at:
            p.hp = min(B.max_hp, p.hp + B.heal_amount)
            p.flasks -= 1

    def in_iframes(self) -> bool:
        p, B = self.player, self.body
        return p.button == ROLL and B.iframe_start <= p.t < B.iframe_start + B.iframe_len

    def airborne(self) -> bool:
        p, B = self.player, self.body
        return p.button == JUMP and B.air_start <= p.t < B.air_start + B.air_len

    # ------------------------------------------------------------------ boss
    def _choose_move(self) -> int:
        b, p, spec = self.boss, self.player, self.spec
        rel = p.pos - b.pos
        dist = np.linalg.norm(rel) - BOSS_R
        behind = abs(_wrap(_ang(rel) - b.facing)) > 1.75
        w = np.zeros(len(spec.moves))
        for i, m in enumerate(spec.moves):
            if m.phase == 2 and not self.phase:
                continue
            lo, hi = m.ideal
            w[i] = 3.0 if lo <= dist <= hi else 0.12
            if behind:
                w[i] *= 4.0 if m.rear else 0.3
        for (act, i, boost) in spec.laws:
            code = HEAL if act == "heal" else ROLL
            if code in list(self.recent)[-8:] and w[i] > 0:
                w[i] *= np.exp(boost)
        return int(self.rng.choice(len(w), p=w / w.sum()))

    def _start_move(self, i: int):
        b = self.boss
        if self.rng.random() < 0.6:
            self.audio[0] = 1.0
        b.move, b.seg = i, -1
        self._next_segment()

    def _next_segment(self):
        b, spec = self.boss, self.spec
        m = spec.moves[b.move]
        b.seg += 1
        if b.seg >= len(m.segments):
            row = spec.cont[b.move]
            avail = row.copy()
            for j, mm in enumerate(spec.moves):
                if mm.phase == 2 and not self.phase:
                    avail[j] = 0.0
            nxt = int(self.rng.choice(len(avail), p=avail / avail.sum()))
            if nxt < len(spec.moves):
                self._start_move(nxt)
            else:
                b.move = -1
                b.neutral_left = int(np.clip(self.rng.lognormal(np.log(14), 0.5), 3, 60))
                b.strafe = float(self.rng.choice([-1.0, 1.0]))
                b.pose0, b.t, b.seg_len = b.pose.copy(), 0, 8
            return
        s = m.segments[b.seg]
        if s.kind == "hold":
            med = s.hold_median * (spec.phase2_speed if self.phase else 1.0)
            b.seg_len = int(np.clip(round(self.rng.lognormal(np.log(max(med, 1.0)), s.hold_sigma)), 1, 120))
        else:
            b.seg_len = s.ticks
        b.t, b.pose0, b.hit_done = 0, b.pose.copy(), False
        if s.kind == "strike":
            self.audio[1] = 1.0
        if s.leap:
            b.leap_target = self.player.pos.copy()

    def _boss_tick(self):
        b, p, spec = self.boss, self.player, self.spec
        st = spec.stats
        b.since_hit += 1
        if b.since_hit > 60:
            b.posture = max(0.0, b.posture - 0.4)
        if b.posture >= st["posture"] and b.stagger == 0:
            b.stagger, b.posture, b.move = 75, 0.0, -1
            b.pose0, b.t, b.seg_len = b.pose.copy(), 0, 10
            self.events.append(("posture_break", self.t))
        rel = p.pos - b.pos
        dist = np.linalg.norm(rel)
        target_ang = _ang(rel)
        if b.stagger > 0:
            b.stagger -= 1
            b.pose = lerp_pose(b.pose0, POSES["stagger"], b.t / 10)
            b.t += 1
            if b.stagger == 0:
                b.neutral_left, b.pose0, b.t = 10, b.pose.copy(), 0
            return
        if b.move < 0:                                   # neutral: approach / strafe, walk cycle
            b.facing += np.clip(_wrap(target_ang - b.facing), -st["turn"], st["turn"])
            if dist - BOSS_R > st["range"] + 0.6:
                b.pos = b.pos + st["walk"] * np.array([np.cos(b.facing), np.sin(b.facing)])
            elif dist - BOSS_R < st["range"] - 1.5:
                b.pos = b.pos - 0.5 * st["walk"] * np.array([np.cos(b.facing), np.sin(b.facing)])
            else:
                side = b.facing + b.strafe * np.pi / 2
                b.pos = b.pos + 0.4 * st["walk"] * np.array([np.cos(side), np.sin(side)])
            cyc = POSES["step_a"] if (self.t // 8) % 2 == 0 else POSES["step_b"]
            b.pose = lerp_pose(b.pose, cyc, 0.25)
            b.neutral_left -= 1
            if b.neutral_left <= 0:
                self._start_move(self._choose_move())
            return
        m = spec.moves[b.move]
        s = m.segments[b.seg]
        b.t += 1
        target = POSES[s.pose]
        b.pose = lerp_pose(b.pose0, target, b.t / (4 if s.kind == "hold" else max(1, b.seg_len)))
        if s.track:
            b.facing += np.clip(_wrap(target_ang - b.facing), -s.track, s.track)
        if s.travel:
            if s.leap and b.leap_target is not None:
                to = b.leap_target - b.pos
                remaining = max(0.0, np.linalg.norm(to) - 1.2)
                step = min(abs(s.travel) / b.seg_len, remaining / max(1, b.seg_len - b.t + 1))
                b.facing = _ang(to) if np.linalg.norm(to) > 1e-6 else b.facing
                b.pos = b.pos + step * np.array([np.cos(b.facing), np.sin(b.facing)])
            else:
                b.pos = b.pos + (s.travel / b.seg_len) * np.array([np.cos(b.facing), np.sin(b.facing)])
        if s.kind == "strike" and s.hitbox is not None and b.t - 1 == s.impact and not b.hit_done:
            b.hit_done = True
            self.audio[2] = 1.0
            if s.hitbox.shape == "projectile":
                from .skeleton import J
                hand = self._kp_world()[J["l_hand"]][:2]
                self.projectiles.append(Projectile(hand.copy(), _ang(p.pos - hand), s.hitbox, s.hitbox.life))
            elif self._in_hitbox(s.hitbox):
                self._take_hit(s.hitbox, m.name)
            elif np.linalg.norm(rel) < 6.0:
                self.events.append(("miss", self.t, m.name, b.seg, self._local(p.pos)))
        if b.t >= b.seg_len:
            self._next_segment()

    def _local(self, pos):
        b = self.boss
        return _rot(pos - b.pos, -b.facing)

    def _in_hitbox(self, hb: Hitbox) -> bool:
        q = self._local(self.player.pos)
        if hb.shape == "arc":
            r = np.linalg.norm(q)
            if r > hb.r + PLAYER_R:
                return False
            a = _ang(q)
            return hb.a1 - hb.a0 >= 2 * np.pi - 1e-6 or hb.a0 <= a <= hb.a1 or r < BOSS_R + PLAYER_R
        if hb.shape == "capsule":
            return hb.x0 - PLAYER_R <= q[0] <= hb.x1 + PLAYER_R and abs(q[1]) <= hb.w + PLAYER_R
        if hb.shape == "disc":
            r = np.linalg.norm(q - np.array([hb.cx, hb.cy]))
            return hb.r_in <= r <= hb.r + PLAYER_R
        return False

    def _take_hit(self, hb: Hitbox, name: str):
        p, B = self.player, self.body
        if self.in_iframes():
            self.events.append(("dodged", self.t, name, p.t))
            return
        if hb.low and self.airborne():
            self.events.append(("jumped", self.t, name))
            return
        dmg = hb.damage
        if p.button == GUARD and not hb.unblockable:
            if p.t <= B.parry_ticks:
                self.boss.posture += 40.0
                self.boss.since_hit = 0
                self.events.append(("parry", self.t, name))
                return
            p.stamina -= 0.6 * hb.poise
            if p.stamina < 0:
                p.stamina, p.stagger = 0.0, 30
                p.button, p.t, p.length = NONE, 0, 0
                dmg *= 0.5
                self.events.append(("guard_break", self.t, name))
            else:
                dmg *= 0.1
                self.events.append(("blocked", self.t, name))
        else:
            p.poise -= hb.poise
            if p.poise <= 0:
                p.stagger, p.poise = 18, B.max_poise
                p.button, p.t, p.length = NONE, 0, 0
            self.events.append(("hit", self.t, name, self._local(p.pos)))
        p.hp -= dmg

    def _projectiles_tick(self):
        p = self.player
        keep = []
        for pr in self.projectiles:
            if pr.hb.homing:
                pr.ang += np.clip(_wrap(_ang(p.pos - pr.pos) - pr.ang), -pr.hb.homing, pr.hb.homing)
            pr.pos = pr.pos + pr.hb.speed * np.array([np.cos(pr.ang), np.sin(pr.ang)])
            pr.life -= 1
            if np.linalg.norm(pr.pos - p.pos) <= pr.hb.r + PLAYER_R:
                self._take_hit(pr.hb, "projectile")
                continue
            if pr.life > 0 and np.linalg.norm(pr.pos) < ARENA_R + 2:
                keep.append(pr)
        self.projectiles = keep

    # ------------------------------------------------------------------ observation
    def _kp_world(self):
        b = self.boss
        c, s = np.cos(b.facing), np.sin(b.facing)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return b.pose @ R.T + np.array([b.pos[0], b.pos[1], 0.0])

    def _boss_part(self):
        """Boss-related observation in the player's lock-on frame (what perception delay applies to)."""
        p, b = self.player, self.boss
        lock = _ang(b.pos - p.pos)
        kp = self._kp_world()
        rel = kp[:, :2] - p.pos
        c, s = np.cos(-lock), np.sin(-lock)
        rel = rel @ np.array([[c, -s], [s, c]]).T
        kp_obs = np.concatenate([rel, kp[:, 2:3]], 1)
        if self.kp_noise:
            kp_obs = kp_obs + self.rng.normal(0, self.kp_noise, kp_obs.shape)
        if self.dropout:
            kp_obs[self.rng.random(N_JOINTS) < self.dropout] = 0.0
        face = _wrap(b.facing - (lock + np.pi))          # 0 = boss faces the player
        proj = np.zeros((4, 4))
        prs = sorted(self.projectiles, key=lambda q: np.linalg.norm(q.pos - p.pos))[:4]
        for i, pr in enumerate(prs):
            rp = _rot(pr.pos - p.pos, -lock)
            vel = _rot(pr.hb.speed * np.array([np.cos(pr.ang), np.sin(pr.ang)]), -lock)
            proj[i] = [rp[0], rp[1], vel[0] * 10, vel[1] * 10]
        return np.concatenate([kp_obs.ravel(), [np.cos(face), np.sin(face)], proj.ravel(),
                               [b.hp / self.spec.stats["hp"], b.posture / self.spec.stats["posture"]], self.audio])

    def _body_part(self):
        p, B = self.player, self.body
        onehot = np.zeros(N_BUTTONS)
        onehot[p.button] = 1.0
        return np.concatenate([[p.hp / 100.0, p.stamina / 100.0, p.poise / 100.0, p.flasks / 3.0,
                                p.t / 30.0, float(p.stagger > 0), np.linalg.norm(p.pos) / ARENA_R], onehot])

    def _observe(self):
        bp = self._boss_part()
        self._obs_q.append(bp)
        while len(self._obs_q) > self.latency + 1:
            self._obs_q.popleft()
        seen = self._obs_q[0]
        return np.concatenate([seen, self._body_part()]).astype(np.float32)

    def truth(self):
        b = self.boss
        return {"move": b.move, "seg": b.seg, "t": b.t, "seg_len": b.seg_len, "stagger": b.stagger}


OBS_DIM = N_JOINTS * 3 + 2 + 16 + 2 + 3 + 7 + N_BUTTONS
