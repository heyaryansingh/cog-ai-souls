"""B65 Mirror simulation: understand the boss by re-enacting what you have seen it do.

Motor-simulation theories of action understanding (mirror systems, "simulation theory") say we predict others'
actions by running them on our own body model. Here the agent keeps an episodic memory of every boss move it has
watched: the boss's perceived full-body pose over the move (noisy pose estimate), expressed in the move's onset
frame (origin = boss position at onset, x-axis = boss->player line at onset), plus when shockwaves/projectiles
appeared. To imagine a future it re-enacts a remembered episode of the hypothesised move on its own copy of the
arena (the same body/contact physics it lives in: capsules, blade, i-frames, rings), aligned to the current boss
position, facing and time since onset, and checks real 3D contact against its own imagined body. No boss program,
mocap library or boss simulator is used: identity comes from noisy reading (70% at 12 frames) and learned strings,
motion from memory.
"""
from __future__ import annotations

import os
import types

import numpy as np

from .env import N_BUTTONS, NONE, ROLL, LIGHT, GUARD, HEAL, HEAVY, BACKSTEP, REAL2
from .mocap import _qmul
from .boss import _qz, heading


def _yaw_of(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class MirrorMemory:
    def __init__(self, n_moves, seed=0, read_k=12, acc=0.7, keep=10, noise_j=0.03, noise_p=0.03):
        self.n, self.read_k, self.acc, self.keep = n_moves, read_k, acc, keep
        self.nj, self.np_ = noise_j, noise_p
        self.rng = np.random.default_rng(seed)
        self.eps = {}            # label -> list of episodes
        self.ring_R = {}         # label -> observed final ring radii
        self.update = True

    @property
    def turn(self):
        d = getattr(self, "dyaw", [])
        return float(np.percentile(d, 75)) if len(d) > 20 and os.environ.get("MIRROR_TRACK", "1") == "1" else 0.0

    def reset(self, env):
        self.serial, self.cur, self.label, self.onset, self.origin, self.fwd = -1, None, None, 0, None, 0.0
        self._rings = {}

    def _close(self):
        e = self.cur
        if e is not None and self.update and self.label is not None and len(e["q"]) >= 8:
            tv = np.array(e["tipv"]); tv[:4] = 0
            e["commit"] = int(np.argmax(tv))                                # fastest seen blade motion = the swing commits
            # how fast does the boss turn to follow me? per-frame change of (player bearing) it kept up with before commit
            yaw = np.unwrap([_yaw_of(q[3:7]) for q in e["q"][:e["commit"] + 1]])
            if len(yaw) > 3: self.dyaw = (getattr(self, "dyaw", []) + np.abs(np.diff(yaw)).tolist())[-3000:]
            L = self.eps.setdefault(self.label, []); L.append(e); del L[:-self.keep]
        self.cur = None

    def to_frame(self, q, origin, fwd):
        """boss qpos (63) -> onset frame"""
        out = q.copy(); d = q[:2] - origin; c, s = np.cos(-fwd), np.sin(-fwd)
        out[:2] = [c * d[0] - s * d[1], s * d[0] + c * d[1]]
        out[3:7] = _qmul(_qz(-fwd), q[3:7])
        return out

    @staticmethod
    def from_frame(qf, origin, fwd):
        out = qf.copy(); c, s = np.cos(fwd), np.sin(fwd)
        out[:2] = origin + np.array([c * qf[0] - s * qf[1], s * qf[0] + c * qf[1]])
        q = _qmul(_qz(fwd), qf[3:7]); out[3:7] = q / np.linalg.norm(q)
        return out

    def observe(self, env):
        b = env.boss; s = getattr(env, "boss_serial", -1)
        if SEGFREE:      # rolling window of perceived poses, independent of any move segmentation
            rq = env.d.qpos[:63].copy(); self.recent = (getattr(self, "recent", []) + [rq])[-4:]
        # rings: follow each to learn how far it gets (attributed to the move it spawned in)
        cur = {}
        for rg in env.rings:
            k = (round(float(rg["c"][0]), 2), round(float(rg["c"][1]), 2))
            if k not in self._rings:
                self._rings[k] = {"lab": None, "r": 0.0, "ep": self.cur}
                if self.cur is not None: self.cur["rings"].append(len(self.cur["q"]))
            self._rings[k]["r"] = max(self._rings[k]["r"], rg["r"] + self.rng.normal(0, 0.05)); cur[k] = 1
        for k in list(self._rings):
            if k not in cur:
                e = self._rings.pop(k)
                lab = e["ep"]["label"] if e["ep"] is not None else None
                if lab is not None and self.update:
                    self.ring_R.setdefault(lab, []).append(e["r"]); self.ring_R[lab] = self.ring_R[lab][-12:]
        if self.cur is not None and len(env.projectiles) > self.cur.get("_np", 0):
            self.cur["proj"].append(len(self.cur["q"]))
        if self.cur is not None: self.cur["_np"] = len(env.projectiles)
        if b.kind == "strike" and s != self.serial:
            if self.cur is not None: self.cur["cont"] = True     # cut by the next move of the string
            self._close()
            self.serial, self.onset, self.label = s, env.t, None
            to_p = env.p_xy - b.root
            self.origin = b.root + self.rng.normal(0, 0.1, 2)
            self.fwd = float(np.arctan2(to_p[1], to_p[0])) + self.rng.normal(0, 0.05)
            self.cur = {"q": [], "beta": [], "tipv": [], "d0": float(np.linalg.norm(to_p)), "rings": [], "proj": [], "label": None, "_np": len(env.projectiles)}
        if self.cur is None:
            return
        if b.kind != "strike" and len(self.cur["q"]) > 0:
            # keep recording the recovery for a while (it's part of the move's reach in time), then close
            if env.t - self.onset > 90:
                self._close(); return
        tau = env.t - self.onset
        if tau >= self.read_k and self.label is None:
            self.label = b.move_id if self.rng.random() < self.acc else int(self.rng.integers(self.n))
            self.cur["label"] = self.label
        q = env.d.qpos[:63].copy()
        q[:3] += self.rng.normal(0, self.np_, 3); q[7:] += self.rng.normal(0, self.nj, 56)
        self.cur["q"].append(self.to_frame(q, self.origin, self.fwd))
        to_p = env.p_xy - self.origin
        self.cur["beta"].append(float((np.arctan2(to_p[1], to_p[0]) - self.fwd + np.pi) % (2 * np.pi) - np.pi))
        self.cur.setdefault("pd", []).append(float(np.linalg.norm(env.p_xy - b.root)))
        tip = env.d.site_xpos[env.tip].copy() + self.rng.normal(0, 0.05, 3)
        lt = self.cur.get("_tip"); self.cur["tipv"].append(0.0 if lt is None else float(np.linalg.norm(tip - lt))); self.cur["_tip"] = tip

    def sample(self, label, d_now, rng):
        L = self.eps.get(label)
        if not L: return None
        w = np.array([np.exp(-0.5 * ((e["d0"] - d_now) / 1.0) ** 2) for e in L]) + 1e-3
        return L[int(rng.choice(len(L), p=w / w.sum()))]


class ReplayBoss:
    """Stand-in for the boss in an imagined arena: re-enacts a remembered episode from frame f0, then (B65b) chains
    into the next move of the string as the learned strings predict it, re-anchored to where the boss is and to
    where the imagined player is at that moment (bosses face you when a move starts)."""

    def __init__(self, real, ep, f0, origin, fwd, mem, label, g=None, rng=None, phase=False):
        self.mem, self.g, self.rng, self.phase = mem, g, rng, phase
        self.kind = "strike"; self.player_healing = False
        self.root = real.root.copy(); self.yaw_off = 0.0; self.hold = 0
        self._load(ep, int(f0), origin, fwd, label)

    def _load(self, ep, f0, origin, fwd, label):
        self.ep, self.f, self.origin, self.fwd, self.move_id, self.unit = ep, f0, origin, fwd, label, ep
        self.fwd0, self.delta = fwd, 0.0
        mem = self.mem
        R = np.median(mem.ring_R[label]) if mem.ring_R.get(label) else 0.0
        self.peak = ep["rings"][0] if (ep["rings"] and R > 0) else (ep["proj"][0] if ep["proj"] else 10 ** 6)
        # remembered fastest blade motion = the active frames; MIRROR_ACTWIN=0: imagination treats any contact as a hit
        self.act_peak = ep.get("commit", 10 ** 6) if os.environ.get("MIRROR_ACTWIN", "1") == "1" else None
        self.cur_move = types.SimpleNamespace(ring=float(R) if ep["rings"] else 0.0, projectile=bool(ep["proj"]))
        self.kind = "strike"

    def _next(self, player_xy):
        """the learned strings: does a move follow this one, and which?"""
        g, lab = self.g, self.move_id
        if g is None or self.rng is None or os.environ.get("MIRROR_CHAIN", "1") != "1": return False
        if os.environ.get("MIRROR_CHAIN_EP", "1") == "1":
            # the remembered episode itself says whether a move followed (it was cut by the next one) or the boss
            # went into recovery (the episode already contains it)
            if not self.ep.get("cont", False): return False
            n_cont, n_lab = 1.0, 1.0
        conts = {k: c for k, c in g.counts.items() if k[0] == lab}
        if os.environ.get("MIRROR_CHAIN_EP", "1") != "1":
            n_lab = sum(c[lab] for c in g.counts.values()) + 1.0
            n_cont = sum(c.sum() for c in conts.values())
            if self.rng.random() > n_cont / n_lab: return False
        if not conts: return False
        d = float(np.linalg.norm(player_xy - self.root))
        from .scripted import TOM, _uclass
        if TOM:   # H2: the imagined boss chooses its next move given what the imagined player is doing
            key = (lab, d > 4.0, bool(self.phase), _uclass(getattr(self, "pstate", (0, 0))[0]))
            c = conts.get(key)
            vs = [v for k, v in conts.items() if k[:3] == key[:3]]; pool = sum(vs) if vs else None
            if pool is not None: c = (c if c is not None else 0) + 2.0 * pool / max(pool.sum(), 1.0)
        else:
            c = conts.get((lab, d > 4.0, bool(self.phase)))
        if c is None or np.sum(c) == 0: c = sum(conts.values())
        p = c + 0.1; nxt = int(self.rng.choice(len(p), p=p / p.sum()))
        ep = self.mem.sample(nxt, d, self.rng)
        if ep is None: return False
        to_p = player_xy - self.root           # a new move starts facing the player (as at every remembered onset)
        self._load(ep, 0, self.root.copy(), float(np.arctan2(to_p[1], to_p[0])), nxt)
        self.blend = 8 if os.environ.get("MIRROR_BLEND", "1") == "1" else 0
        return True

    def step(self, player_xy, phase=False):
        Q = self.ep["q"]
        if self.kind == "strike" and 0 <= self.f < self.ep.get("commit", 0) and "beta" in self.ep:
            # B65c aim, counterfactual: the remembered boss turned to follow where *that* player went; turn this replay by
            # the difference between where the imagined player is and where the remembered one was (rate-limited)
            to_p = player_xy - self.origin
            want = (np.arctan2(to_p[1], to_p[0]) - self.fwd0 - self.ep["beta"][min(self.f, len(self.ep["beta"]) - 1)] + np.pi) % (2 * np.pi) - np.pi
            self.delta += float(np.clip(want - self.delta, -self.mem.turn, self.mem.turn))
            self.fwd = self.fwd0 + self.delta
        if self.f >= len(Q) and self.kind == "strike":
            if not self._next(player_xy):
                self.kind = "idle"
            Q = self.ep["q"]
        i = min(max(self.f, 0), len(Q) - 1)
        q = MirrorMemory.from_frame(Q[i], self.origin, self.fwd)
        lq = getattr(self, "last_q", None)
        if lq is not None and getattr(self, "blend", 0) > 0:     # bodies move continuously into a new movement
            w = self.blend / 9.0
            q[7:] = w * lq[7:] + (1 - w) * q[7:]
            q0 = lq[3:7] if np.dot(lq[3:7], q[3:7]) >= 0 else -lq[3:7]
            qq = w * q0 + (1 - w) * q[3:7]; q[3:7] = qq / np.linalg.norm(qq)
            q[2] = w * lq[2] + (1 - w) * q[2]; self.blend -= 1
        self.f += 1; self.root = q[:2].copy(); self.last_q = q
        return q


def place(Qf, q_now):
    """origin, facing that put remembered frame Qf (onset-frame coords) exactly on the perceived pose q_now:
    facing = the z-rotation taking Qf's root orientation to q_now's (exact when only yaw differs)."""
    a = q_now[3:7]; b = Qf[3:7] * np.array([1, -1, -1, -1])
    r = _qmul(a, b); fwd = 2 * np.arctan2(r[3], r[0])
    c, s = np.cos(fwd), np.sin(fwd)
    return q_now[:2] - np.array([c * Qf[0] - s * Qf[1], s * Qf[0] + c * Qf[1]]), float(fwd)


def anchor(ep, q_now, tau, rng=None, window=25):
    """B65d motion matching: find where in the remembered episode the boss is now (closest joint pose near the
    expected time), and the placement (origin, facing) that puts that remembered frame exactly on the boss's
    currently perceived root position and heading. Replay then continues from there."""
    Q = ep["q"]; lo, hi = max(0, tau - window), min(len(Q) - 1, tau + window)
    if hi < lo: lo = hi = len(Q) - 1
    J = np.array([Q[i][7:] for i in range(lo, hi + 1)])
    cost = np.linalg.norm(J - q_now[7:], axis=1) + 0.02 * np.abs(np.arange(lo, hi + 1) - tau)
    f = lo + int(np.argmin(cost))
    origin, fwd = place(Q[f], q_now)
    return f, origin, fwd


SEGFREE = os.environ.get("MIRROR_SEGFREE") == "1"    # B77b: recognition without onsets (search whole episodes)


def recognise(mem, tau, L=4, window=25, T=None, d_now=None):
    """B65e episodic recognition without labels: which remembered episodes (of any move) look like what the boss
    has done over the last L frames, and where in them is it now? Returns [(weight, episode, frame, label)]."""
    src = mem.recent if (SEGFREE and tau is None) else mem.cur["q"]
    cur = np.array([q[7:] for q in src[-L:]]); L = len(cur)
    T = T or float(os.environ.get("MIRROR_T", 0.5))
    out = []
    for lab, eps in mem.eps.items():
        for e in eps:
            Q = e.setdefault("_J", np.array([q[7:] for q in e["q"]]))
            if tau is None: lo, hi = L - 1, len(Q) - 1
            else: lo, hi = max(L - 1, tau - window), min(len(Q) - 1, tau + window)
            if hi < lo: continue
            # cost of aligning the last L observed frames to episode frames f-L+1..f
            idx = np.arange(lo, hi + 1)
            cost = sum(np.linalg.norm(Q[idx - (L - 1 - j)] - cur[j], axis=1) for j in range(L)) / L
            if tau is not None: cost = cost + 0.02 * np.abs(idx - tau)
            if d_now is not None and "pd" in e:     # lunges depend on how far away the player is
                pd = np.array(e["pd"]); cost = cost + float(os.environ.get("MIRROR_DW", 0.3)) * np.abs(pd[np.minimum(idx, len(pd) - 1)] - d_now)
            k = int(np.argmin(cost)); out.append((float(cost[k]), e, int(idx[k]), lab))
    if not out: return []
    c0 = min(o[0] for o in out); mem._c0 = c0
    w = np.array([np.exp(-(o[0] - c0) / T) for o in out]); w /= w.sum()
    return [(w[i],) + out[i][1:] for i in range(len(out))]


class StillBoss:
    """Imagined boss when memory offers nothing to re-enact: it holds its current pose (never the real boss runner)."""

    def __init__(self, real, q):
        self.kind, self.unit, self.player_healing, self.move_id = "idle", None, False, -1
        self.root, self.q, self.cur_move, self.f, self.peak, self.hold, self.yaw_off = real.root.copy(), q.copy(), None, 0, 10 ** 6, 0, 0.0

    def step(self, player_xy, phase=False):
        return self.q


class MirrorPlanner:
    """MPC over the agent's macro-actions; futures imagined by re-enacting remembered boss episodes."""

    def __init__(self, mem, grammar, seed=0, H=24, n=3):
        self.mem, self.g, self.H, self.n = mem, grammar, H, n
        self.view = None          # B77: a percept.FairView; None = ground-truth boss state (B65-B76)
        self.rng = np.random.default_rng(seed)

    def reset(self, env):
        self.mem.reset(env); self.g.reset(env); self.hold, self.hold_n = None, 0
        self._bhp, self._dmg_t = env.boss_hp, env.t

    def _imagine(self, env, a, seed, a2=None, H=None):
        rng = np.random.default_rng(seed); mem = self.mem
        V = self.view or env; b = V.boss; lag = V.L if self.view is not None else 0
        sim = env.clone(seed)
        gs = os.environ.get("MIRROR_GEOM_SCALE")
        if gs:     # B79: the agent's belief about the boss's limb/blade sizes is off by this factor (imagination only)
            if getattr(self, "_m2", None) is None:
                import copy as _c
                m2 = _c.deepcopy(env.m); f = float(gs)
                for i in range(m2.ngeom):
                    if m2.geom(i).name.startswith("b_"):
                        m2.geom_size[i] *= f; m2.geom_pos[i] *= f; m2.geom_rbound[i] *= f
                self._m2 = m2
            sim.m = self._m2
        lm = os.environ.get("MIRROR_LAT_MODEL")
        if lm:     # B77c: imagination knows its own *typical* input lag and what it has pressed, not the jitter
            if lm == "oracle":   # diagnostic only: imagination knows the exact queue and future lag drift
                import copy as _c
                sim.act_lat = env.act_lat; sim.lat_d = env.lat_d; sim.lat_rng = _c.deepcopy(env.lat_rng); sim.aq = list(env.aq)
                dl = None
            elif lm == "emp":   # B77d: each future gets its own lag, drawn from press-to-motion delays measured online
                L = getattr(self, "lags", [])
                dl = int(rng.choice(L)) if len(L) >= 5 else 6
            else:
                dl = int(lm)
            if dl is not None:
                sim.act_lat = (dl, dl); sim.lat_d = dl; sim.lat_rng = np.random.default_rng(seed)
                sim.aq = [(x[2] + dl, x[1], x[2]) for x in getattr(env, "aq", [])]
        else:
            sim.act_lat = None; sim.aq = []          # imagination doesn't know about input latency (B77)
        if self.view is not None:
            # imagination starts from the perceived boss pose
            sim.d.qpos[:63] = V.d.qpos[:63]; sim.prev_qpos[:63] = V.d.qpos[:63]
        if REAL2 and hasattr(self, "st_est"):   # the boss's stance is hidden: imagination uses the agent's belief
            sim.stance, sim.stance_max, sim.stance_t = self.st_est, self.st_max, self.st_t
        tau = V.t - mem.onset
        rec = getattr(self, "_rec", None)
        if rec and (SEGFREE or (b.kind == "strike" and mem.cur is not None and tau < 90)):
            i = int(rng.choice(len(rec), p=np.array([r[0] for r in rec])))
            _, ep, f, lab = rec[i]
            q_now = V.d.qpos[:63].copy(); q_now[7:] += rng.normal(0, mem.nj, 56); q_now[:2] += rng.normal(0, mem.np_, 2)
            o, fwd = place(ep["q"][f], q_now)
            sim.boss = ReplayBoss(b, ep, f + 1 + lag, o, fwd, mem, lab, self.g, rng, env.phase)
        elif b.kind == "strike" and mem.cur is not None and tau < 90:
            lab = mem.label if mem.label is not None else self.g.sample(V, rng)
            ep = mem.sample(lab, mem.cur["d0"], rng)
            if ep is not None:
                if os.environ.get("MIRROR_ANCHOR", "1") == "1":
                    q_now = V.d.qpos[:63].copy(); q_now[7:] += rng.normal(0, mem.nj, 56); q_now[:2] += rng.normal(0, mem.np_, 2)
                    f, o, fw = anchor(ep, q_now, tau)
                    sim.boss = ReplayBoss(b, ep, f + 1 + lag, o, fw, mem, lab, self.g, rng, env.phase)
                else:
                    sim.boss = ReplayBoss(b, ep, tau + 1, mem.origin, mem.fwd, mem, lab, self.g, rng, env.phase)
        elif os.environ.get("MIRROR_OPENER", "1") == "1":
            # boss between moves: it may open within 0-15 frames; frame = now
            lab = self.g.sample(V, rng, next_move=True)
            to_p = env.p_xy - b.root; d = float(np.linalg.norm(to_p))
            ep = mem.sample(lab, d, rng)
            if ep is not None:
                start = -int(rng.integers(0, 16)) + lag
                rb = ReplayBoss(b, ep, 0, b.root.copy(), float(np.arctan2(to_p[1], to_p[0])), mem, lab, self.g, rng, env.phase)
                rb.last_q = V.d.qpos[:63].copy(); rb.blend = 8 if os.environ.get("MIRROR_BLEND", "1") == "1" else 0
                if start > 0: rb.f = start
                if start < 0:     # hold the current pose for the delay, then re-enact
                    rb.f = start; q_now = V.d.qpos[:63].copy(); step0 = rb.step
                    def step(px, ph=False, rb=rb, q_now=q_now, step0=step0):
                        if rb.f < 0:
                            rb.f += 1; return q_now
                        return step0(px, ph)
                    rb.step = step
                sim.boss = rb
        if sim.boss is b or not isinstance(sim.boss, (ReplayBoss, StillBoss)):
            # no memory to re-enact: never fall back to the real boss runner (that would be a true simulator)
            sim.boss = StillBoss(b, V.d.qpos[:63]); self.n_still = getattr(self, "n_still", 0) + 1
        self.n_imag = getattr(self, "n_imag", 0) + 1
        hp0, b0 = sim.hp, sim.boss_hp
        mv = a % N_BUTTONS in (NONE, GUARD); k2 = None
        for k in range(H or self.H):
            if a2 is not None and k2 is None and k >= (6 if mv else 1) and sim.player_free() and \
                    not any(x[1] % N_BUTTONS != NONE for x in sim.aq):     # under lag: only once the first press has landed
                k2 = k
            if k2 is not None:
                mv2 = a2 % N_BUTTONS in (NONE, GUARD)
                u = a2 if (k == k2 or (mv2 and k < k2 + 6)) else 0
            else:
                u = a if (k == 0 or (mv and k < 6)) else 0
            _, _, done, info = sim.step(u)
            if done: break
        v = (b0 - sim.boss_hp) - 2.5 * (hp0 - sim.hp) - (100 if sim.hp <= 0 else 0)
        reach = 0.8 + 0.25 * env.spec.scale
        v -= self._w * max(0.0, float(np.linalg.norm(b.root - sim.p_xy)) - reach)
        return v

    def _stance_belief(self, env):
        """H3: belief over the boss's hidden stance, from my own hits (seen on the HP bar), the published rules
        (stance damage per attack type, regen 13/s after max/13 s) and a prior max of 100, re-estimated at each
        observed break (the boss visibly goes down)."""
        from .env import STANCE_DMG
        if not hasattr(self, "st_max"): self.st_max = float(os.environ.get("MIRROR_STANCE_PRIOR", 100.0)); self.st_dealt = 0.0
        if not hasattr(self, "st_est") or env.t == 0: self.st_est, self.st_t, self._bhp_s, self._down = self.st_max, -999, env.boss_hp, False
        if env.t - self.st_t > self.st_max / 13.0 * 30: self.st_est = min(self.st_max, self.st_est + 13.0 / 30 * max(1, env.t - getattr(self, "_st_last", env.t)))
        self._st_last = env.t
        down = env.boss_stun > 0
        if env.boss_hp < self._bhp_s and not down and not self._down:
            k = "light" if env.act == LIGHT else ("gc" if env.gc else "heavy")
            self.st_est -= STANCE_DMG[k]; self.st_dealt += STANCE_DMG[k]; self.st_t = env.t
        if down and not self._down:     # observed break: learn the boss's max stance from what I had dealt since full
            self.st_max = 0.5 * self.st_max + 0.5 * max(40.0, self.st_max - self.st_est + 0.0); self.st_est = self.st_max
        self._down = down; self._bhp_s = env.boss_hp

    def act(self, env):
        V = self.view or env
        self.g.observe(V)
        if REAL2: self._stance_belief(env)
        pend = getattr(self, "_pend", None)    # own press -> motion delay (proprioception)
        if pend is not None and env.act == pend[1] and env.act_t == 1 and env.t > pend[0]:
            self.lags = (getattr(self, "lags", []) + [env.t - pend[0] - 1])[-200:]; self._pend = None
        elif pend is not None and env.t - pend[0] > 20:
            self._pend = None
        if not env.player_free():
            return 0
        if os.environ.get("MIRROR_WAIT_PRESS", "1") == "1" and any(x[1] % N_BUTTONS in (ROLL, LIGHT, HEAL, HEAVY, BACKSTEP) for x in getattr(env, "aq", [])):
            return 0      # B77f: a press is still in flight (input lag): don't re-decide until it lands
        if self.hold_n > 0:
            self.hold_n -= 1; return self.hold
        if env.boss_hp != self._bhp: self._bhp, self._dmg_t = env.boss_hp, env.t
        self._w = float(os.environ.get("MIRROR_CLOSE", 2.0)) * (1 + float(os.environ.get("MIRROR_HUNGER", 1.0)) * (env.t - self._dmg_t) / 300.0)
        acts = [NONE, 1 * N_BUTTONS + NONE, 5 * N_BUTTONS + NONE, 3 * N_BUTTONS + NONE, 7 * N_BUTTONS + NONE,
                LIGHT, 3 * N_BUTTONS + ROLL, 7 * N_BUTTONS + ROLL, 5 * N_BUTTONS + ROLL]
        if os.environ.get("WM_ROLL_FWD") == "1":
            acts += [1 * N_BUTTONS + ROLL, 2 * N_BUTTONS + ROLL, 8 * N_BUTTONS + ROLL]
        if REAL2:
            acts += [HEAVY, BACKSTEP]
        if os.environ.get("MIRROR_GUARD") == "1" or REAL2:
            acts += [GUARD, 5 * N_BUTTONS + GUARD, 3 * N_BUTTONS + GUARD, 7 * N_BUTTONS + GUARD]
        if env.flasks and env.hp < 60: acts.append(HEAL)
        tau = V.t - self.mem.onset
        if SEGFREE:
            self._rec = recognise(self.mem, None, d_now=float(np.linalg.norm(V.p_xy - V.boss.root))) if (self.mem.eps and len(getattr(self.mem, "recent", [])) >= 2) else None
        else:
          self._rec = recognise(self.mem, tau, d_now=float(np.linalg.norm(V.p_xy - V.boss.root))) if (os.environ.get("MIRROR_RECOG", "1") == "1" and V.boss.kind == "strike"
                                                  and self.mem.cur is not None and len(self.mem.cur["q"]) >= 2 and tau < 90) else None
        seeds = [int(s) for s in self.rng.integers(1 << 30, size=self.n)]
        risk = float(os.environ.get("MIRROR_RISK", 0.0))     # 0 = mean over futures, 1 = worst future
        agg = lambda xs: (1 - risk) * np.mean(xs) + risk * np.min(xs)
        H2 = int(os.environ.get("MIRROR_H2", 0))
        if H2:   # two-step plans: first action, then a follow-up as soon as the player is free again
            vals = [agg([self._imagine(env, a, s, H=H2) for s in seeds]) for a in acts]
            fol = [LIGHT, 5 * N_BUTTONS + ROLL, 3 * N_BUTTONS + ROLL, 7 * N_BUTTONS + ROLL, 1 * N_BUTTONS + ROLL]
            if REAL2: fol.append(HEAVY)
            for i in np.argsort(vals)[::-1][:int(os.environ.get("MIRROR_TOPK", 2))]:
                for a2 in fol:
                    vals[i] = max(vals[i], agg([self._imagine(env, acts[i], s, a2=a2, H=H2) for s in seeds]))
        else:
            vals = [agg([self._imagine(env, a, s) for s in seeds]) for a in acts]
        a = acts[int(np.argmax(vals))]
        if a % N_BUTTONS in (ROLL, LIGHT, HEAVY, BACKSTEP) and getattr(self, "_pend", None) is None: self._pend = (env.t, a % N_BUTTONS)
        if a % N_BUTTONS in (NONE, GUARD) and a not in (NONE, GUARD): self.hold, self.hold_n = a, int(os.environ.get("MIRROR_HOLD", 5))
        return a
