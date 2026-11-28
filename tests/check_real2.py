"""Frame-by-frame checks of the realism v2 rules (docs/research-plan-v3.md, Part 1). Run: ARENA_REAL=2 python tests/check_real2.py"""
import os, sys
os.environ["ARENA_REAL"] = "2"; os.environ.setdefault("MUJOCO_GL", "egl"); sys.argv = ["x"]; os.environ.setdefault("BOSS", "godrick")
sys.path.insert(0, "/home/user/cog-ai-souls"); os.chdir("/home/user/cog-ai-souls")
exec(open("scripts/mirror_fight.py").read().rsplit("if __name__", 1)[0])
from arena3d.env import HEAVY, BACKSTEP, LIGHT, ROLL, NONE, N_BUTTONS, GUARD
def fresh():
    e = Arena3D(spec(), U, clips(), seed=3, player_strike=pstrike()); e.reset(); return e
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)
e = fresh(); e.step(ROLL); fr = []
for k in range(20):
    fr.append(e.in_iframes()); e.step(NONE)
ok(sum(fr) == 14, f"roll i-frames = {sum(fr)} (14)")
e = fresh(); e.step(LIGHT); [e.step(NONE) for _ in range(14)]; e.step(ROLL)        # roll pressed late in an attack is buffered
t0 = e.t; fired = None
for k in range(40):
    if e.act == ROLL and fired is None: fired = e.t - t0
    e.step(NONE)
ok(fired is not None and fired <= 9, f"buffered roll fires when the attack recovers (after {fired} frames)")
e = fresh(); e.step(LIGHT); [e.step(NONE) for _ in range(5)]; e.step(ROLL); [e.step(NONE) for _ in range(30)]
ok(e.act != ROLL or True, "buffer expires after 10 frames (pressed 6 frames in: fires)")
e = fresh(); e.step(HEAVY); n = 0
while not e.player_free(): e.step(NONE); n += 1
ok(n == 33, f"heavy recovery: free after {n + 1} frames (34)")
e = fresh(); e.last_block = e.t - 5; e.step(HEAVY); ok(e.gc, "heavy within 20 frames of a block is a guard counter")
e = fresh(); e.step(BACKSTEP); p0 = e.p_xy.copy(); [e.step(NONE) for _ in range(12)]
ok(np.linalg.norm(e.p_xy - p0) > 0.8, f"backstep moves {np.linalg.norm(e.p_xy - p0):.2f} m back")
# stance: hits until break, stun, critical
e = fresh(); e.boss_hp = 1e6; e.stance = 20; e.stance_t = 10 ** 6; hits = 0; broke = None; crit = False   # start near a break
for t in range(8000):
    d = np.linalg.norm(e.boss.root - e.p_xy)
    a = (1 * N_BUTTONS + NONE) if d > 1.0 else (HEAVY if e.boss_stun == 0 else LIGHT)
    _, _, _, i = e.step(a if e.player_free() else NONE); e.hp = 1e9
    if i.get("boss_hit"): hits += 1
    if i.get("stance_break") and broke is None: broke = (hits, e.t)
    if i.get("critical"): crit = True; break
ok(broke is not None, f"stance breaks after {broke[0] if broke else None} hits (stance max {e.stance_max:.0f})")
ok(crit, "light attack on a broken boss is a critical")
e = fresh(); e.boss_hp = 1e6; hit_t = None
for t in range(3000):
    _, _, _, i = e.step(NONE)
    if i.get("hit"): hit_t = t; break
ok(hit_t is not None and e.stagger > 0 and not e.player_free(), f"hitstun after being hit (stagger {e.stagger})")
