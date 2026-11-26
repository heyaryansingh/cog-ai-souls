"""R7 data: perceived boss pose windows at the start of each strike -> the weapon tip's future path, across many
procedural bosses. Train bosses b=100..139, test bosses b=0..17 (the general-suite bosses, never trained on).
Path: tip xy for tau = 0..87 step 3, in the onset frame (origin = boss root at onset, x axis = toward player at onset),
in units of boss height. NaN once the strike ends."""
import os, sys, pickle
__file__ = "scripts/general_learner.py"
exec(open(__file__).read().rsplit("if __name__", 1)[0])
from arena3d.agent import Sensor, centric
from multiprocessing import Pool
W, TAUS = 20, np.arange(0, 90, 3)


def collect(b, frames=None):
    frames = frames or int(os.environ.get("FRAMES", 3000))
    sp = pboss(b); env = Arena3D(sp, U, clips(), seed=900 + b, player_strike=pstrike()); env.reset()
    sen = Sensor(env, seed=b); rng = np.random.default_rng(b); env.hp = 1e9; env.flasks = 0
    hist, rows, cur, last = [], [], None, -1
    h = 1.75 * sp.scale
    for t in range(frames):
        P, _ = sen(); hist.append(None if P is None else centric(P)[0])
        _, _, done, info = env.step(S.aggro(env, rng))
        bo = env.boss; s = getattr(env, "boss_serial", -1)
        if s != last and bo.kind == "strike":
            if cur: rows.append(cur)
            to_p = env.p_xy - bo.root
            cur = {"b": b, "move": int(bo.move_id), "onset": t, "o": bo.root.copy(), "f": float(np.arctan2(to_p[1], to_p[0])),
                   "path": np.full((len(TAUS), 2), np.nan), "scale": sp.scale, "prel": None}
            c, si = np.cos(-cur["f"]), np.sin(-cur["f"]); cur["R"] = np.array([[c, -si], [si, c]])
            cur["prel"] = cur["R"] @ to_p / h
            last = s
        if cur is not None and s == last and bo.kind == "strike":
            tau = t - cur["onset"]
            if tau < 90 and tau % 3 == 0:
                cur["path"][tau // 3] = cur["R"] @ (env.d.site_xpos[env.tip][:2] - cur["o"]) / h
        if env.boss_hp < 0.1 * sp.hp: env.boss_hp = 0.5 * sp.hp
        if done: break
    for r in rows:
        r["obs"] = [hist[i] if i < len(hist) else None for i in range(r["onset"], r["onset"] + W)]
        del r["R"]
    return rows


if __name__ == "__main__":
    tr = [int(x) for x in os.environ.get("TRAIN", ",".join(map(str, range(100, 140)))).split(",")]
    te = list(range(18))
    with Pool(int(os.environ.get("PROCS", 4))) as p:
        rows = sum(p.map(collect, tr + te), [])
    pickle.dump(rows, open(os.environ.get("R7_OUT", "data/r7_swings.pkl"), "wb"))
    print(len(rows), "swings;", sum(r["b"] >= 100 for r in rows), "train")
