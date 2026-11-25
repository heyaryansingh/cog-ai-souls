"""Score the pixel perception pipeline against ground truth in Arena3D.

Renders fights (lock-on camera, 256x192), runs the human-like Observer (motion channel + MediaPipe pose
on a foveated crop) and compares its 2D landmarks with the true projected boss joints.
Metrics: detection rate, boss-association rate, PCK@0.1 (of boss image height), mean error / height.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco
from arena3d import library as L, mocap
from arena3d.boss import sample_boss
from arena3d.env import Arena3D
from arena3d import scripted as S
from arena3d.perception import Observer, PoseEstimator

# MediaPipe landmark -> CMU body (joint origin)
PAIRS = [(0, "head"), (11, "lhumerus"), (12, "rhumerus"), (13, "lradius"), (14, "rradius"), (15, "lwrist"),
         (16, "rwrist"), (23, "lfemur"), (24, "rfemur"), (25, "ltibia"), (26, "rtibia"), (27, "lfoot"), (28, "rfoot")]


def project(env, pts, W, H, fovy=45.0):
    cam = env.camera()
    az, el = np.radians(cam.azimuth), np.radians(cam.elevation)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    pos = np.array(cam.lookat) - cam.distance * fwd
    right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rel = pts - pos
    x, y, z = rel @ right, rel @ up, rel @ fwd
    f = 0.5 * H / np.tan(np.radians(fovy) / 2)
    return np.stack([(W / 2 + f * x / z) / W, (H / 2 - f * y / z) / H], 1)


def main(nb=4, frames=900, W=256, H=192):
    U = L.load("checkpoints/arena3d_units.pkl")
    _, _, SUBS = L.split_subjects(U)
    TEST = [i for i, u in enumerate(U) if u.subject in SUBS]
    C = mocap.load()
    res = []
    for b in range(nb):
        spec = sample_boss(np.random.default_rng(5000 + b), U, TEST)
        env = Arena3D(spec, U, C, seed=b, render=False)
        env.reset()
        R = mujoco.Renderer(env.m, H, W)
        obs_ = Observer(PoseEstimator(video=os.environ.get("PE_VIDEO", "0") == "1"), crop=os.environ.get("CROP", "1") == "1", mask_self=os.environ.get("MASK", "1") == "1", grow=float(os.environ.get("GROW", "1.6")))
        ids = [env.m.body("b_" + n).id for _, n in PAIRS]
        rng = np.random.default_rng(b)
        st = {"n": 0, "det": 0, "assoc": 0, "pck": [], "err": [], "onset_hits": 0}
        t0 = time.time()
        for t in range(frames):
            a = S.oracle_dodger(env, rng) if False else (5 * 5 if t % 90 < 30 else 1 * 5)  # wander in/out
            _, _, done, info = env.step(a)
            if done:
                env.reset(); obs_.reset()
            if t % 2:
                continue
            px = env.render(R)
            lock = project(env, env.d.xpos[[env.m.body("b_thorax").id]], W, H)[0]       # lock-on reticle
            pp = project(env, env.d.xpos[[i for i in range(env.m.nbody) if env.m.body(i).name.startswith("p_")]], W, H)
            pad = 0.02
            sbox = (pp[:, 0].min() - pad, pp[:, 1].min() - pad, pp[:, 0].max() + pad, pp[:, 1].max() + pad)
            o = obs_(px, lock, sbox)
            gt = project(env, env.d.xpos[ids], W, H)
            hgt = gt[:, 1].max() - gt[:, 1].min()
            st["n"] += 1
            if o["ok"]:
                st["det"] += 1
                kp = o["kp2"][[p for p, _ in PAIRS], :2]
                e = np.linalg.norm(kp - gt, axis=1) / max(hgt, 1e-3)
                # a faceless mannequin has no left/right cue: score the better of the two labelings
                sw = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]
                e2 = np.linalg.norm(kp[sw] - gt, axis=1) / max(hgt, 1e-3)
                if e2.mean() < e.mean():
                    e = e2; st["swap"] = st.get("swap", 0) + 1
                # association: is the detection on the boss (median error < 0.5 boss heights)?
                if np.median(e) < 0.5:
                    st["assoc"] += 1
                    st["pck"].append(float((e < 0.1).mean()))
                    st["err"].append(float(np.median(e)))
        r = {"boss": b, "scale": spec.scale, "detect": st["det"] / st["n"], "assoc": st["assoc"] / st["n"],
             "pck01": float(np.mean(st["pck"])) if st["pck"] else 0.0,
             "median_err_over_height": float(np.mean(st["err"])) if st["err"] else None,
             "sec": time.time() - t0, "frames": st["n"]}
        print(r, flush=True)
        res.append(r)
    json.dump(res, open("results/arena3d_perception.json", "w"), indent=1)
    print("mean detect", np.mean([r["detect"] for r in res]), "assoc", np.mean([r["assoc"] for r in res]),
          "PCK@0.1", np.mean([r["pck01"] for r in res]))


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
