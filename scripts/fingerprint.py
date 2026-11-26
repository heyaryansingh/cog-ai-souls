"""B53 probe: a perceptual 'fingerprint' of a boss from the scouting phase (no program access): how often it swings,
how long a swing lasts, how often swings chain into combos, how far it reaches, how fast it closes distance.
Question: is Margit out-of-family relative to procedural bosses 0-17 while held-out procedural bosses 18-35 are not?"""
import os, sys, json
sys.path.insert(0, "/home/user/cog-ai-souls"); os.chdir("/home/user/cog-ai-souls")
__file__ = "/home/user/cog-ai-souls/scripts/general_learner.py"
exec(open(__file__).read().rsplit("if __name__", 1)[0])


def fingerprint(sp, frames=1500, seed=0):
    env = Arena3D(sp, U, clips(), seed=seed, player_strike=pstrike()); env.reset(); rng = np.random.default_rng(seed)
    strike = 0; serials = {}; starts = []; ends = []; prev_kind = None; speeds = []; dists = []; last_root = env.boss.root.copy()
    for t in range(frames):
        dist = np.linalg.norm(env.boss.root - env.p_xy); want = [1.5, 3.0, 5.0, 8.0][(t // 300) % 4]
        if env.player_free() and rng.random() < 0.03:
            a = int(rng.choice([3, 5, 7])) * N_BUTTONS + ROLL
        else:
            a = (1 if dist > want + 0.4 else 5 if dist < want - 0.4 else int(rng.choice([3, 7]))) * N_BUTTONS + NONE
        env.step(a); env.hp = 1e9
        if env.boss_hp < 0.5 * sp.hp: env.boss_hp = sp.hp
        k = env.boss.kind; s = getattr(env, "boss_serial", -1)
        if k == "strike":
            strike += 1; serials[s] = serials.get(s, 0) + 1
            if prev_kind != "strike": starts.append(t)
        elif prev_kind == "strike": ends.append(t)
        speeds.append(np.linalg.norm(env.boss.root - last_root)); last_root = env.boss.root.copy(); dists.append(dist)
        prev_kind = k
    gaps = [s - e for s, e in zip(starts[1:], ends) if s > e]
    dur = np.array(list(serials.values()) or [0])
    return dict(strike_frac=strike / frames, swing_len=float(np.mean(dur)), swings_per_100=100 * len(serials) / frames,
                combo_frac=float(np.mean([g < 20 for g in gaps])) if gaps else 0.0, swings_per_bout=len(serials) / max(1, len(starts)),
                move_speed=float(np.mean(speeds)) * 60, scale=float(getattr(sp, "scale", 1.0)))


if __name__ == "__main__":
    from multiprocessing import Pool
    items = [("proc", b) for b in range(36)] + [("margit", 0)]
    def f(it):
        sp = spec() if it[0] == "margit" else pboss(it[1]); return it, fingerprint(sp, seed=11)
    with Pool(4) as p: res = p.map(f, items)
    json.dump([[a, b, d] for (a, b), d in res], open("results/fingerprints.json", "w"), indent=0)
    keys = list(res[0][1]); F = {it: np.array([d[k] for k in keys]) for it, d in res}
    tr = np.array([F[("proc", b)] for b in range(18)]); mu, sd = tr.mean(0), tr.std(0) + 1e-6
    print("feature", keys); print("family mean", mu.round(3), "sd", sd.round(3))
    for it in items:
        z = (F[it] - mu) / sd; print(it, F[it].round(3), "z-dist", round(float(np.sqrt((z ** 2).mean())), 2))
