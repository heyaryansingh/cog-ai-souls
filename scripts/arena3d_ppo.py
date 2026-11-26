import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn
from collections import deque
sys.argv = sys.argv[:2] + ["none"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "arena3d_agent.py")).read().split("if __name__")[0])
from arena3d import rl as R
from arena3d.boss import sample_boss

SPECS = {}


MARGIT = None


def make_train_env(rng):
    global MARGIT
    if os.environ.get("PPO_BOSS") == "margit":      # single grounded boss; PPO_PHASE=2 starts fights in phase 2
        if MARGIT is None:
            from arena3d.program import build_program_boss
            MARGIT = build_program_boss(np.random.default_rng(42), U, TEST, clips())
        env = Arena3D(MARGIT, U, clips(), seed=int(rng.integers(1 << 30)), player_strike=pstrike())
        if os.environ.get("PPO_PHASE") == "2":
            r0 = env.reset

            def reset():
                o = r0(); env.boss_hp = 0.49 * env.spec.hp; env.phase = True; env.h0 = env.boss_hp
                return o
            env.reset = reset
        return env
    # a pool of 200 verified training bosses (sampling+verification is the slow part), drawn at random
    i = int(rng.integers(200))
    if i not in SPECS:
        SPECS[i] = train_boss(i)
    return Arena3D(SPECS[i], U, clips(), seed=int(rng.integers(1 << 30)), player_strike=pstrike())


def train(n_envs=16, T=128, updates=3000, lr=3e-4, gamma=0.995, lam=0.95, clip=0.2, epochs=4, ent=0.01, seed=0):
    torch.manual_seed(seed); torch.set_num_threads(2)
    pol = R.Policy(); opt = torch.optim.Adam(pol.parameters(), lr=lr)
    if os.environ.get("PPO_INIT"):
        pol.load_state_dict(torch.load(os.environ["PPO_INIT"]))
    ck = os.environ.get("PPO_CKPT", "checkpoints/arena3d_ppo.pt"); dmg = deque(maxlen=100)
    W = [R.Worker(make_train_env, seed * 1000 + i) for i in range(n_envs)]
    obs = np.stack([w.obs() for w in W]); h = torch.zeros(1, n_envs, pol.hidden)
    wins, frames, t0 = deque(maxlen=300), 0, time.time()
    for u in range(updates):
        h0 = h.clone(); O, A, LP, V, Rw, D = [], [], [], [], [], []
        with torch.no_grad():
            for t in range(T):
                lg, v, h = pol(torch.from_numpy(obs)[:, None], h)
                dist = torch.distributions.Categorical(logits=lg[:, 0]); a = dist.sample()
                O.append(obs); A.append(a.numpy()); LP.append(dist.log_prob(a).numpy()); V.append(v[:, 0].numpy())
                rs, ds, no = [], [], []
                for i, w in enumerate(W):
                    r, done, out, n = w.step(int(a[i])); frames += n
                    if done:
                        wins.append(out == "win"); h0_ = getattr(w.env, "h0", w.env.spec.hp)
                        dmg.append(1 - w.env.boss_hp / h0_); o = w.new(); h[:, i] = 0
                    else:
                        o = w.obs()
                    rs.append(r); ds.append(done); no.append(o)
                Rw.append(rs); D.append(ds); obs = np.stack(no)
            _, vl, _ = pol(torch.from_numpy(obs)[:, None], h); vl = vl[:, 0].numpy()
        Rw, D, V = np.array(Rw, np.float32), np.array(D, np.float32), np.array(V)
        adv = np.zeros_like(Rw); last = np.zeros(n_envs, np.float32)
        for t in reversed(range(T)):
            nv = vl if t == T - 1 else V[t + 1]
            delta = Rw[t] + gamma * nv * (1 - D[t]) - V[t]
            last = delta + gamma * lam * (1 - D[t]) * last; adv[t] = last
        ret = adv + V
        Ot = torch.from_numpy(np.stack(O, 1)); At = torch.from_numpy(np.stack(A, 1)); LPt = torch.from_numpy(np.stack(LP, 1))
        advt = torch.from_numpy(adv.T.copy()); advt = (advt - advt.mean()) / (advt.std() + 1e-8)
        rett = torch.from_numpy(ret.T.copy()); dt = torch.from_numpy(D.T.copy())
        for _ in range(epochs):
            hh, lgs, vs = h0, [], []
            for t in range(T):
                lg, vv, hh = pol(Ot[:, t:t + 1], hh); lgs.append(lg[:, 0]); vs.append(vv[:, 0])
                hh = hh * (1 - dt[:, t])[None, :, None]
            lgs, vs = torch.stack(lgs, 1), torch.stack(vs, 1)
            dd = torch.distributions.Categorical(logits=lgs); ratio = torch.exp(dd.log_prob(At) - LPt)
            lpi = -torch.min(ratio * advt, torch.clamp(ratio, 1 - clip, 1 + clip) * advt).mean()
            lv = 0.5 * ((vs - rett) ** 2).mean()
            loss = lpi + 0.5 * lv - ent * dd.entropy().mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(pol.parameters(), 0.5); opt.step()
        if u % 10 == 0:
            print(f"update {u} frames {frames/1e6:.2f}M fights {len(wins)} recent win {np.mean(wins) if wins else 0:.3f} "
                  f"dmg {np.mean(dmg) if dmg else 0:.3f} lv {lv.item():.3f} ent {dd.entropy().mean().item():.2f} {frames/(time.time()-t0):.0f} fps", flush=True)
            torch.save(pol.state_dict(), ck)
    return pol


if __name__ == "__main__":
    train(updates=int(os.environ.get("UPDATES", 3000)))
