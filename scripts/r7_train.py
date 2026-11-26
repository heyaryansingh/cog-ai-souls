"""R7 amortised intuition: one network, trained on swings of 40 procedural bosses, predicts the weapon tip's
future path from the first k frames of perceived boss pose. Tested zero-shot on 18 never-seen bosses, against
(a) the average path, (b) the current learner's knowledge: per-move mean path of the SAME boss from its other swings
(needs the move identity and experience of that boss; leave-one-out), (c) network + that boss's experience."""
import os, pickle, numpy as np, torch, torch.nn as nn
rows = pickle.load(open(os.environ.get("R7_DATA", "data/r7_swings.pkl"), "rb"))
torch.manual_seed(0); rng = np.random.default_rng(0)


def feat(r, k):
    o = [(i, x) for i, x in enumerate(r["obs"][:k]) if x is not None]
    if not o:
        return np.zeros(39 * 3 + 5)
    i1, a1 = o[-1]; i0, a0 = o[0]
    v = (a1 - a0) / max(1, i1 - i0)
    mid = o[len(o) // 2][1]
    return np.concatenate([a1, v * 10, mid - a0, [len(o) / k, i1 / k, r["prel"][0], r["prel"][1], r["scale"]]])


def tensors(R, k):
    X = np.array([feat(r, k) for r in R], np.float32)
    Y = np.array([r["path"] for r in R], np.float32).reshape(len(R), -1)
    M = ~np.isnan(Y); Y = np.nan_to_num(Y)
    return torch.tensor(X), torch.tensor(Y), torch.tensor(M, dtype=torch.float32)


def err(P, Y, M):     # mean tip distance over valid future points (boss heights)
    d = ((P - Y) ** 2).reshape(len(Y), -1, 2).sum(-1).sqrt(); m = M.reshape(len(Y), -1, 2)[..., 0]
    return float((d * m).sum() / m.sum())


tr = [r for r in rows if r["b"] >= 100]; te = [r for r in rows if r["b"] < 100]
print(len(tr), "train swings,", len(te), "test swings on", len({r['b'] for r in te}), "unseen bosses")
for k in [int(x) for x in os.environ.get("KS", "4,8,12,16,20").split(",")]:
    Xtr, Ytr, Mtr = tensors(tr, k); Xte, Yte, Mte = tensors(te, k)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, Ytr.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    for ep in range(400):
        net.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 256):
            j = perm[i:i + 256]; P = net((Xtr[j] - mu) / sd)
            loss = (((P - Ytr[j]) ** 2) * Mtr[j]).sum() / Mtr[j].sum()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        P = net((Xte - mu) / sd)
    mean_path = (Ytr * Mtr).sum(0) / Mtr.sum(0).clamp(min=1)
    e_net, e_mean = err(P, Yte, Mte), err(mean_path.expand_as(Yte), Yte, Mte)
    # same-boss, same-move mean of OTHER swings (oracle identity + experience), leave one out
    Pm = torch.zeros_like(Yte); has = torch.zeros(len(te), dtype=torch.bool)
    for i, r in enumerate(te):
        idx = [j for j, q in enumerate(te) if q["b"] == r["b"] and q["move"] == r["move"] and j != i]
        if idx:
            w = Mte[idx]; Pm[i] = (Yte[idx] * w).sum(0) / w.sum(0).clamp(min=1); has[i] = True
    e_move = err(Pm[has], Yte[has], Mte[has]); e_net_h = err(P[has], Yte[has], Mte[has])
    blend = 0.5 * P + 0.5 * Pm
    print(f"k={k:2d}  zero-shot net {e_net:.3f}  | average path {e_mean:.3f}  | same-boss move mean (needs ID+experience) {e_move:.3f}"
          f"  | net on same swings {e_net_h:.3f}  | net+experience blend {err(blend[has], Yte[has], Mte[has]):.3f}", flush=True)
    if k == int(os.environ.get("SAVE_K", 12)):
        torch.save({"net": net.state_dict(), "mu": mu, "sd": sd, "k": k, "din": Xtr.shape[1]}, "checkpoints/r7_intuition.pt")
