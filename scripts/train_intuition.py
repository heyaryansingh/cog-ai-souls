"""Inference compilation: simulate (boss, log) pairs from the META prior, train the intuition net.
  python scripts/train_intuition.py --n 40000 --epochs 15
"""
import argparse, os, sys, time
from multiprocessing import Pool
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtb.intuition import IntuitionNet, loss_fn, simulate

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=40000)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--procs", type=int, default=4)
ap.add_argument("--out", default="checkpoints/intuition.pt")
a = ap.parse_args()
os.makedirs("checkpoints", exist_ok=True)
data_path = f"checkpoints/ic_data_{a.n}.npz"
t0 = time.time()
if os.path.exists(data_path):
    D = dict(np.load(data_path)); parts = [D[k] for k in ["X", "L", "C", "M", "H", "I"]]
else:
    chunk = a.n // (a.procs * 4)
    with Pool(a.procs) as pool:
        res = pool.starmap(simulate, [(chunk, 77 + i) for i in range(a.procs * 4)])
    parts = [np.concatenate([r[j] for r in res]) for j in range(6)]
    np.savez_compressed(data_path, X=parts[0], L=parts[1], C=parts[2], M=parts[3], H=parts[4], I=parts[5])
print(f"data {len(parts[0])} fights in {time.time()-t0:.0f}s; mean tokens {parts[1].mean():.0f}", flush=True)
T = [torch.from_numpy(p) for p in parts]
n = len(T[0]); nv = n // 10
torch.manual_seed(0); torch.set_num_threads(a.procs)
net = IntuitionNet()
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
for ep in range(a.epochs):
    perm = torch.randperm(n - nv) + nv
    net.train()
    for b in range(0, len(perm), 128):
        idx = perm[b:b + 128]
        loss, _, _ = loss_fn(net, [t[idx] for t in T])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    net.eval()
    with torch.no_grad():
        vl, ce, nll = loss_fn(net, [t[:nv] for t in T])
        lc, lm, _ = net(T[0][:nv], T[1][:nv])
        acc = (lc.argmax(-1) == T[2][:nv]).float().mean().item()
        h = T[4][:nv] > 0
        mae = (lm[..., 0] - T[3][:nv]).abs()[h].mean().item()
    print(f"epoch {ep} val loss {vl.item():.3f} count-acc {acc:.3f} logmed-MAE {mae:.3f} ({time.time()-t0:.0f}s)", flush=True)
    torch.save(net.state_dict(), a.out)
