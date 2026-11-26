import numpy as np, pickle, torch
exec(open("scripts/r7_train.py").read().split("tr = [r for r")[0])
from sklearn.linear_model import Ridge
tr = [r for r in rows if r["b"] >= 100]; te = [r for r in rows if r["b"] < 100]
for k in (12, 20):
    for name, A, B in (("unseen bosses", tr, te), ("seen bosses, held-out swings", [r for i, r in enumerate(tr) if i % 5], [r for i, r in enumerate(tr) if i % 5 == 0])):
        Xa, Ya, Ma = tensors(A, k); Xb, Yb, Mb = tensors(B, k)
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-4
        mp = (Ya * Ma).sum(0) / Ma.sum(0).clamp(min=1)
        out = [f"k={k} {name}: mean {err(mp.expand_as(Yb), Yb, Mb):.3f}"]
        for al in (10, 100, 1000):
            # fill masked targets with mean path for ridge
            Yf = torch.where(Ma > 0, Ya, mp.expand_as(Ya))
            m = Ridge(alpha=al).fit(((Xa - mu) / sd).numpy(), Yf.numpy())
            P = torch.tensor(m.predict(((Xb - mu) / sd).numpy()), dtype=torch.float32)
            out.append(f"ridge{al} {err(P, Yb, Mb):.3f}")
        # kNN on features
        D = torch.cdist((Xb - mu) / sd, (Xa - mu) / sd); nn_ = D.topk(10, largest=False).indices
        Pk = torch.stack([(Ya[n] * Ma[n]).sum(0) / Ma[n].sum(0).clamp(min=1) for n in nn_])
        out.append(f"knn10 {err(Pk, Yb, Mb):.3f}")
        print("  ".join(out), flush=True)
