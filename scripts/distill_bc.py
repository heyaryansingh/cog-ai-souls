"""System 2 -> System 1: distil the slow planning mind (perceptual planner + strings learned by watching phase 1)
into the fast recurrent policy used by PPO (same perceived observations), by behaviour cloning. The distilled
policy is then the starting point for the PPO fine-tune arm of the phase-switch head-to-head (B19 / coordinator ask).
  STAGE=collect : teacher fights on phase 1, record (obs, teacher action) per decision -> data/bc_margit.pkl
  STAGE=train   : BC on the recurrent policy -> checkpoints/bc_margit.pt ; then evaluate it on phase 1 and 2"""
import os, sys, pickle, time
sys.path.insert(0, "/home/user/cog-ai-souls"); os.chdir("/home/user/cog-ai-souls")
__file__ = "scripts/arena3d_ppo.py"
exec(open(__file__).read().rsplit("if __name__", 1)[0])
import arena3d.scripted as SC
from arena3d.env import N_BUTTONS
os.environ["PPO_BOSS"] = "margit"
CI = {c: i for i, c in enumerate(R.CANDS)}


def to_cand(a):
    if a in CI:
        return CI[a]
    d, b = divmod(int(a), N_BUTTONS)
    same = [i for c, i in CI.items() if c % N_BUTTONS == b]
    return same[0] if same else CI[0]


def collect(seed):
    os.environ.pop("PPO_PHASE", None)
    w = R.Worker(make_train_env, seed); rng = np.random.default_rng(seed)
    lg = pickle.load(open("checkpoints/margit_grammar_phase1.pkl", "rb")); lg.update = False
    O, A = [], []; lg.reset(w.env); SC._LGP = lg; env = w.env
    done = False
    while not done:
        a = SC.learned_grammar_planner(env, rng)
        O.append(w.obs()); A.append(to_cand(a))
        r, done, out, n = w.step(A[-1])
    print("teacher fight", seed, out, round(1 - env.boss_hp / env.spec.hp, 3), env.t, len(A), flush=True)
    return np.array(O, np.float32), np.array(A), out, 1 - env.boss_hp / env.spec.hp


def evaluate(pol, phase, n=8, seed0=0):
    if phase == 2: os.environ["PPO_PHASE"] = "2"
    else: os.environ.pop("PPO_PHASE", None)
    res = []
    for k in range(n):
        w = R.Worker(make_train_env, 77000 + seed0 + k); h = torch.zeros(1, 1, pol.hidden); o = w.obs(); done = False
        while not done:
            with torch.no_grad():
                lg, _, h = pol(torch.from_numpy(o)[None, None], h)
            r, done, out, nfr = w.step(int(lg[0, 0].argmax()))
            o = w.obs() if not done else o
        h0 = getattr(w.env, "h0", w.env.spec.hp); res.append((out, 1 - w.env.boss_hp / h0))
    return res


if __name__ == "__main__":
    st = os.environ.get("STAGE", "collect")
    if st == "collect":
        from multiprocessing import Pool
        with Pool(int(os.environ.get("PROCS", 4))) as p:
            data = p.map(collect, range(int(os.environ.get("NFIGHT", 16))))
        pickle.dump(data, open("data/bc_margit.pkl", "wb"))
        print("teacher wins", sum(d[2] == "win" for d in data), "/", len(data), "dmg", np.mean([d[3] for d in data]).round(3))
    else:
        data = pickle.load(open("data/bc_margit.pkl", "rb")); torch.manual_seed(0); torch.set_num_threads(4)
        pol = R.Policy(); opt = torch.optim.Adam(pol.parameters(), 1e-3); L = 64
        chunks = [(o[i:i + L], a[i:i + L]) for o, a, _, _ in data for i in range(0, len(a) - 8, L // 2)]
        for ep in range(int(os.environ.get("EPOCHS", 30))):
            np.random.shuffle(chunks); tot, acc, nn_ = 0, 0, 0
            for i in range(0, len(chunks), 32):
                B = chunks[i:i + 32]; T = min(len(c[1]) for c in B)
                X = torch.tensor(np.stack([c[0][:T] for c in B])); Y = torch.tensor(np.stack([c[1][:T] for c in B]))
                lg, _, _ = pol(X, torch.zeros(1, len(B), pol.hidden))
                loss = nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), Y.reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(B); acc += (lg.argmax(-1) == Y).float().sum().item(); nn_ += Y.numel()
            print("epoch", ep, "loss", round(tot / len(chunks), 3), "acc", round(acc / nn_, 3), flush=True)
        torch.save(pol.state_dict(), "checkpoints/bc_margit.pt")
        for ph in (1, 2):
            r = evaluate(pol, ph); print("BC policy phase", ph, "wins", sum(x[0] == "win" for x in r), "/", len(r), "dmg", np.mean([x[1] for x in r]).round(3), flush=True)
