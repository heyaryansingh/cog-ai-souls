"""Fast hands, learned: a reflex policy distilled from the planner (DAgger).

The planner (MPC with imagined rollouts) is slow: 8 actions x n rollouts x 30 ticks per decision.
The reflex maps a compact, boss-agnostic summary of the *belief* to an action in one MLP pass:

  - threat clock: P(next in-reach impact lands in t ticks), t = 1..20, 21+   (from the filter)
  - attack clock: same, ignoring reach (what would hit if I closed in)
  - punish clock: P(boss is helpless for exactly t more ticks)              (recovery windows)
  - scalars: unread-wind-up mass, idle mass, expected threat damage, unblockable share, mean
    reach margin, and the body: hp, stamina, heals, distance, boss hp, phase, affordability.

Because the features are a function of the belief (not of a particular boss), one reflex serves
every boss program the slow mind believes in, including programs it has never seen.
Training is DAgger (Ross et al. 2011): run the current reflex (mixed with the teacher at rate
beta), label every visited decision with the teacher's Q-values, aggregate, refit.
"""
from __future__ import annotations

import numpy as np

from .boss import D_MAX
from .env import N_ACTIONS
from .policies import OraclePlanner, MixturePlanner

NB = 21          # clock bins: 1..20 ticks, 21 = later
N_FEAT = 3 * NB + 16


class _BossTables:
    """Per-boss precomputation: for each wind-up state (p, k, d) the distribution of ticks to impact."""
    def __init__(self, boss):
        K = boss.K
        pmf = boss.windup_pmf()
        self.prims = [m.primitive for m in boss.moves]
        self.impact = np.array([(p.active if p.lunge else 1) for p in self.prims])
        self.attack = np.array([p.damage > 0 and not p.feint for p in self.prims], float)
        self.reach = np.array([p.reach + p.lunge for p in self.prims])
        self.dmg = np.array([p.damage for p in self.prims]) * boss.dmg_scale
        self.unbl = np.array([p.unblockable for p in self.prims], float)
        self.a = np.array([m.active for m in boss.moves])
        self.r = np.array([m.recovery for m in boss.moves])
        self.tell = boss.tell_ticks
        self.idle = np.array([p.name == "idle" for p in self.prims], float)
        # Hw[p, k, d, b]: wind-up d ticks elapsed -> remaining r ~ tail; impact after r + impact_k ticks
        self.Hw = np.zeros((2, K, D_MAX, NB))
        for p in range(2):
            for k in range(K):
                for d in range(1, D_MAX + 1):
                    tail = pmf[p, k, d - 1:]
                    s = tail.sum()
                    if s <= 0:
                        continue
                    tail = tail / s
                    t = np.arange(len(tail)) + self.impact[k]
                    b = np.minimum(t, NB) - 1
                    np.add.at(self.Hw[p, k, d - 1], b, tail)


_TABLES: dict = {}


def tables(boss):
    key = id(boss)
    if key not in _TABLES:
        if len(_TABLES) > 256:
            _TABLES.clear()
        _TABLES[key] = (_BossTables(boss), boss)   # keep a ref so id() stays unique
    return _TABLES[key][0]


def belief_features(f, obs, env) -> np.ndarray:
    """f: a (latency-projected) BossFilter. obs: current body observation. env: for the rules only."""
    T, R = tables(f.boss), env.R
    dist = obs["dist"]
    threat = T.attack * (dist <= T.reach + 0.3)
    W, A, Rr = f.W, f.A, f.R
    z = W.sum() + A.sum() + Rr.sum()
    W, A, Rr = W / z, A / z, Rr / z
    wk = np.einsum("pkd,pkdb->kb", W, T.Hw)                     # (K, NB) impact clock per move from wind-ups
    # active window: impact at tick `impact`; at stage tick j it is impact - j + 1 ticks away (if not past)
    ak = np.zeros_like(wk)
    for k in range(f.K):
        for j in range(1, int(T.a[k]) + 1):
            t = T.impact[k] - j + 1
            if t >= 1 and A[k, j - 1] > 0:
                ak[k, min(t, NB) - 1] += A[k, j - 1]
    clock = wk + ak
    threat_clock = (threat[:, None] * clock).sum(0)
    attack_clock = (T.attack[:, None] * clock).sum(0)
    # punish clock: remaining helpless ticks (recovery, plus the rest of an active window that already hit)
    pun = np.zeros(NB)
    for k in range(f.K):
        for j in range(1, int(T.r[k]) + 1):
            if Rr[k, j - 1] > 0:
                pun[min(T.r[k] - j + 1, NB) - 1] += Rr[k, j - 1]
    m_attack = clock.sum(1) * T.attack
    tot_attack = m_attack.sum() + 1e-9
    unread = W[:, :, :T.tell].sum() if T.tell > 0 else 0.0
    idle = (W.sum(axis=(0, 2)) * T.idle).sum()
    tclock_mass = (threat[:, None] * clock).sum(1)
    scal = np.array([
        unread, idle,
        (tclock_mass * T.dmg).sum() / 40.0,
        (tclock_mass * T.unbl).sum(),
        ((m_attack * (T.reach - dist)).sum() / tot_attack) / 3.0,
        obs["hp"] / R.max_hp, obs["stamina"] / R.max_stamina, obs["heals"] / max(R.heal_charges, 1),
        dist / 8.0, obs["boss_hp"] / env.boss.max_hp, float(obs["phase"]),
        float(obs["stamina"] >= R.dodge_cost), float(obs["stamina"] >= R.light_cost),
        float(obs["stamina"] >= R.heavy_cost), float(dist <= R.light_reach), float(dist <= R.heavy_reach),
    ])
    return np.concatenate([threat_clock, attack_clock, pun, scal]).astype(np.float32)


def mixture_features(filters, logw, obs, env, project) -> np.ndarray:
    """Belief features under a program mixture: the posterior-weighted average of per-program features."""
    w = np.exp(logw - logw.max())
    w /= w.sum()
    out = np.zeros(N_FEAT, np.float32)
    for f, wi in zip(filters, w):
        if wi > 1e-3:
            out += wi * belief_features(project(f), obs, env)
    return (out / max(float(w[w > 1e-3].sum()), 1e-9)).astype(np.float32)


# ---------------------------------------------------------------------------------- the network
def make_net(hidden=128):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(N_FEAT, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, N_ACTIONS))


class ReflexPolicy(OraclePlanner):
    """Acts from belief features with the distilled net. Optionally a *metareasoning gate*: when the
    reflex is unsure (top-action probability < gate), fall back to the planner for this decision.

    teacher_q: if set (DAgger data collection), also compute the planner's Q-values at every decision
    and store (features, Q); with probability beta execute the teacher's action, else the reflex's."""

    def __init__(self, net=None, gate=0.0, collect=False, beta=0.0, **kw):
        super().__init__(**kw)
        self.net, self.gate, self.collect, self.beta = net, gate, collect, beta
        self.data, self.planned, self.decisions = [], 0, 0

    def _features(self, obs, env):
        return belief_features(self.projected(self.filter), obs, env)

    def _probs(self, x):
        import torch
        with torch.no_grad():
            return torch.softmax(self.net(torch.from_numpy(x)[None]), -1)[0].numpy()

    def act(self, obs, env):
        self.decisions += 1
        x = self._features(obs, env)
        if self.collect:
            q = self.q_values(obs, env)
            self.data.append((x, q.astype(np.float32)))
            if self.net is None or self.rng.random() < self.beta:
                return self.actions[int(np.argmax(q))]
            return int(np.argmax(self._probs(x)))
        p = self._probs(x)
        if self.gate > 0 and p.max() < self.gate:
            self.planned += 1
            return self.actions[int(np.argmax(self.q_values(obs, env)))]
        return int(np.argmax(p))


class MixtureReflexPolicy(MixturePlanner):
    """Reflex over the slow mind's program mixture (features averaged over programs)."""

    def __init__(self, bosses, weights, net=None, gate=0.0, surprise=0, window=40, blend=0.0, **kw):
        """surprise > 0: model-adequacy gate. Plan with the model while it explains the cue stream;
        act on reflex while the last `window` ticks hold >= `surprise` cues the mixture found
        implausible (predictive p < 0.01). One noisy cue is common; a run of them is an unmodelled move."""
        super().__init__(bosses, weights, **kw)
        self.net, self.gate, self.planned, self.decisions = net, gate, 0, 0
        self.surprise, self.window = surprise, window
        self.blend = blend      # > 0: habit as a prior for deliberation, argmax_a Q(a) + blend * log pi_reflex(a)
        from collections import deque
        self._odd = deque(maxlen=window)
        self.reflexed = 0
        self._lp = deque(maxlen=100)     # per-tick mixture log predictive of the cue (model-adequacy signal)
        self.attempt = 0                 # set by the caller (fights already seen on this boss)
        self.metagate, self.meta_cost = None, 0.0
        self.blend_adapt = None   # R11: (lam_lo, lam_hi, a, b): lambda_t rises with model inadequacy u_t

    def _mix_lev(self):
        lw = self.prior_logw + np.array([f.log_evidence for f in self.filters])
        m = lw.max()
        return m + np.log(np.exp(lw - m).sum())

    def observe(self, obs):
        before = self._mix_lev()
        super().observe(obs)
        lp = self._mix_lev() - before
        self._odd.append(lp < np.log(0.01))
        self._lp.append(lp)

    def meta_features(self, p) -> np.ndarray:
        """Inputs for the learned metacontrol gate (R7), besides the belief features."""
        lw = self.prior_logw + np.array([f.log_evidence for f in self.filters])
        w = np.exp(lw - lw.max())
        w /= w.sum()
        lps = np.array(self._lp) if self._lp else np.zeros(1)
        return np.array([sum(self._odd) / 10.0, lps.mean() / 3.0, lps[-40:].min() / 7.0, float(p.max()),
                         float(-(w * np.log(w + 1e-12)).sum()), min(self.attempt, 5) / 5.0], np.float32)

    def reflex_probs(self, obs, env):
        import torch
        lw = self.prior_logw + np.array([f.log_evidence for f in self.filters])
        x = mixture_features(self.filters, lw, obs, env, self.projected)
        with torch.no_grad():
            p = torch.softmax(self.net(torch.from_numpy(x)[None]), -1)[0].numpy()
        return x, p

    def act(self, obs, env):
        import torch
        self.decisions += 1
        if self.surprise:
            if sum(self._odd) < self.surprise:
                self.planned += 1
                return super().act(obs, env)
            self.reflexed += 1
        x, p = self.reflex_probs(obs, env)
        if self.metagate is not None:       # R7: plan only where the predicted gain of planning beats its cost
            z = torch.from_numpy(np.concatenate([x, self.meta_features(p)]))[None]
            with torch.no_grad():
                gain = float(self.metagate(z)[0, 0])
            if gain > self.meta_cost:
                self.planned += 1
                return super().act(obs, env)
            self.reflexed += 1
            return int(np.argmax(p))
        if self.blend_adapt is not None:
            lo, hi, a_, b_ = self.blend_adapt
            if len(self._lp) >= 20:
                u = float(np.clip((-np.mean(self._lp) - a_) / (b_ - a_), 0.0, 1.0))
            else:
                u = 1.0 if self.attempt == 0 else 0.5
            self.lam_trace = getattr(self, "lam_trace", []) + [u]
            q = self.q_values(obs, env)
            return self.actions[int(np.argmax(q + (lo + (hi - lo) * u) * np.log(np.maximum(p[list(self.actions)], 1e-6))))]
        if self.blend > 0:
            q = self.q_values(obs, env)
            return self.actions[int(np.argmax(q + self.blend * np.log(np.maximum(p[list(self.actions)], 1e-6))))]
        if self.gate > 0 and p.max() < self.gate:
            self.planned += 1
            return super().act(obs, env)
        return int(np.argmax(p))


def load_reflex(spec: str):
    """'a.pt' or 'a.pt+b.pt+c.pt' (an ensemble: logits averaged as log-probs)."""
    import torch
    import torch.nn as nn
    nets = []
    for path in spec.split("+"):
        n = make_net()
        n.load_state_dict(torch.load(path))
        n.eval()
        nets.append(n)
    if len(nets) == 1:
        return nets[0]

    class Ensemble(nn.Module):
        def __init__(self, ms):
            super().__init__()
            self.ms = nn.ModuleList(ms)

        def forward(self, x):
            return torch.stack([torch.log_softmax(m(x), -1) for m in self.ms]).mean(0)
    return Ensemble(nets).eval()
