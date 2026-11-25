# Reading the Boss, v2: learning from death

*Refined design after professor feedback, 25 Nov 2026. Builds on [proposal.md](proposal.md). Aryan Singh.*

The professor pointed to **AgentOdyssey** and **AI GameStore**. Both tighten the project. AgentOdyssey names combat as a skill agents fail to learn at test time. AI GameStore shows that LLM agents can't keep up with real-time games. So v2 makes three changes:

1. **The question gets sharper.** Can an agent learn a new opponent at test time, from a handful of deaths, the way a person does? And what is the minimum cognitive machinery that makes this work?
2. **The architecture gets one clean idea.** A *slow mind and fast hands*: slow Bayesian program induction between attempts, and fast exact inference during the fight.
3. **The evaluation borrows both papers' protocols.** That means AgentOdyssey-style test-time learning (fixed budget, reset between bosses, no training on test bosses) and AI GameStore-style human-normalized scores from people playing the same bosses.

---

## 1. Where this sits relative to the two papers

| | AgentOdyssey (Zhang, …, Khashabi, Shu; NeurIPS 2026) | AI GameStore (Ying, …, Gershman, Tenenbaum; Feb 2026) | **Reading the Boss** |
|---|---|---|---|
| What it is | Procedural text-game generator plus benchmark for **test-time continual learning** | 100 LLM-generated versions of popular human games, scored against humans | Procedural **boss generator** plus a **cognitive agent** |
| Setting | Turn-based, open-ended text worlds (crafting, NPCs, quests) | Real-time 2D/3D games, paused every second for the model | **Real-time combat**: 20 ticks/s, frame-tight dodges |
| Agents | LLM agents with memory: long context, RAG, Voyager, SFT, latent memory | Zero-shot frontier LLMs with a scratchpad | **Bayesian program induction + exact/SMC inference + planning**, with no LLM in the core |
| Key finding we build on | Agents show "weak procedural combat skill," myopic exploration and hallucinated rules; SFT forgets | Best models score under 10% of the human median and run 10× slower than people | We target exactly these failures: combat, exploration and calibrated rules, all in real time |
| What it does not do | No bosses, no learning from death, no cross-game transfer | No learning across sessions, no opponent modeling, no transfer (variants were built but not evaluated) | We measure all three |

**One-line positioning.** AgentOdyssey asks whether agents can learn a world at test time, and AI GameStore asks whether they can play human games as well as humans. This project takes the hardest slice of both: learning a *real-time adversary* at test time. It tests whether the course's structured Bayesian approach does what memory-augmented LLMs and deep RL can't.

---

## 2. The environment: MiniSouls (kept deliberately small)

**Arena.** One lane (1D distance), with facing and a dodge direction. 20 ticks per second. Pure NumPy, so it runs at well over 10k ticks per second on a laptop core. A 2D arena is a stretch goal and is not needed for the science.

**Player.** HP, stamina, and 7 actions: approach, retreat, light attack, heavy attack, dodge (i-frames), block, heal (3 charges). The player's body follows rules R = {i-frame length, dodge cost, stamina regen, heal time}. In test set T4 these rules change.

**Boss = a probabilistic program**, built from a shared library of about 8 primitives: overhead slam, sweep, lunge, n-hit combo, delayed slam, AoE burst, grab, feint.

```
Boss B:
  moves k = 1..K (K ∈ 3..7), each with:
     c_k ∈ library                       # which primitive
     w_k  windup ticks ~ LogNormal(μ_c, σ_c)
     a_k  active ticks, r_k recovery ticks, range ρ_k, damage d_k
  next move ~ π[k_prev, dist_bin, phase]   # Dirichlet rows
  phase 2 at 50% HP: speed factor s, maybe one extra move
Per tick the agent sees y_t = noisy animation-cue token of (c_k, stage),
plus distance, its own HP/stamina and the boss's HP.
```

Cues are **ambiguous early in a wind-up**. A delayed slam and a normal slam look identical for the first few ticks, and a feint looks like a lunge. That ambiguity is the point: it makes hedging between hypotheses matter.

**Splits** (fixed seeds):
- Train: 40 bosses.
- T1: same moves, new timings.
- T2: new combinations of known moves.
- T3: one unseen primitive.
- T4: same bosses, changed player rules (the "different souls-like game").

**Protocol** (adapted from AgentOdyssey):
- Each test boss gets a budget of **20 attempts**.
- Memory is reset between test bosses, except for the cross-boss prior, which is frozen after training.
- Nothing trains on test bosses beforehand.

---

## 3. The architecture: slow mind, fast hands

```
                ┌──────────────── LIFELONG (across bosses) ────────────────┐
                │  Hierarchical prior Θ: per-primitive timing hyperparams,  │
                │  co-occurrence of moves, phase-2 tendencies, rule priors  │
                └──────────────▲───────────────────────────┬───────────────┘
                   empirical Bayes after each boss          │ prior
                ┌──────────────┴───────────────────────────▼───────────────┐
                │ SLOW MIND (on each death / end of attempt)                │
                │ Posterior over boss programs = N weighted particles B¹..Bᴺ│
                │ MCMC rejuvenation: FFBS segmentation → conjugate Gibbs on  │
                │ timings + transitions → split/merge MH on moves (CRP)     │
                │ Scouting value: expected info gain about B for next try   │
                └──────────────┬───────────────────────────▲───────────────┘
                   particles Bⁱ, weights                    │ fight log
                ┌──────────────▼───────────────────────────┴───────────────┐
                │ FAST HANDS (every tick, 20 Hz)                            │
                │ For each Bⁱ: exact forward filter over (move, stage, τ)   │
                │ Mixture over i, with weights updated online (RBPF)        │
                │ Action = argmax expected utility over Monte Carlo rollouts│
                │ sampled from the predictive (+ λ·info gain if scouting)   │
                └───────────────────────────────────────────────────────────┘
```

### 3.1 Fast hands: exact inference inside, importance weights outside

Given one boss hypothesis Bⁱ, the hidden state z_t = (move k, stage ∈ {windup, active, recovery}, ticks τ in stage) has at most about 7 × 3 × 40 ≈ 850 values. **The forward algorithm is exact and cheap** (Lec 8), so no sampling is needed inside a hypothesis.

Across N ≈ 16–32 boss hypotheses, each forward filter also gives the marginal likelihood p(y_1:t | Bⁱ). That updates the hypothesis weights online, which is **sequential importance sampling over static parameters** (Lec 9, 12).

Together this is a **Rao-Blackwellized particle filter**: exact where the state is small, sampled where it is large. Cost: 32 × 850 states × 20 Hz, which is trivial in NumPy.

**Acting.** For each of the 7 actions, roll out ~64 futures of 1–1.5 s. Each rollout samples a hypothesis by weight, a current state from its filter, and the boss's next moves from that hypothesis. The agent knows its own body (rules R), so it can simulate itself. It picks the action with the highest expected utility: damage dealt, minus damage taken, minus stamina cost. The agent replans only when the cue changes or every 3 ticks, which keeps it fast.

**The behavior this produces.** The agent dodges when the predicted hit time falls inside its i-frame window. When it is unsure whether a slam is the delayed version, it waits or backs off. It punishes only when it is confident the boss is in recovery. These are exactly what a good player does, and here they fall out of expected utility rather than being scripted.

### 3.2 Slow mind: Bayesian program induction from the fight log

After each attempt, which usually ends in death, the agent updates its posterior over boss programs using **resample-move SMC** (Saad et al. 2023; Lec 11–12):

1. **Reweight** each particle by the likelihood of the new attempt's log. This comes free from the forward filter.
2. **Resample** if the effective sample size drops below N/2.
3. **Rejuvenate** with MCMC moves (Lec 10–11):
   - **FFBS**: sample the move sequence of the attempt with forward-filter backward-sample. This segments the log into moves.
   - **Conjugate Gibbs**: given the segmentation, windup/active/recovery durations are LogNormal with a Normal-Inverse-Gamma prior, and transition rows are Dirichlet-multinomial. Both updates are closed form.
   - **Split/merge Metropolis–Hastings** on moves under a **Chinese restaurant process** prior. "These two slams are really one move with variable timing" versus "this is a new move I haven't seen" (T3).
   - **Primitive relabeling**: Gibbs on c_k, using the library prior Θ (the size principle: a primitive with tight timing that fits is preferred over a vague one).

**Scouting** (learning from death on purpose). Early attempts are worth more as information than as wins. In attempts 1–3 the planner adds λ · expected information gain about B, estimated from particle disagreement. In practice this means: stay at mid-range, bait moves, watch. Souls players call these scouting runs. This answers AgentOdyssey's "myopic exploration" failure with a principled exploration term.

### 3.3 Lifelong: learning to learn across bosses

After each training boss, Θ is updated by empirical Bayes from the boss's posterior:
- the typical timing of each primitive, and how much it varies;
- which primitives co-occur;
- how phase 2 changes speed.

Θ is frozen before testing. Learning to learn is just "inference one level up" (course motif M11).

**T4 rule inference.** The rules R are latent too. When the agent dodges, the outcome (hit or no hit at a known timing) is evidence about i-frame length. A small Bayesian update on R lets the agent re-learn its own controls in a "different game." This is a self-model, the counterpart of the boss model.

### 3.4 Where an LLM could fit (optional, not core)

An LLM could act as a **data-driven proposal** for the slow mind: read the text fight log and suggest a new move or a restructured transition, with the suggestion accepted or rejected by Metropolis–Hastings so the posterior stays correct. This ties the project to the course's "proposals are the bottleneck" motif and to both papers' LLM focus. It is a stretch goal, because the core must run offline on a laptop.

---

## 4. Baselines

| | Agent | Why it's there |
|---|---|---|
| B1 | PPO, MLP, trained on the 40 training bosses and then run on test bosses | Model-free RL, the DSLE-style baseline |
| B2 | Recurrent PPO (RL²-style) trained across training bosses | Meta-RL: amortized, black-box learning of a new boss within its 20 attempts |
| B3 | PPO fine-tuned on each test boss for the same 20 attempts | Plain test-time learning by gradient descent |
| B4 | LLM agent with memory, AgentOdyssey/GameStore-style: text-rendered fight, paused decisions, notes kept between attempts | Connects directly to both papers. Run on 5 bosses only, to cap API cost (or use a local 4B model) |
| B5 | Scripted "dodge every wind-up" heuristic | Floor |
| H | Humans playing the same bosses in a pygame viewer | Human-normalized score and human learning curves |

**Ablations of our agent:**
- **MAP**: N = 1, a single boss hypothesis. Tests "averaging beats committing."
- **No Θ**: flat prior. Tests learning to learn.
- **No CRP**: fixed number of moves. Tests handling of novelty.
- **No scouting**: λ = 0. Tests whether information-seeking matters.
- **Oracle B**: the true boss program is given. This is the upper bound, and it separates *learning* errors from *acting* errors.

---

## 5. Experiments and what would make them interesting

**E1 (headline): learning curves in deaths.** Win rate against attempt number on held-out bosses, for our agent, B1–B4 and humans, plus deaths to first win. Prediction: our agent and humans win within roughly 3–8 attempts. B1 and B3 barely move in 20 attempts. B2 does well on T1 but drops on T2–T4.

**E2: what drives it.** The ablation table on T1–T4. Prediction: MAP fails on delayed attacks and feints, no-Θ is slower on every set, and no-CRP collapses on T3.

**E3: transfer.** T2 (new combinations) and T4 (new rules) measure transfer explicitly. This is the cross-game transfer the original idea asked for.

**E4 (the cognitive-science payoff): does the agent learn like a person?** With 5–8 human players on 6 bosses:
- Compare per-move death rates between agent and humans. Do both die to the delayed slam?
- Compare how many deaths each needs before they "get" each move.
- Check whether the agent's posterior uncertainty about a move predicts where humans die.

A positive correlation would say the Bayesian model captures something about *human* opponent learning, not just good play. That links back to Lake et al.'s argument and to AI GameStore's human-comparison approach.

**E5: calibration.** Predictive log-likelihood of hit timing, and reliability curves. An agent that knows when it doesn't know is the property LLM agents lacked in AgentOdyssey (hallucinated recipes) and in AutumnBench.

---

## 6. Course mapping (unchanged in spirit, now one component each)

| Lecture topic | Component |
|---|---|
| World and agent models (Lec 1–2), intuitive psychology (Lake et al.) | Boss program as a generative model of another agent; self-model of the player's rules |
| Bayesian concept learning, size principle (Lec 3–5) | Recognizing a move from a partial wind-up; primitive relabeling |
| Bayes nets and PPLs (Lec 6–7) | The boss as a hidden semi-Markov model in a probabilistic program |
| Exact inference (Lec 8) | Forward filtering inside each hypothesis; FFBS segmentation |
| Importance sampling (Lec 9) | Online weights over boss hypotheses |
| MCMC (Lec 10–11) | Conjugate Gibbs, split/merge MH, CRP |
| SMC (Lec 11–12) | Resample-move SMC over boss programs (Saad et al. 2023) |
| BPL and learning to learn (Lake 2015, 2017) | Compositional primitive library; hierarchical prior Θ |
| EMPA, Frostbite (Tsividis 2021) | Model-based versus model-free sample efficiency, measured in deaths |

---

## 7. Feasibility and scope

**Laptop budget.**
- Env: NumPy, over 10k ticks/s.
- Our agent needs no gradient training. Fast hands run faster than real time; slow-mind rejuvenation takes about 1–5 s per death; fitting Θ over 40 bosses takes minutes.
- PPO baselines: about 1–3 CPU-hours each at 2–5M steps, with 3 seeds.
- B4: a few dollars of API calls, or a local 4B model run slowly.

**Scope tiers.** Build in this order so there is always a working result.

| Tier | Contents | Done when |
|---|---|---|
| **MVP** | MiniSouls (1D, 4 primitives) + fast hands with the **true** boss model (oracle) + B1 PPO + B5 | Oracle agent beats bosses; PPO doesn't in a matched budget |
| **Core** | Slow mind (reweight + FFBS + conjugate Gibbs, fixed K) + Θ + MAP ablation + T1/T2 + E1/E2 | Learning curves in deaths; MAP gap shown |
| **Full** | CRP split/merge, scouting, T3/T4, B2/B3, calibration (E5) | All of E1–E3, E5 |
| **Stretch** | Human study (E4), LLM baseline B4, LLM-as-proposal, 2D arena | Any of these |

**Suggested schedule from today:**
- Week of Nov 24: MVP. Env, cue model and oracle agent. Start PPO running in the background.
- Week of Dec 1: Core. Slow mind, Θ, T1/T2, ablations.
- Week of Dec 8: Full tier, figures, write-up. Human pilot if there is time; it only needs a few friends and 30 minutes each.

**Main risks and fallbacks:**
- **Split/merge mixes poorly.** Fall back to fixed K with an "unknown move" slot.
- **The planner is too myopic for combos.** Extend the horizon only for combo primitives, or add a learned value estimate at the rollout leaf.
- **PPO is surprisingly strong in-distribution.** Then the story shifts to deaths-to-competence and transfer (T2–T4), where it should fall off.

---

## 8. Refined one-paragraph pitch (for an updated proposal or the report intro)

Souls-like players learn a new boss in a handful of deaths: they read each attack's wind-up, dodge on the right frame, and carry what they learned to the next boss. Current agents can't. Deep RL wins 0% of Dark Souls boss fights from pixels (DSLE 2026), LLM agents score under 10% of humans on real-time games (AI GameStore), and memory-augmented LLMs fail to learn combat at test time (AgentOdyssey). We build MiniSouls, a procedural generator of boss fights defined as probabilistic programs, and an agent with a **slow mind and fast hands**. Between attempts, resample-move SMC with MCMC rejuvenation infers the boss's program from the fight log. During the fight, exact forward filtering inside each hypothesis, combined across hypotheses by importance weighting, predicts when the next hit lands, and the agent acts by expected utility. A hierarchical prior learned across bosses gives learning to learn, and a CRP lets it posit new moves. We compare learning curves measured in deaths against PPO, meta-RL, an LLM agent with memory, and human players, and test transfer to new move combinations, unseen moves and changed game rules.

## Sources

- AgentOdyssey: https://agentodyssey.github.io/ · paper https://arxiv.org/abs/2606.24893 · code https://github.com/agentodyssey/agentodyssey (homepage, abstract and README read; full paper not read)
- AI GameStore: https://arxiv.org/abs/2602.17594 · https://aigamestore.org (abstract and about 40% of the HTML read; the rest from a summary. The paper itself states both "under 10%" and "under 30%" of human performance in different places)
- DSLE: https://arxiv.org/abs/2608.09902 (abstract)
- Saad et al. 2023 SMC learning; Lake et al. 2015, 2017; Tsividis et al. 2021: course readings, see `../synthesis/papers/`.
- Chopin et al. on SMC² and Rao-Blackwellized particle filters (Doucet et al. 2000) are cited from memory.
