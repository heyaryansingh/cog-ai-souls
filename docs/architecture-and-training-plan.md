# Reading the Boss, v3: architecture and training plan

*25 Nov 2026. Aryan Singh. Supersedes the architecture section of [refined-design.md](refined-design.md); the environment, splits, baselines and course mapping there still hold. This version adds:*

- *a research base (§1);*
- *a trainable neural component that makes inference fast without giving up correctness (§3);*
- *a full staged training plan with data, losses and laptop compute (§4);*
- *a measurable definition of "adaptive intelligence" for the evaluation (§5).*

> **Update 25 Nov:** capability targets, milestones, end results and the design changes from reading all papers in full are in [capabilities-and-results.md](capabilities-and-results.md) (v4). Where they differ, v4 wins.

---

## 0. The design in five sentences

1. **What the agent is.** It adapts to a new boss **by inference, not by gradient descent**. At test time no weights change. Everything it learns about a boss lives in a posterior over boss programs.
2. **How the posterior stays fast.** Speed comes from a small neural network, the **intuition net**, trained on millions of simulated fights to *propose* boss programs from a fight log. Correctness comes from **SMC, which weighs those proposals against the evidence**. In short, neural proposes and Bayes disposes. This is the inference-compilation idea from probabilistic programming, and it matches cognitive models of goal inference as "top-down filtering of bottom-up proposals."
3. **How it acts.** It acts with two speeds. A distilled **reflex policy** handles familiar situations, and a **planner** that simulates futures inside the posterior takes over when uncertainty is high. A learned **metareasoning gate** decides which one runs, following resource-rational accounts of when to think.
4. **What carries across bosses.** Three things carry over: a hierarchical prior over move timings, a library of boss *families* under a Chinese restaurant process, and a self-model of the game's rules.
5. **The headline experiment.** The amortized network alone should do well on familiar bosses and fail on novel ones, while SMC on top keeps working. If that holds, it shows *when* explicit Bayesian inference is needed on top of learned intuition, which is a sharp, testable claim about adaptive intelligence.

---

## 1. What the research says, and what each finding changes in the design

### 1.1 How people learn opponents and skills

| Finding | Source | Design consequence |
|---|---|---|
| People reproduce time intervals as Bayesian observers. Their estimates are pulled toward the mean of previously seen intervals, more so for longer intervals, because timing noise grows with the interval (the scalar property) | Jazayeri & Shadlen 2010, *Nat Neurosci* ([PMC2916084](https://pmc.ncbi.nlm.nih.gov/articles/PMC2916084)); review of temporal context in timing models ([PMC9552499](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9552499/)) | Give the agent a **scalar-noise timing model** as an option. It predicts that **delayed attacks are adversarial examples against a timing prior**, a trick game designers use on purpose. Agent and humans should both die to them early; test this in E4 |
| People combine a prior with noisy sensory feedback in a Bayes-optimal way when learning motor tasks | Körding & Wolpert 2004, *Nature* ([link](https://ideas.repec.org/a/nat/nature/v427y2004i6971d10.1038_nature02169.html)) | Dodge timing is decided from the posterior predictive hit time, not from a point estimate |
| In a very large sample of online game players, those who explored more early on learned faster later | Stafford & Dewar 2014, *Psych Sci* ([link](https://www.psychologicalscience.org/journals/psychological-science/0956797613511466/)); long-term motor learning in Aim Lab data, Listman et al. 2021 ([Frontiers](https://www.frontiersin.org/articles/10.3389/fnhum.2021.777779)) | Justifies **scouting attempts** (information-seeking early, exploitation later) and gives a human pattern to compare against |
| People learn physical tool use in a few tries by sampling actions, simulating outcomes in a mental model and updating from failures (SSUP) | Allen, Smith & Tenenbaum 2020, *PNAS* ([arXiv 1907.09620](https://arxiv.org/abs/1907.09620)) | The closest cognitive precedent for **learning from death**: sample, simulate, update. The slow mind is SSUP applied to an adversary |
| People learn Atari games in minutes; theory-based agents (EMPA) match that efficiency where DDQN needs orders of magnitude more data | Tsividis et al. 2017 ([CBMM](https://cbmm.mit.edu/publications/human-learning-atari-0)); Tsividis et al. 2021 ([arXiv 2107.12544](https://arxiv.org/abs/2107.12544), course reading) | Use the same comparison metric: experience needed to reach competence |
| People exploit opponents with simple sequential patterns in rock-paper-scissors, but fail on more complex dependencies. An LLM-based cognitive model reproduces this and shows that **hypothesis generation is the bottleneck** | Brockbank & Vul 2021, *Games* ([link](https://mdpi.com/2073-4336/12/3/70)); Cross et al. 2025 ([arXiv 2508.06503](https://arxiv.org/abs/2508.06503)) | Manipulate the **order of boss transition dependencies** (first-order versus conditioned on two prior moves) as a controlled difficulty axis, and test whether humans and the agent both break at the same point |
| People segment continuous activity into events at moments of high prediction error, and latent-cause models built on a CRP explain segmentation and learning without interference | Event segmentation theory; Franklin et al. 2020, *Psych Rev* ("Structured event memory"); Kumar et al. 2023 (Bayesian surprise predicts event segmentation); Lu et al. 2024, *Sci Rep* (LCNet) | Detect **move boundaries by surprise**, and use a **CRP over moves and over boss families**. Shared structure lives in weights, and context-specific structure lives in latent causes, which is exactly LCNet's split |
| People infer other agents' goals online with something like SMC over bounded-rational plans (SIPS), with bottom-up proposals filtered top-down | Zhi-Xuan et al. 2020 ([arXiv 2006.07532](https://arxiv.org/abs/2006.07532)); Zhi-Xuan et al. 2024 ([arXiv 2407.16770](https://arxiv.org/abs/2407.16770)) | The **intuition net (bottom-up) plus SMC (top-down)** split is borrowed directly from this |
| How long chess players think tracks the value of computation | Russek et al., Princeton ([link](https://collaborate.princeton.edu/en/publications/time-spent-thinking-in-online-chess-reflects-the-value-of-computa/)); resource-rational analysis (Lieder & Griffiths 2020) | The **metareasoning gate**: plan only when the expected gain of planning beats its cost |

### 1.2 What recent AI work offers

| Work | What it shows | How we use it |
|---|---|---|
| **AgentOdyssey** (Zhang … Shu, NeurIPS 2026) | Memory-augmented LLM agents fail to learn combat at test time; SFT agents forget | Evaluation protocol and the LLM baseline |
| **AI GameStore** (Ying … Tenenbaum, 2026) | Frontier LLMs score far below the human median on real-time games | Human-normalized scoring |
| **Assessing adaptive world models with novel games** (Ying … Tenenbaum, 2025, [arXiv 2507.12821](https://arxiv.org/abs/2507.12821)) | Argues for measuring *world-model induction speed* under continually refreshed novelty | Our definition of adaptive intelligence (§5) |
| **OneLife** (Khan … Bansal, 2025, [arXiv 2510.12088](https://arxiv.org/abs/2510.12088)) | Learns symbolic world models of *stochastic* environments from one unguided life, as **conditionally activated programmatic laws** inside a PPL | Boss rules as **precondition → effect laws** (for example, "if distance < 2 and phase 2, then grab is likely"). Only active laws enter the likelihood, which keeps inference cheap |
| **OPINE-World** (Courtis … Sanner, 2026, [arXiv 2607.01531](https://arxiv.org/abs/2607.01531)) | A Bayesian hypothesis space with *provisional* objects and unresolved interactions, revised at test time and steered by exploration; raises an LLM's action efficiency on ARC-AGI-3 from 1.5% to 78.4% | Supports "provisional moves" (the CRP's new-table option) and ontology-error-driven scouting |
| **PoE-World** (2025), **WorldCoder** (2024), **WorldLLM** (2025, [arXiv 2506.06725](https://arxiv.org/abs/2506.06725)) | LLM-written programmatic world models; WorldLLM refines hypotheses by Bayesian inference with an LLM as the proposal | Shows the pattern works with LLM proposals. We use a small trained network instead so it runs on a laptop |
| **Language and Experience** (Colas … Tenenbaum, ICLR 2026, [arXiv 2509.00074](https://arxiv.org/abs/2509.00074)) | Joint inference over executable world models from experience *and* advice speeds learning in humans and models across 10 games | Stretch: a **"read the wiki" condition** where a one-line tip ("roll toward the hammer on the third swing") enters as likelihood evidence |
| **Inference compilation and data-driven proposals** (Le, Baydin & Wood 2017; Paige & Wood 2016, [arXiv 1602.06701](https://arxiv.org/abs/1602.06701); Perov et al. 2015, [arXiv 1512.04387](https://arxiv.org/abs/1512.04387)); **Reinforced SMC** (Choi et al. 2025, [arXiv 2510.11711](https://arxiv.org/abs/2510.11711)) | Train a neural network on samples from the model to act as an SMC/MH proposal, which keeps importance-weight correctness | The **intuition net's training objective** |
| **In-context Bayesian inference with transformers**: prior-fitted networks; Reuter et al. 2025 ([arXiv 2501.16825](https://arxiv.org/abs/2501.16825)); Distribution Transformers (ICML 2026, [arXiv 2502.02463](https://arxiv.org/abs/2502.02463)) | Transformers trained on prior samples approximate posteriors in milliseconds, but only within the prior they were trained on | The **amortized-only baseline**, and the reason for the headline experiment on novelty |
| **Algorithm Distillation** (Laskin et al. 2022, [arXiv 2210.14215](https://arxiv.org/abs/2210.14215)) and **Recurrent AD** (Ma et al. 2026, [arXiv 2609.35333](https://arxiv.org/abs/2609.35333)) | Transformers learn an RL algorithm in context from learning histories; RAD compresses long histories into a fixed latent buffer | The strongest **black-box test-time learning baseline**: an in-context RL agent across attempts |
| **Tiny Recursive Model** (Jolicoeur-Martineau 2025, [arXiv 2510.04871](https://arxiv.org/abs/2510.04871)) | A 7M-parameter, 2-layer network that refines its answer recursively gets 45% on ARC-AGI-1 and 8% on ARC-AGI-2 | The intuition net uses **recursive refinement**: the same small block applied several times, with SMC feedback between passes (§3.3) |
| **Bayesian nonparametric hidden semi-Markov models** (Johnson & Willsky 2013, *JMLR*, [link](https://www.jmlr.org/papers/v14/johnson13a.html)) | An HDP-HSMM with explicit durations and efficient Gibbs samplers; the `pyhsmm` library implements it | The **formal backbone of the boss model**, and a reference implementation to check our sampler against |
| **FightingICE** fighting-game competition; RHEA with online opponent model learning (Tang … Lucas 2020, [arXiv 2003.13949](https://arxiv.org/abs/2003.13949)) | A long-running real-time fighting-game AI platform where opponent modeling helps | Optional **external-validity testbed** that, unlike SoulsGym, runs on a laptop |

---

## 2. Information requirements: what the agent needs to see and know

A cognitive-AI design should say what each part of the agent needs. Each row lists what a module gets, what it must output, and where its knowledge comes from.

| Module | Input | Output | Built in or learned |
|---|---|---|---|
| **Perception** (given by env) | Raw game state | Event tokens: noisy animation cue (primitive id + stage, with confusable pairs), distance, facing, HP, stamina, hit events | Built in. We do not study vision; a pixel version is a stretch goal |
| **Self-model** | Own action history plus outcomes (was I hit during a dodge?) | Posterior over rules R (i-frame length, stamina costs, heal time) | Prior built in; posterior inferred online |
| **Boss model** (the generative program) | Prior Θ, family library | Probability of any cue sequence | Structure built in (an HSMM with conditional laws); parameters inferred |
| **Fast filter** | Cue stream, N boss hypotheses | Belief over (move, stage, ticks) per hypothesis, hypothesis weights, predictive hit-time distribution | Exact algorithm, no learning |
| **Intuition net** | Tokenized fight log (all attempts so far on this boss) | Proposal distribution over boss programs: segmentation, primitive labels, timing parameters, transition counts, family | **Trained** (§4 stage 2) |
| **Slow mind** (SMC) | Intuition proposals, fight logs, Θ | Weighted posterior over boss programs | Algorithm; no learning |
| **Planner** | Belief and the self-model | Action values from simulated rollouts | Algorithm; utility weights tuned once |
| **Reflex policy** | Compact belief features: predicted hit-time quantiles, stage probabilities, own stamina/HP, distance | Action | **Trained** by imitating the planner (§4 stage 4) |
| **Metareasoning gate** | Posterior entropy, ESS, time since surprise, reflex confidence | Plan or react; whether to request slow-mind rejuvenation mid-fight | **Trained** to predict the planner's value gain (§4 stage 4) |
| **Lifelong memory** | Posteriors from past bosses | Updated Θ, family library (CRP) | Empirical Bayes, done between bosses |

**Minimum information the agent needs** (also a check on the environment design): the cue must carry *some* information about the move before the hit, or no learner can dodge on reaction alone. We control this with a **cue informativeness** parameter: how many ticks pass before confusable moves become distinguishable. Sweeping it gives a clean curve showing where prior knowledge has to replace perception. Souls players know this as "reading the boss" versus "memorizing the boss."

---

## 3. The architecture in detail

```
                 ┌───────────────────── LIFELONG MEMORY ─────────────────────┐
                 │ Θ: hierarchical timing priors per primitive                │
                 │ Family library: CRP over boss families (shared moves)      │
                 │ Self-model prior over rules R                               │
                 └──────────▲─────────────────────────────┬───────────────────┘
          empirical Bayes   │                             │ priors
                 ┌──────────┴─────────────────────────────▼───────────────────┐
  fight log ───► │ SLOW MIND: resample-move SMC over boss programs              │
                 │   proposals ◄── INTUITION NET q_φ (small recursive          │
                 │                 transformer, trained on simulated fights)    │
                 │   weights   ◄── exact likelihood from the fast filter       │
                 │   MCMC      ◄── FFBS + conjugate Gibbs + split/merge (CRP)  │
                 └──────────┬─────────────────────────────▲───────────────────┘
          N hypotheses Bⁱ   │                             │ surprise → early rejuvenation
                 ┌──────────▼─────────────────────────────┴───────────────────┐
  cue stream ──► │ FAST FILTER: exact HSMM forward pass per Bⁱ (RBPF)          │
                 │   → predictive hit-time distribution, stage beliefs, ESS     │
                 └──────────┬──────────────────────────────────────────────────┘
                            ▼
                 ┌──────────────────────────────────────────────────────────────┐
                 │ METAREASONING GATE: E[gain from planning] > cost ?           │
                 │   no  → REFLEX POLICY (small MLP on belief features)         │
                 │   yes → PLANNER (rollouts sampled from posterior + self-model)│
                 └──────────────────────────────────────────────────────────────┘
```

### 3.1 The boss model (the generative program)

An explicit-duration HSMM with compositional, law-structured transitions:

```
family f ~ CRP(α_f)                         # boss families share move sets
moves:   K ~ from family f, plus new moves via CRP(α_m)
for each move k:
    c_k ~ Cat(library | f)                  # primitive
    (w_k, a_k, r_k) ~ LogNormal(μ_{c_k}, Σ_{c_k}) with Θ as hyperprior
    cue model ψ_{c_k}: stage-dependent confusion matrix (shared, learned in Θ)
transitions: next-move logits = base_k_prev + Σ_j active_law_j(dist, phase, hp, last2)
             laws are sparse precondition→bias terms (OneLife-style), prior favors few
phase 2:  speed factor s ~ LogNormal, optional added move
```

The same Python function **generates** training bosses and **scores** hypotheses. This is the discipline of a probabilistic program: one model, used both ways.

### 3.2 The fast filter

For each hypothesis Bⁱ, the hidden state (move, stage, ticks) has under 1,000 values, so the **forward algorithm runs exactly** each tick (Lec 8). Hypothesis weights update online from the per-step marginal likelihoods, which is sequential importance sampling over static parameters (Lec 9, 12). Together this is a Rao-Blackwellized particle filter.

If ESS collapses mid-fight (a big surprise, such as a never-seen move), the filter **asks the slow mind to rejuvenate now** instead of waiting for death. This is surprise-gated event segmentation, and it mirrors how people mark event boundaries at prediction-error spikes.

### 3.3 The intuition net (the trained part)

- **Input.** All attempts so far on this boss, as a token sequence. Each token is (cue id, stage hint, distance bin, phase, own action, hit-or-no-hit, Δt). About 200–2,000 tokens per attempt, with attempts separated by a boundary token.
- **Body.** A small transformer encoder (4 layers, d = 128, about 1–2M parameters), plus a **recursive refinement loop** in the style of TRM. The network outputs a proposal, SMC scores it, and a short summary of where the proposal failed goes back in for another pass, 2–4 times. That makes the net an *iterative* proposer that learns to fix its own mistakes.
- **Heads:**
  1. per-token segmentation and move-assignment logits;
  2. per-move primitive class;
  3. per-move timing mean and variance (Gaussian in log space);
  4. transition-count pseudo-observations;
  5. K and family logits;
  6. a law-activation head (which preconditions matter).
- **Use.** The heads define a proposal q_φ(B | log). SMC draws particles from q_φ and weights them by p(B) p(log | B) / q_φ(B | log). It is also used as an independence MH proposal inside rejuvenation. Because of this weighting, a wrong network can slow inference down but cannot bias the final answer, given enough particles.

### 3.4 Acting: planner, reflex and gate

- **Planner.** Model-predictive control. For each of the 7 actions, it runs 64 rollouts of 1–1.5 s, each sampling a hypothesis, a current state and future boss moves, and simulating the player through the self-model. Utility = damage dealt − λ_hp · damage taken − λ_st · stamina spent, plus β · expected information gain during scouting attempts.
- **Reflex.** A 2-layer MLP on about 30 belief features. Trained by DAgger: it imitates the planner on states the reflex itself visits.
- **Gate.** A small regressor predicts the value gap Q_planner − Q_reflex from uncertainty features. The agent plans when that gap exceeds a compute cost c. Sweeping c traces a **speed–accuracy curve**, which is a resource-rational prediction: the agent should "think" more in exactly the situations where people hesitate.

### 3.5 Lifelong memory and transfer

- **Θ.** Empirical Bayes over the posteriors of the 40 training bosses (method of moments on posterior means plus a variance correction).
- **Families.** A CRP over families lets a test boss be recognized as "a variant of family 3" within one attempt. The family's move set then becomes the prior, which works like Bayesian policy reuse but for models instead of policies.
- **Self-model.** For T4 (changed rules), rules R are inferred from the outcomes of the agent's own dodges and blocks. A deliberate test dodge in a safe moment is worth its information, and the planner's information-gain term makes this happen on its own.

---

## 4. Training plan

All "data" is synthetic, from the MiniSouls generator. Nothing is trained on test bosses.

### Stage 0: generator and data (days 1–3)

- Implement MiniSouls in vectorized NumPy (optionally Numba), targeting over 50k ticks/s for batched environments.
- **Prior bosses for amortization.** Sample bosses from the generator prior, *excluding* the T3 held-out primitive and the T4 rule settings, so novelty tests stay honest.
- **Behavior policies for logs.** Logs need varied behavior, or the intuition net only sees one style of play. Use a mix:
  - random policy;
  - scripted "dodge everything" policy;
  - oracle planner with injected noise;
  - later, the agent's own play (wake data).
- **Dataset size.** About 500k–2M simulated attempts at about 600 tokens each, generated on the fly so nothing is stored. Generation is cheap: 2M attempts × 1,800 ticks is about 3.6B ticks, roughly 20 CPU-hours at 50k/s. In practice, generate about 300k attempts and stream them.

### Stage 1: oracle agent (days 3–5), no training

- Give the planner the *true* boss program. Tune utility weights and the rollout horizon on training bosses only.
- **Gate check.** The oracle beats most training bosses. This sets the acting ceiling and separates acting errors from learning errors for the rest of the project.

### Stage 2: intuition net, "sleep" training (days 5–9)

- **Objective (inference compilation).** Minimize E_{B ~ p(B), log ~ p(log | B, policy)} [−log q_φ(B | log)] using the heads' factorized likelihoods: cross-entropy for discrete heads, Gaussian NLL for timings.
- **Curriculum.**
  1. single attempt, 3 moves, first-order transitions;
  2. multi-attempt logs;
  3. laws, phase 2 and families.
- **Recursive refinement training.** Unroll 3 passes, with loss on every pass, and feed back SMC log-weight summaries computed on a small particle set.
- **Model and compute.** About 1.5M parameters, batch of 64 sequences × 1,024 tokens. On a laptop CPU, expect very roughly 30–80k tokens/s training throughput, and several times that on Apple MPS or a consumer GPU. Budget: **4–8 hours** for 300–500M tokens. *This is an estimate to verify in week 1.*
- **Gate check.** Proposal quality on held-out *prior* bosses: segmentation accuracy, timing error, and ESS per particle when used inside SMC. The aim is to cut the particles needed for a given accuracy by at least 5×, compared with prior-only proposals.

### Stage 3: Θ and family learning, "wake" refinement (days 9–11)

- Run the full slow mind on the 40 training bosses with 20 attempts each.
- Fit Θ and the family CRP from the resulting posteriors.
- **Wake-sleep refinement.** Fine-tune q_φ for one short epoch on (log, posterior sample) pairs from the agent's own play. This covers the gap between prior-sampled behavior and real agent behavior.
- **Compute.** 40 bosses × 20 attempts × about 3 s of slow mind per attempt ≈ 40 minutes.

### Stage 4: reflex and gate distillation (days 11–13)

- **DAgger.** Run the reflex, label visited states with planner actions, aggregate, retrain. Five iterations of about 200k labeled states each; the MLP trains in minutes.
- **Gate.** Regress Q_planner − Q_reflex on uncertainty features from the same states.
- **Gate check.** The reflex matches the planner at least 85% of the time on low-entropy states. The gated agent keeps at least 95% of the full planner's win rate while planning on under 30% of ticks.

### Stage 5: test (days 13–16)

- **Protocol.** For each held-out boss in T1–T4, run 20 attempts. φ, Θ and the reflex are frozen; only the posterior changes. Use 5 seeds.
- **Baselines** (trained on the same 40 bosses; see refined-design.md §4):
  - PPO;
  - RecurrentPPO (RL²);
  - per-boss PPO fine-tuning;
  - **Algorithm Distillation** (a small transformer trained on PPO learning histories across training bosses; the in-context RL baseline, if time allows);
  - **Amortized-only**: the intuition net's single best proposal, no SMC, which is the PFN-style baseline;
  - LLM agent with memory (5 bosses);
  - humans.
- **Compute.** PPO baselines at 2–5M steps, about 1–3 CPU-hours each, with 3 seeds. AD needs PPO learning histories (reuse the PPO runs) plus about 2 hours of training.

### Total compute

About **20–35 laptop CPU-hours**, spread over two weeks, mostly running overnight. No cloud GPU is needed. If the laptop has an Apple M-series chip or an NVIDIA GPU, stage 2 shrinks to 1–2 hours.

### Code layout

```
minisouls/        env.py (vectorized), bosses.py (generative program: sample + log_prob), render.py (pygame)
inference/        hsmm_filter.py (exact forward/FFBS), smc.py (resample-move), mcmc.py (Gibbs, split/merge), selfmodel.py
intuition/        tokenizer.py, model.py (recursive transformer), train_sleep.py, train_wake.py
agent/            planner.py, reflex.py, gate.py, lifelong.py (Θ, family CRP), agent.py
baselines/        ppo.py, rppo.py, ad.py, llm_agent.py
experiments/      e1_learning_curves.py … e6_amortization_gap.py, configs/
tests/            sampler correctness against brute-force on tiny bosses, and against pyhsmm
```

**Correctness test worth doing early.** On a tiny boss (2 moves, short durations), compute the exact posterior by enumeration and check that SMC with and without q_φ converges to it. This is also a great report figure for the course.

---

## 5. Defining and measuring adaptive intelligence

Following Chollet's "skill-acquisition efficiency" and Ying et al.'s "world-model induction":

- **Deaths to first win** and the **area under the win-rate curve over 20 attempts**.
- **Induction speed.** Attempts until the posterior predicts the next move's hit time within ±2 ticks with 80% coverage. This measures the *model*, not the play.
- **Novelty slope.** How much each metric worsens from T1 to T2, T3 and T4. A truly adaptive agent has a flat slope.
- **Calibration.** Reliability of hit-time predictions (expected calibration error).
- **Compute-normalized score.** Wins per CPU-second, so the comparison with big models is fair (AI GameStore found frontier models are about 10× slower than people).
- **Human-normalized score.** As in AI GameStore: agent area under the curve divided by the human median, on the same bosses.

---

## 6. Experiments

| | Question | Prediction |
|---|---|---|
| **E1** | Learning curves in deaths: ours vs baselines vs humans | Ours ≈ humans (3–8 deaths); PPO flat; RL² and AD good on T1, falling off on T2–T4 |
| **E2** | Ablations: MAP (N = 1), no Θ, no CRP, no scouting, no laws, oracle | MAP fails on feints and delayed attacks; no-CRP collapses on T3 |
| **E3** | Transfer: T2 combos, T3 new move, T4 new rules, family recognition | Family recognition within one attempt; T4 self-model recovers within 2–3 attempts |
| **E4** | Human-likeness: per-move death rates, delayed-attack effect, transition-order manipulation | Agent with scalar timing noise correlates with human death patterns; both fail on second-order transitions |
| **E5** | Calibration and the speed–accuracy curve of the gate | Gate plans mostly at wind-up starts and after surprises, which is where people hesitate |
| **E6 (headline)** | **Amortization gap under novelty**: amortized-only vs SMC with prior proposals vs SMC with q_φ, across T1→T4 and cue informativeness | Amortized-only ≈ full agent on T1 but degrades sharply on T3/T4; SMC + q_φ keeps accuracy at a fraction of prior-proposal cost. This shows when explicit inference earns its keep |

---

## 7. What is novel here (claims stated carefully)

From the papers I checked, I found no prior work that combines these pieces for a real-time adversary. That is my reading of the sources above, not a full literature search.

1. **Adaptation by inference in real-time combat.** Test-time learning of an opponent with no weight updates, using exact HSMM filtering inside importance-weighted boss hypotheses, set against in-context RL and RL².
2. **"Neural proposes, Bayes disposes," measured under novelty.** An explicit measurement of the amortization gap as structural novelty increases, with the trained proposal made recursive in the style of TRM.
3. **Surprise-gated deliberation.** One signal, the filter's ESS collapse, drives both event segmentation (new move) and the decision to think (rejuvenate or plan). This links event segmentation theory to resource-rational metareasoning.
4. **Delayed attacks as adversarial examples for Bayesian timing.** A cognitive prediction, testable with people, linking interval-timing research to game design.
5. **A self-model for cross-game transfer.** Treating "a different souls-like" as inference over one's own body rules.

---

## 8. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Intuition net training is slower than estimated | Shrink to d = 64, use single-attempt logs, or a GRU. SMC with prior-only proposals still works, just with more particles |
| Laws or families make inference mix poorly | Drop laws (plain first-order transitions); fix families to known labels for training bosses |
| Planner too slow in Python | Numba-compile rollouts; lean on the reflex; plan every 3 ticks |
| AD baseline too costly | Keep RL² as the in-context baseline and mention AD as future work |
| Human study logistics | 5 friends × 30 minutes in the pygame viewer is enough for E4's direction of effect |

## 9. Two-and-a-half-week schedule from today

| Dates | Work | Must-have result |
|---|---|---|
| Nov 25–28 | Stage 0–1: env, generator, exact filter, oracle planner, start PPO overnight | Oracle beats training bosses; PPO comparison running |
| Nov 29–Dec 3 | Stage 2: intuition net (sleep), SMC + MCMC, tiny-boss correctness test | Correctness figure; ESS gain from q_φ |
| Dec 4–7 | Stage 3–4: Θ, families, wake refinement, reflex + gate | Full agent on T1/T2 |
| Dec 8–11 | Stage 5: T1–T4, baselines, ablations, E6 | E1, E2, E6 figures |
| Dec 12+ | Human pilot (E4), write-up | Report |

If time runs short, cut in this order: AD baseline, families, laws, human study, LLM baseline. The core story survives on E1, E2 and E6.

---

## Sources

Cognitive science: Jazayeri & Shadlen 2010 (https://pmc.ncbi.nlm.nih.gov/articles/PMC2916084) · temporal-context review (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9552499/) · Körding & Wolpert 2004 (https://ideas.repec.org/a/nat/nature/v427y2004i6971d10.1038_nature02169.html) · Stafford & Dewar 2014 (https://www.psychologicalscience.org/journals/psychological-science/0956797613511466/) · Listman et al. 2021 (https://www.frontiersin.org/articles/10.3389/fnhum.2021.777779) · Allen, Smith & Tenenbaum 2020 (https://arxiv.org/abs/1907.09620) · Tsividis et al. (https://cbmm.mit.edu/publications/human-learning-atari-0, https://arxiv.org/abs/2107.12544) · Brockbank & Vul 2021 (https://mdpi.com/2073-4336/12/3/70) · Cross et al. 2025 (https://arxiv.org/abs/2508.06503) · Franklin et al. 2020 (https://www.psychologie.uzh.ch/dam/jcr:1f7b969a-3b15-418b-bd27-1a7e785e7570/Franklin_PsychRev_127_2020.pdf) · Kumar et al. 2023 (https://hassonlab.princeton.edu/publications/bayesian-surprise-predicts-human-event-segmentation-story-listening) · Lu et al. 2024 LCNet (https://cocosci.princeton.edu/papers/Lu2024a.pdf) · Zhi-Xuan et al. 2020, 2024 (https://arxiv.org/abs/2006.07532, https://arxiv.org/abs/2407.16770) · value of computation in chess (https://collaborate.princeton.edu/en/publications/time-spent-thinking-in-online-chess-reflects-the-value-of-computa/)

AI: AgentOdyssey (https://arxiv.org/abs/2606.24893) · AI GameStore (https://arxiv.org/abs/2602.17594) · Ying et al. 2025 (https://arxiv.org/abs/2507.12821) · OneLife (https://arxiv.org/abs/2510.12088) · OPINE-World (https://arxiv.org/abs/2607.01531) · WorldLLM (https://arxiv.org/abs/2506.06725) · Colas et al. (https://arxiv.org/abs/2509.00074) · Paige & Wood 2016 (https://arxiv.org/abs/1602.06701) · Perov et al. 2015 (https://arxiv.org/abs/1512.04387) · Reinforced SMC (https://arxiv.org/abs/2510.11711) · Reuter et al. 2025 (https://arxiv.org/abs/2501.16825) · Distribution Transformers (https://arxiv.org/abs/2502.02463) · Algorithm Distillation (https://arxiv.org/abs/2210.14215) · Recurrent AD (https://arxiv.org/abs/2609.35333) · TRM (https://arxiv.org/abs/2510.04871) · Johnson & Willsky 2013 (https://www.jmlr.org/papers/v14/johnson13a.html) · Tang et al. 2020 FightingICE (https://arxiv.org/abs/2003.13949) · DSLE (https://arxiv.org/abs/2608.09902)

Cited from memory and worth checking before the report: Le, Baydin & Wood 2017 (inference compilation); Lieder & Griffiths 2020 (resource-rational analysis); Chollet 2019 is a course reading. Most entries were checked at the abstract level only.
