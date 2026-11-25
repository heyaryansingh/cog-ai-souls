# OneLife (One Life to Learn): Inferring Symbolic World Models for Stochastic Environments from Unguided Exploration

## 1. Citation and reading status
- Zaid Khan, Archiki Prasad, Elias Stengel-Eskin, Jaemin Cho, Mohit Bansal (UNC Chapel Hill, UT Austin, AI2, JHU). arXiv:2510.12088 (v2, 8 Apr 2026). No venue stated in the text. https://arxiv.org/abs/2510.12088 , project page onelife-worldmodel.github.io
- **Reading status (honest):** arXiv was blocked for direct download in my sandbox, so I read the full HTML (all ~109k characters, including Appendix A law code and the start of Appendix B) through a web-fetch tool that returns extracted/condensed text, in three targeted passes. Equations below are as the tool quoted them. I did not see Appendix C (mutator list) or the LLM names; the paper text I saw does not name them.

## 2. Core idea and exact method
**Problem.** Learn an executable, *stochastic* symbolic world model from a single unguided episode in a hostile open world (no reward, no goals, no reset: "one life").

**Representation: a mixture/product of atomic laws.**
- Each law = Python class with `precondition(state, action) -> bool` and `effect(state, action)` that returns a *distribution* (discrete/categorical) over values of a small set of **observables** (state attributes, e.g. `player.inventory.stone`, `zombie[i].position.x`).
- Laws are *atomic*: each touches only a few observables (vs PoE-World experts which predict the entire next state).
- Observables are assumed **conditionally independent** given (s, a).
- Per-observable combination (Eq. 1): p(o = v | s, a; θ) ∝ ∏_{i ∈ I_o(s,a)} φ_{i,o}(v | s, a)^{θ_i}, where I_o(s,a) is the set of laws whose precondition fires *and* that predict observable o.
- Equivalent log-score form (Eq. 4–5): ℓ_o(v) = Σ_i θ_i · log φ_{i,o}(v), normalized by log-softmax over supp(o).

**Inference = weight fitting (point estimate, not a posterior).**
- Loss (Eq. 3): negative log-likelihood of observed transitions, summed over observables then transitions. Optimized with **L-BFGS**.
- **Gradient routing:** a law's weight receives gradient only from transitions where it was active *for that observable*. This is the key fix over PoE-World: irrelevant experts do not dilute the likelihood signal.
- Wrong laws get θ → 0, i.e. pruning by weight. Regularization, initialization and the "no law fires" fallback are not stated in the text I saw.

**Proposal (law synthesis).** One general LLM synthesizer. It diffs consecutive object-oriented states, automatically detects which attributes changed, and asks the LLM to write a precondition/effect law explaining each change. (PoE-World uses 10+ specialized synthesizers; OneLife claims one is enough when the diff is computed for the LLM.)

**Exploration.** An LLM agent (BALROG scaffolding) given only *genre priors* (hostile entities exist, resources can be crafted, etc.), not the specific rules. It keeps a rolling history window and a running summary of its current hypotheses. Unguided: no curiosity bonus or information-gain objective.

**Simulation.** Forward rollouts by sampling each observable from its combined distribution and rebuilding the next state.

## 3. Experiments
- **Environment: Crafter-OO**, a reimplementation of Crafter with explicit object-oriented JSON/Pydantic state (player, objects, 2D material map, daylight, serialized RNG), pure transition function T(s,a) -> s', 17 actions, and programmatic scenario generation covering every achievement in the tech tree (23 scenarios).
- **Evaluation protocol (this is the most stealable part).** For each test transition, build a candidate set of 7–11 next states: the true one plus **distractors from "mutators"** that apply plausible-but-illegal edits (crafting the wrong item, producing an item without prerequisites, teleporting entities, moving the player to an unreachable cell). The model scores each candidate.
  - **State ranking:** Rank@1 (true next state ranked first) and **MRR** (mean 1/rank of the true state).
  - **State fidelity:** sample a predicted next state; **raw edit distance** = number of JSON-Patch ops to turn predicted into true; **normalized edit distance** = raw / number of state elements.
- **Baselines:** Random (uniform over candidates), WorldCoder (single deterministic program), PoE-World (re-implemented with OneLife's explorer and synthesizer, so only the inference differs), and No-inference ablation (all θ = 1).
- **Table 1 (mean of 10 trials):**

| Method | Rank@1 | MRR | Raw ED | Norm ED |
|---|---|---|---|---|
| Random | 8.5% | 0.322 | 121.5 | 0.809 |
| WorldCoder | 0.0% | 0.264 | 27.18 | 0.181 |
| PoE-World inference | 10.8% | 0.351 | 10.63 | 0.071 |
| No inference | 13.0% | 0.429 | **8.54** | **0.057** |
| OneLife | **18.7%** | **0.479** | 8.76 | 0.058 |

  OneLife beats PoE-World on 16/23 scenarios. Weight inference adds +5.7 Rank@1 and +0.05 MRR over no-inference, but does not improve edit distance.
- **Planning check (Table 2):** 3 scenarios (Zombie Fighter, Stone Miner, Sword Maker), each with a reward function and two hand-written plans (5–33 steps), each run 10x in the model and in the real env. Success = model ranks the plans the same way as the true env. All 3 match (e.g., Zombie Fighter true 2.0 vs 1.0, model 2.03 vs 1.67).
- Compute and LLM names not reported in the text I saw.

## 4. Failure modes and limitations
- Reported: Rank@1 only 18.7%; precise generation of complex future states "remains challenging"; only discrete/categorical distributions implemented; WorldCoder's determinism a poor fit.
- Noticed: (a) inference helps discrimination (ranking) but not generation (edit distance), so the weights mostly calibrate *which* law dominates rather than fixing wrong laws; (b) a single point estimate of θ, no posterior or uncertainty over laws, so no principled exploration signal; (c) conditional independence of observables means correlated effects (e.g., x and y of the same move) can produce incoherent samples (the skeleton law allows diagonal moves because axes are independent); (d) planning eval is tiny (3 scenarios, 2 plans each); (e) "unguided" exploration relies on a strong LLM with genre priors; (f) 7 losing scenarios are not analyzed.

## 5. Ideas to steal for Reading the Boss
1. **Distractor-ranking evaluation of the boss model (add to E2/E6).** For held-out fight segments, build candidate futures = true next K ticks of boss behavior plus **mutated** variants: move swapped for a confusable primitive, wind-up shortened/lengthened by 20–40% (delayed attack), transition to a move the boss never uses, phase-2 speed factor applied early, law precondition violated (e.g., grab at long range). Report **Rank@1 and MRR** of the true continuation under the posterior predictive. Cheap, discriminative, and directly comparable across amortized-only vs SMC (the E6 gap) and across T1–T4.
2. **Edit-distance-style fidelity for sampled rollouts:** sample a boss trajectory from the posterior; score it against the true one with a normalized edit distance over (move id, onset tick) tokens (or a tick-tolerant alignment). Complements log-likelihood, which can be dominated by easy idle ticks.
3. **Gradient/credit routing for laws.** In the transition model `logits = base + Σ_j active_law_j`, compute each law's evidence only from transitions where its precondition fired. In SMC terms: when proposing law birth/death in rejuvenation, compute the MH ratio from the subset of transitions the law touches (cheap local likelihood).
4. **Diff-then-explain proposals.** For a proposal over new laws, compute the "surprise diff" first (which transitions the current program predicts worst, and which context features differ) and feed only that to the intuition net's law-activation head. That matches OneLife's single general synthesizer working from auto-detected changes.
5. **Plan-ranking agreement as a world-model metric:** for a handful of fixed dodge/punish scripts, does the learned posterior rank them in the same order as the true boss? This is a decision-relevant metric (rank correlation of plan values, model vs. truth) that fits E1/E6.
6. **Scenario generators covering every mechanic.** Like Crafter-OO's per-achievement scenarios, write MiniSouls scenario starts that force each law (low HP phase, long range, after two combos) so evaluation covers rare preconditions.

## 6. Structures and models of thought
- **Laws as sparse precondition -> distribution programs, combined by weighted product.** This is the right granularity for boss "rules" (e.g., "if player is behind and boss is in combo, tail swipe"). The product lets several partial laws jointly sharpen one prediction, and weights act as soft structure selection. Reading the Boss already plans OneLife-style laws; the upgrade is to treat law weights as latent variables with a sparsity prior inside SMC rather than an L-BFGS point estimate. That gives a posterior over laws, which OneLife lacks and which the planner and gate need.
- **Factored observables.** Predicting each observable separately keeps hypotheses small and composable. In the HSMM, keep move identity, timing and cue confusion as separate factors with separate laws.
- **Unguided exploration with genre priors** is analogous to the agent's built-in prior over souls-like bosses (wind-up, active, recovery primitives). A coarse prior plus a hypothesis summary in working memory is enough to explore usefully, but adding information gain (as the plan does for scouting) should beat it.
