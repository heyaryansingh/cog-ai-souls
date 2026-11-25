# WorldLLM: Improving LLMs' World Modeling Using Curiosity-Driven Theory-Making

## 1. Citation and reading status
- Guillaume Levy, Cédric Colas, Pierre-Yves Oudeyer, Thomas Carta, Clément Romac (Inria Flowers, Univ. Bordeaux; MIT; Hugging Face). arXiv:2506.06725, 2025. Venue not stated in the text. Code: github.com/flowersteam/WorldLLM. https://arxiv.org/abs/2506.06725
- **Reading status (honest):** Direct arXiv download was blocked in my sandbox. I read the full HTML (main text and appendices A–D) through a web-fetch tool that returns extracted/condensed text. Eq. 1 (global objective) was missing from the extraction. Everything else below is from that pass.

## 2. Core idea and exact method
**Goal.** Make a frozen LLM a better forward model P(s' | s, a) in a text environment by putting a small set of natural-language **hypotheses H** in its prompt. H is found by Bayesian search, and the data used is collected by a curiosity-driven RL agent. No reward (reward-free MDP); budget of T experiences.

**Three components:**
1. **Statistician** P^St(s' | s, a, H): LLM scoring next-state strings given hypotheses (likelihood = token log-prob of s').
2. **Scientist** P^Sc(Ĥ | D_t, H_t): LLM that proposes new hypothesis sets. Prompt holds recent trajectories, the current accepted hypotheses, and the **worst-predicted trajectories**. **Uniform prior** over hypothesis space.
3. **Experimenter**: collects data. Oracles (O-Random, O-Ideal = fixed hand-designed mix, O-Curriculum = staged plants -> small herbivores -> big herbivores at iters 133/266, O-Hardest = only hardest transitions) or RL agents.

**Inference (Algorithm 1): single-chain Metropolis(-Hastings-like) search.**
- Each iteration: collect D_t (150 transitions to the Scientist); current_score = Σ_{D_t} log P^St(s' | s, a, H_t).
- n_steps = 5 proposals Ĥ ~ P^Sc(· | D_t, Ĥ); accept if log u < score(Ĥ) − current_score. (No proposal-ratio correction, so it is really stochastic hill-climbing with Metropolis acceptance.) **One particle.**
- Likelihood is evaluated only on the *current batch* D_t, so the chain forgets older data.
- Acceptance rate ≈ 20% early, ≈ 2% late.

**Curiosity-driven Experimenters (RL, Masked PPO, MLP 2×64 on symbolic state, γ = 0.99, lr = 5e-4, 20 epochs; 3,600 transitions per iteration for RL, last 150 to the Scientist):**
- **RL-LogP:** r = −log P^St(s' | s, a, H_t) (prediction error / surprise).
- **RL-ALP:** r = |log P^St(s' | s, a, H_{t−1}) − log P^St(s' | s, a, H_t)| (absolute learning progress, Eq. 2).
- **RL-ALPEXP:** ALP computed per transition type τ and smoothed: r_τ = α·m_{f(τ)} + (1 − α)·r_alp, α = 0.9.

## 3. Experiments
- **Environment:** Playground-Text: textual gridless world, actions Go to / Grasp / Release, objects water, plants (5), small herbivores (3), big herbivores (3); growing requires feeding water to plants and plants to herbivores (a tech tree). 6 transition types: Standing, Holding 1, Holding 2, Grow Plant, Grow S.Herbivore, Grow B.Herbivore. Next-state text describes only the change.
- **LLM:** Phi-3-mini-4k-Instruct, 4-bit, for both Statistician and Scientist. 8 seeds, 400 iterations.
- **Test set:** 168 held-out transitions (120 Standing, 20 Holding 1, 7 Holding 2, 12 Grow Plant, 6 Grow S.H., 3 Grow B.H.).
- **Metrics:**
  - Per-type mean test log-likelihood under the Statistician over iterations.
  - **Normalized area (Eq. 3):** area = 1 − ∫_0^T [log P(D_test | H_t) / (log P(D_test | ∅) · T)] dt. 1 = best, 0 = no better than no hypotheses, < 0 = hypotheses hurt. It measures how fast and how much the hypotheses help over the whole learning run (an anytime metric).
  - **Top-3 accuracy with constrained decoding** over the set of valid next states.
- **Table 1 (normalized area, selected):** O-Ideal: Standing 0.92, Holding1 0.72, Holding2 0.50, GrowPlant 0.81, GrowSH 0.77, GrowBH 0.74. **RL-LogP:** 0.93, 0.52, 0.53, 0.62, 0.61, 0.60. RL-ALPEXP: 0.95, 0.44, 0.40, 0.42, 0.52, 0.49. **RL-ALP:** 0.98, 0.18, 0.17, **−0.07**, 0.16, 0.08. O-Random: 0.97, 0.45, 0.40, 0.34, 0.44, 0.40. O-Hardest: 0.74, 0.48, 0.39, 0.52, 0.57, 0.56.
- **Table 2 (top-3):** No hypotheses: Standing 0.82, Holding1 1.00, all growth types 0.00. O-Ideal: 1.00/0.94/–/0.94/0.25/0.46. RL-LogP: 1.00/0.67/0.34/0.74/0.02/0.29. RL-ALP ≈ 0 on growth.
- **vs fine-tuning:** F-LogP (causal-LM fine-tune on RL-LogP's data) gets perfect top-3 in-distribution, but in a **reworded** environment ("The X is beneath you") F-LogP collapses (Standing 0.88, Grow Plant 0.07) and WorldLLM falls back to roughly no-hypothesis level, because its hypotheses contain concrete examples rather than abstract rules.
- RL-LogP's data distribution ends up close to O-Hardest; RL-ALP collects mostly Standing.
- Compute: 4-bit Phi-3-mini, 15 parallel envs, HPC (IDRIS/GENCI); no GPU-hours stated.

## 4. Failure modes and limitations
- Reported: simple environment where "harder transitions are later in the tech tree" helps surprise-seeking; only one small LLM, which writes example-like rather than abstract hypotheses; **single particle -> stuck in seed-specific local optima** (t-SNE/inertia analysis); **likelihood only on recent batch -> forgetting** (they suggest replay buffer or particle filter); no length/simplicity prior; ALP reward is mostly zero because accepted improvements are rare, so the RL agent gets no signal.
- Noticed: (a) no MH proposal correction, so not a valid posterior sampler; (b) hypotheses don't transfer to surface changes, so "theory" is partly memorization; (c) evaluation is at the transition level only, no planning test; (d) O-Hardest underperforms, showing pure hardest-first data lacks diversity.

## 5. Ideas to steal for Reading the Boss
1. **Normalized area metric for learning curves (E1/E6).** For each boss, area = 1 − (1/T)·Σ_t [log p(D_test | posterior after t deaths) / log p(D_test | prior)]. One number per (method, boss, novelty level) summarizing how fast the posterior improves over the prior, comparable across T1–T4, and it can go negative (inference hurt), which is exactly what we want to detect in the amortized-only arm under novelty.
2. **Per-transition-type breakdown.** Report predictive log-likelihood / top-k move accuracy separately by move type (common idle/approach vs rare combo enders, delayed attacks, phase-2 moves). WorldLLM shows aggregate metrics hide failure on rare, late-tech-tree transitions; souls bosses have the same long tail.
3. **Scouting reward choice: use surprise, not learning progress, or smooth LP per type.** Their clean negative result: raw ALP fails when hypothesis updates are rare (sparse reward); −log p (surprise) works; per-type smoothed ALP is in between. For our planner's β·EIG term during scouting, expect the same sparsity issue if EIG is computed only after accepted posterior changes; use per-step predictive surprise or EIG under the current particle set instead.
4. **Proposal context = worst-predicted segments.** Feed the intuition net (or rejuvenation proposals) the segments of the fight log with the lowest likelihood under the current particles, plus the current best hypothesis. A cheap way to focus proposals; also matches their Scientist prompt.
5. **Avoid their two failure modes explicitly:** score proposals on the *whole* fight log (all deaths), not the latest attempt (forgetting), and use many particles with resampling (local optima). These are exactly what resample-move SMC fixes — cite WorldLLM as motivation for E2's "MAP (N = 1)" ablation.
6. **Surface-change generalization test.** Their rewording test is a model for a "same boss, re-skinned cues" condition: permute cue ids/animation tokens while keeping dynamics. Tests whether the intuition net learned abstract structure or memorized cue tokens.

## 6. Structures and models of thought
- **Theory as a prompt-level latent:** hypotheses are a latent variable that conditions a fixed predictive model, inferred by propose-score-accept. This is the same "neural proposes, Bayes disposes" pattern as the intuition net + SMC, but with one particle and no proposal correction; it shows what breaks without proper SMC.
- **Scientist / Statistician / Experimenter split** maps onto intuition net (proposer), fast filter (scorer), and planner in scouting mode (experimenter). Keeping them separate lets each be ablated.
- **Curiosity as surprise-seeking** is the cheapest information-seeking signal; learning progress needs a dense signal to work.
