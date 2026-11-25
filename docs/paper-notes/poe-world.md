# PoE-World: Compositional World Modeling with Products of Programmatic Experts

## 1. Citation and reading status
- Wasu Top Piriyakulkij, Yichao Liang, Hao Tang, Adrian Weller, Marta Kryven, Kevin Ellis (Cornell; Cambridge; Alan Turing Institute; Dalhousie). arXiv:2505.10819 (v4, 19 Nov 2025). 2025; no venue in the text I saw. https://arxiv.org/abs/2505.10819
- **Reading status (honest):** Direct arXiv download was blocked in my sandbox. I read the HTML (truncated at Table 2) and then the full PDF, including Section 6 and the appendices (hard constraints, planner, synthesizers, hyperparameters, compute), through a web-fetch tool that returns extracted/condensed text in targeted passes.

## 2. Core idea and exact method
**Representation.** A world model is an exponentially weighted **product of many small programmatic experts** (hundreds of LLM-written Python functions; total programs of 4,000+ lines):
- p_θ(o_{t+1} | o_{1:t}, a_{1:t}) ∝ ∏_i p_i^expert(o_{t+1} | o_{1:t}, a_{1:t})^{θ_i}.
- Object-centric symbolic observations (OCAtari object detector: bounding boxes, velocities, visibility). Features f are conditionally independent given history, with a separate normalizer per feature: p_θ = ∏_f (1/Z_f) ∏_i p_i(o^f | ·)^{θ_i}.
- **Full history** instead of a latent state (keeps experts modular; adding a latent would change every expert's inputs).
- **Hard constraints** c_j: O -> {0, 1} multiply in as a mask: p ∝ ∏_i p_i^{θ_i} · 1[∨_j c_j(o_{t+1})] (Eq. 3). Disjunction because game physics is imperfect. MR examples: player center_x = rope center_x when touching rope; player bottom = platform top when touching; same for conveyor and ladder. Learned constraints pruned if they contradict data or explain < 1% of it. Constraints are not part of weight optimization.
- Multi-step experts approximated as products of next-step experts.

**Probabilistic reading of deterministic code.** LLM writes deterministic Python (more in-distribution for LLMs than probabilistic programs). An attribute set by an expert becomes a peaked distribution (RandomValues for single peak, SeqValues for multi-step) plus noise so other values have non-zero probability; unset attributes are uniform. So an expert whose if-condition is false is uninformative.

**Learning loop.** (1) synthesize experts from small batches (10 transitions); (2) fit θ by maximum likelihood with **L-BFGS** (PyTorch, strong Wolfe line search, lr 1, 4 epochs, full batch, **L1 penalty weight 1**; L-BFGS beat Adam/SGD); (3) **prune experts with θ < δ = 0.01**; (4) repeat when new data arrives.

**Synthesizers.** 10 LLM synthesis modules, each operating on one object type at a time, targeting different aspects (passive motion, action effects, interactions, creation/deletion). ActionSynthesizer: from one transition, ask for 4 candidate causal explanations, then turn each into a function. LLM: gpt-4o-2024-08-06.

**Usage.** (a) As a simulator to pretrain PPO, then fine-tune in the real game. (b) **Hierarchical planner (TAMP-like):** abstract states = object-contact relations; abstract graph built by running the low-level planner in the model between all node pairs (edge if any low-level plan succeeds); BFS gives subgoal sequence; low level = modified MCTS (Manhattan heuristic, max-backup, 4,000 iters for MR) with greedy fallback (best action chunk of length 8, backtrack on death), action chunks of 1/4/8 frames. On an edge failure: delete edge, update world model, re-search; rebuild graph if no path.

## 3. Experiments
- **Environments:** Atari Pong and Montezuma's Revenge (MR) via ALE + OCAtari (patched per game). **Alt variants** for zero-shot generalization: Pong-Alt (3 balls, 3 enemies), MR-Alt (more enemies and ladders, Kangaroo-like map; goal = get key). No demos for Alt; models reused from base games.
- **Data:** < 1,000 frames of non-expert demonstration per game (frameskip 3); the MR demo never gets reward.
- **Metrics:** game score (Pong: player − enemy points at 21; MR: 100 iff key collected); **next-observation prediction accuracy** (whole next frame exactly right, Table 2) and **object-attribute accuracy** (per moving object attribute, Table 3), on training data and on 1,000 random-play frames.
- **Table 1 (scores):** PoE-World + Planner: Pong −12.3, Pong-Alt −13.7, MR 100, MR-Alt 100. WorldCoder + Planner: −17.0, −19.0, 0, 0. PPO @100k: −21.0, −18.7, 0, 0. ReAct (LLM policy): −20.0, −20.7, 0, 0. PPO @20M: 17.0, 1.3, 0, 0. PoE-World agent trained ≤ 3k steps.
- **Table 2 (next-obs accuracy, Train Pong / Train MR / Test Pong / Pong-Alt / MR / MR-Alt):** PoE-World 0.51 / 0.75 / 0.06 / 0.03 / 0.31 / 0.43; WorldCoder 0.33 / 0.36 / 0.06 / 0.01 / 0.10 / 0.12.
- **Table 3 (object-attribute accuracy):** PoE-World 0.88 / 0.93 / 0.66 / 0.76 / 0.76 / 0.83; WorldCoder 0.82 / 0.75 / 0.66 / 0.67 / 0.58 / 0.65.
- **Table 4 (hard-constraint ablation):** planning successes MR 5/9 with vs 2/9 without; MR-Alt 4/9 vs 2/9; next-obs accuracy essentially unchanged (0.31 vs 0.30; 0.43 vs 0.43). Constraints act as "damage control" for long-horizon rollouts.
- **Policy pretraining:** PPO pretrained in the PoE-World model beats random after ~200k real steps vs ~1M for vanilla PPO; same asymptote.
- **Compute:** world modeling on 4 CPUs / 64 GB, ≈ 8 h per run including LLM latency, ≤ ~$20 OpenAI credit per run; planning up to 24 h, abstract-graph construction parallelized on ~100 CPUs.

## 4. Failure modes and limitations
- Reported: needs symbolic object input (no pixels; OCAtari needed manual patching); no exploration, decision-making or reward learning (learns from a fixed demo); planner not the focus, results vary across runs (5/9, 4/9); predictions over-approximate (fuzzy "falling" trajectories); random-frame Pong accuracy is low for all methods (random play rarely hits the ball); WorldCoder hallucinates mechanics (bullet firing, flying).
- Noticed: (a) every expert predicts the whole next observation, so irrelevant experts add uniform noise and weakly-identified weights (OneLife's critique, and OneLife's reimplementation of PoE-World inference scores worse: Rank@1 10.8% vs 18.7%); (b) θ is a point estimate, no uncertainty over which experts are true; (c) the conditional-independence and multi-step product approximations ignore correlations; (d) the key metric gap: next-frame exact accuracy is low even when planning succeeds, so planning-relevant accuracy matters more than raw accuracy.

## 5. Ideas to steal for Reading the Boss
1. **Hard constraints as a mask on predictions.** Add hard, known-to-be-true constraints to the boss generative program and to rollouts in the planner: e.g., an attack's active frames cannot start before its wind-up ends; a move cannot start during the previous move's recovery unless it is a declared combo/cancel; boss cannot hit beyond the primitive's reach. Their ablation (2/9 -> 5/9 planning success with unchanged one-step accuracy) predicts that constraints matter most for the **planner's long rollouts**, not for filter likelihood. Test as an E2 ablation ("no constraints") measuring planner success, not just log-likelihood.
2. **L1-regularized weights + prune at δ.** For law weights in the transition model, use a sparsity prior (Laplace ≈ L1) and drop laws whose posterior mean weight is below a threshold after rejuvenation, to keep particles small. Use their δ = 0.01 / L1 = 1 as defaults if a point-estimate fallback is needed.
3. **Two accuracy metrics plus a task metric.** Report (i) exact next-move-and-onset accuracy, (ii) per-attribute accuracy (move id, onset tick within ±k, reach), and (iii) planner success. PoE-World shows (i) can stay low while (iii) is fine; report all three for E2/E6.
4. **Zero-shot "Alt" variants for transfer (E3).** Their Alt games = same mechanics, more objects/new layout, no extra data. MiniSouls analogue: same boss family with more moves in a combo, two bosses at once, or new arena width, evaluated with zero additional deaths. Compositional laws should transfer; a monolithic net should not.
5. **Pretrain-in-model then fine-tune** as a baseline-plus: pretrain the reflex policy entirely inside sampled boss programs from the posterior and measure real-fight steps saved (they get ~5x). Fits Stage 4.
6. **Abstract contact-graph planning** is overkill for 1D MiniSouls, but the **edge-failure replanning** rule (remove failed edge, update model, re-search) is a good template for when a planned punish window turns out wrong.

## 6. Structures and models of thought
- **Compositional causal laws combined by a product of experts:** each law is a small, interpretable, separately testable piece of knowledge; the product composes them to extrapolate to new combinations (more objects, new layouts). This is the strongest argument for the boss program being a set of sparse laws rather than one transition matrix: new bosses reuse laws from the library.
- **Soft experts + hard constraints:** probabilistic soft knowledge for what varies, symbolic hard constraints for what cannot happen; the mix keeps imagined futures physically valid, which is what a planner needs.
- **Data efficiency from priors in code:** < 1,000 frames and ≤ 3k steps vs PPO's 20M — the human-like learning-from-few-deaths target is plausible only with this kind of structured hypothesis space.
