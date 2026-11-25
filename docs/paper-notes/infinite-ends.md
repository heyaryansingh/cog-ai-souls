# Infinite Ends from Finite Samples: Open-Ended Goal Inference as Top-Down Bayesian Filtering of Bottom-Up Proposals

## 1. Citation and coverage
Tan Zhi-Xuan, Gloria Kang, Vikash Mansinghka, Joshua B. Tenenbaum (MIT). 2024. arXiv 2407.16770 (CogSci 2024 proceedings, as far as I know; the fetched page does not name the venue). https://arxiv.org/abs/2407.16770 . Data/code: osf.io/bygwm.
Coverage: full arXiv HTML (main text + appendix tables A1, A2) processed through a web-fetch extraction tool (direct arXiv download blocked by proxy). I read a detailed extraction of the full text, not every sentence verbatim.

## 2. Core idea and method
- Human goal spaces are effectively unbounded, yet people quickly propose plausible goals and drop ones the actions contradict. Model: **open-ended SIPS**, a particle filter where **bottom-up samplers propose goals** from learned subgoal co-occurrence statistics and **top-down Bayesian inverse planning reweights** them.
- Generative model: g ~ P(g); online policy pi_t ~ P(pi_t | s_{t-1}, pi_{t-1}, g); a_t ~ Boltzmann: P(a|s,pi) proportional to exp(-beta * Qhat(s,a)), Qhat = estimated cost-to-go from **real-time heuristic search (RTHS)** with compute budget B. Exact inference over G goals costs O(G B T).
- **Bottom-up proposal Q(g | s_t, a_t)**: given the just-achieved subgoal (a tower of letters), sample a complementary completion from a character-level 5-gram model fit to tempered word frequencies. Retrieval treated as ~constant time. No LLM used (LLM distillation mentioned as future work). Strategies: ANY-TOWER, LAST-TOWER, NEXT-TOWER, LAST-AND-NEXT (default; falls back to NEXT-TOWER if the last action was not stacking or the tower is not word-like).
- **Algorithm 1 (per step t)**: (1) propose N new goals from Q(g | s_t, a_t); (2) simulate their policies over the history; (3) weight new particles by P(g, pi_{1:t}, s_{0:t}, a_{1:t}) / Q(g | s_t, a_t); (4) update existing particles' policies, multiply weights by P(a_t | s_{t-1}, pi_t) (this filters out irrational goals); (5) resample down to N, **coalesce duplicate goals by summing weights**. Fresh proposals each step act as rejuvenation. Runtime O(T^2) worst case (re-simulating history for new goals), ~O(T) in practice; a variant forgets old observations to guarantee O(T). Because Q has auxiliary randomness, an **unbiased density estimator** is used for the weight denominator.

## 3. Experiments
- Task: Block Words with an **open goal space** (any English word 3-8 letters spellable from blocks; 145-807 candidates per scenario vs 5-20 in prior work).
- 16 scenarios in 4 conditions: Bottom-Up Friendly; Irrational Alternatives (bottom-up guesses contradicted by actions); Garden Paths (early actions suggest a merely instrumental word); Uncommon Words.
- Humans: 100 Prolific participants, each saw 8/16 scenarios, 50 per scenario; free-response guess lists edited at judgment points, each guess weight 1/n. Exclusions: 36/800 never updated, 139/800 only added.
- Models: open-ended SIPS (N in {2,5,10,20,50}), exact enumerative inference, proposal-only. Fit by grid search on IoU (Jaccard) with humans. Final: beta = 1.0, B = 100, Delta t = 2, BFS search, prior temperature T_w = 4, per-char termination eps = 0.05, Q = LAST-AND-NEXT, **N = 2 best fit**.
- Results: human-human IoU 0.44 (ceiling); open-ended SIPS 0.33-0.36 (N = 2 best, 0.354); exact inference 0.31; proposal-only 0.30-0.32. **Two particles fit humans better than exact inference**, including humans' between-subject variance.
- Accuracy/runtime (Table A2): exact BFS 0.153 at 7.31 s/action; exact A* 0.186 at 37.95 s; SIPS BFS N = 2: 0.177 at 0.08 s; SIPS A* N = 50: 0.203 at 4.73 s; proposal-only ~0.14-0.15 at <0.003 s. SIPS matches or beats exact at ~1-10% of the cost. **Accuracy rises more with planning budget than with particle count.**
- Resource-rational analysis: SIPS beats exact on reward net of compute cost; pure bottom-up guessing is sometimes more resource-rational when reweighting is expensive. Humans beat proposal-only in Irrational Alternatives (evidence of top-down filtering).
- Qualitative: humans and SIPS both rule out irrational goals; both show sticky garden paths; proposal-only shows recency bias. Rare words under Q ("chump") are missed by humans and SIPS until late, whereas exact inference finds them earlier: **low proposal probability behaves differently from low prior probability**.

## 4. Failure modes and limitations
Stated: human-model gap (0.44 vs 0.35); corpus prior needs tempering; humans stickier than SIPS on garden paths and possibly forgetful; single domain; hand-designed proposals; parameters fit on the same data (no held-out test); 50 per scenario.
Noticed: the proposal is a fixed n-gram, not learned from the task; IoU on guess sets is a coarse metric; Boltzmann action noise is used here (unlike SIPS 2020's planning noise), so the "nearer goal" artifact may return; coalescing duplicates is essential when the proposal is peaky, otherwise N = 2 particles collapse to one hypothesis.

## 5. Ideas to steal
- **Algorithm 1 is the exact template for intuition-net + SMC**: at every surprise point (new move, death), inject N fresh particles from q_phi(B | log), weight them by p(B) p(log|B) / q_phi(B|log), merge with surviving particles, resample, and **coalesce identical boss programs by summing weights** (hash canonicalized programs). This is a cheaper alternative to MH rejuvenation and keeps a mixture of old and new hypotheses.
- If the intuition net uses stochastic internals (dropout, sampled recursion passes), the q density needed for weights must be exact or estimated unbiasedly; design the output heads so q(B|log) is tractable (autoregressive over program slots) rather than relying on auxiliary randomness.
- **"Proposal-rare vs prior-rare" as an E6 probe**: construct novel bosses whose true program has decent prior mass but low q_phi mass (out of training distribution). Prediction: amortized-only and SMC+q_phi both miss it for a while, exact/prior-proposal SMC finds it earlier. That is a concrete, quantifiable signature of the amortization gap. Measure "deaths until true program enters the particle set."
- **Small-N as a human-likeness knob**: fit N (particles) to human data in E4; this paper finds N = 2 best. Report agent behaviour at N in {1,2,5,20,100}; the human-like setting may be small N, while the performance setting is large N. Same for budget B of the planner.
- **Resource-rational scoring**: reward minus lambda * compute (s/action or particle-evaluations). Use this to justify the metareasoning gate quantitatively and to show when "intuition only" is the rational choice (their proposal-only result).
- **Garden-path bosses**: openers that mimic a known family before diverging (feints at the family level). Prediction: sticky misattribution for both humans and small-N agent; good test for family-CRP + rejuvenation.
- **Proposals conditioned on the latest subgoal** (LAST-AND-NEXT): condition q_phi heavily on the most recent move segment; an ablation comparing "whole log" vs "last segment" conditioning mirrors their strategy table.
- Accuracy tracks planning budget more than particles: allocate laptop compute to better per-particle likelihood (exact HSMM filter) before more particles.

## 6. Structures and models of thought
- **Generate-and-test with learned co-occurrence statistics** (bottom-up) + **inverse planning likelihood** (top-down): the intuition net is literally the bottom-up sampler; the HSMM likelihood is the top-down filter.
- **Open-ended hypothesis spaces handled by sampling, not enumeration**, which is what a CRP over boss families / move primitives needs.
- **Few-particle, resource-rational inference** as a model of humans; supports a metacognitive controller that trades particles and planning budget against time.
- **Forgetting** (bounded history) as an optional, human-like memory compression mechanism that keeps cost O(T).
