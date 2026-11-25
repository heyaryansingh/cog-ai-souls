# SIPS: Online Bayesian Goal Inference for Boundedly-Rational Planning Agents

## 1. Citation and coverage
Tan Zhi-Xuan, Jordyn L. Mann, Tom Silver, Joshua B. Tenenbaum, Vikash K. Mansinghka. "Online Bayesian Goal Inference for Boundedly-Rational Planning Agents." NeurIPS 2020. https://arxiv.org/abs/2006.07532 . Code: github.com/ztangent/Plinf.jl (branch neurips-2020-experiments).
Coverage: full arXiv HTML text (main + appendix tables) was processed, but through a web-fetch extraction tool (direct arXiv download was blocked by the proxy), so I read a detailed structured extraction of the full text rather than every sentence verbatim. Numbers below come from that extraction; spot-check before quoting in the report.

## 2. Core idea and method
- Observed agents are modeled not as noisy-optimal (Boltzmann) actors but as **boundedly-rational planners that interleave partial search with execution**. Randomness lives in the *planning process*, not in action selection.
- Generative program (Gen, PDDL domains): goal g ~ P(g); plan p_t ~ P(p_t | s_t, p_{t-1}, g); action a_t ~ P(a_t | s_t, p_t); state transition; observation noise (Gaussian sigma = 0.25 on numeric fluents, Bernoulli flip p = 0.05 on booleans).
- **Replanning trigger**: the agent replans only when its current partial plan is exhausted or the current state was not anticipated by the plan. Otherwise it just executes, at zero cost.
- **Search budget** eta (max nodes expanded) ~ NegativeBinomial(r, q); r > 1 gives a mode at medium budgets. Base planner = A* with stochastic expansion: successor chosen with prob proportional to exp(-f(s,g)/gamma), f = g-cost + heuristic, gamma = search noise; returns the plan to the most recently selected node (so partial plans can head the wrong way).
- **SIPS (Sequential Inverse Plan Search)** = particle filter over (goal, plan-state) traces:
  - init k particles from goal prior (empty plans, equal weights);
  - per observation: extend each particle (transition, replan only if needed, choose action), reweight by observation likelihood;
  - resample when ESS/k < c (c = 1/4);
  - optional **rejuvenation** via two MH kernels (applied w.p. p_g = 0.25 in qualitative runs): (a) **data-driven goal proposal**: softmax over heuristic distance from the latest observed state, which lets pruned goals re-enter; (b) **error-driven replanning**: pick a time near where hypothesized and observed trajectories diverge and resample the plan from there. Acceptance via Gen's involutive MCMC.
  - **Incremental computation**: only particles whose plan does not cover the current state call the planner, so cost per step is small.
- Quantitative settings: 10 particles per goal (50 for Block Words), gamma = 0.1, r = 2, q = 0.95 (mean budget ~38 nodes), no rejuvenation. Heuristics: Manhattan (grid), h_add (Block Words, Intrusion Detection).

## 3. Experiments
Domains: Taxi (3 goals, ~125 states), Doors-Keys-Gems (3 goals, ~1e5), Block Words (5 goals, ~1e5), Intrusion Detection (20 goals, ~1e30).
Baselines: BIRL-U (async value iteration, uniform state sampling, 250k iters; 10k Taxi), BIRL-O (oracle state sampling biased toward test trajectories; not online-usable), PRP (plan recognition as planning, appendix).
Metrics: P(g_true | o) and top-1 accuracy at trajectory quartiles Q1/Q2/Q3; runtime split into start-up cost C0, marginal cost per step MC, average cost per step AC; states visited N.

Selected Table 1a (P(g_true) Q1/Q2/Q3; AC seconds):
| Domain | SIPS | BIRL-O | BIRL-U |
|---|---|---|---|
| Doors-Keys-Gems | 0.37/0.51/0.61; AC 0.86 | 0.37/0.36/0.42; AC 7.0 | 0.33 flat; AC 154 |
| Block Words | 0.47/0.83/0.90; AC 4.15 | 0.20/0.29/0.45; AC 2.1 | ~0.20; AC 63.6 |
| Intrusion Det. | 0.56/0.87/0.87; AC 28 | 0.09/0.24/0.53; AC 6.0 | 0.05; AC 1069 |
SIPS visits 1.4k-13k states vs 10k-250k for BIRL. PRP is most accurate in the appendix but up to ~30x more expensive per step.

Human comparison: Doors-Keys-Gems, 10 trajectories (6 suboptimal or failed, some irreversible failures). Participants rated goal probabilities every 6 timesteps (pick most likely gem, then sliders, normalized). N = 8 in main text (5 in supplement; inconsistency in source). Correlation with humans: SIPS r = 0.89 across all trajectories vs BIRL r = 0.51; on the showcase trajectory r = 0.99 vs 0.80 (BIRL stays on the wrong gem after backtracking).

Robustness (Table 1b, top-1 at Q3, DKG / Block Words): matched 0.73/0.87; q = 0.80 drops to 0.53/0.70 (more random data hurts); Boltzmann agents 0.58/0.82; optimal agents 0.80/0.80; real human plans 0.79/0.73; heuristic mismatch can help (maze distance 0.90) or hurt badly (goal count 0.43).
Compute: laptop CPU (1.9 GHz i7, 16 GB, Julia).

## 4. Failure modes and limitations
Stated: finite goal sets only; goals flat (no hierarchy/instrumental subgoals); deterministic, discrete domains; some domains lack irreversible failures, understating SIPS's advantage.
Noticed: human study tiny (N = 5-8, 10 trajectories); model fit is only "matched vs mismatched" hyperparameters, no inference over the observed agent's own bounded-rationality parameters; heuristic choice matters as much as particle count (0.43 vs 0.90); no rejuvenation in quantitative runs, so the value of the MH kernels is not quantified; Boltzmann-noise baselines show "favor nearer goal" artifacts, which is a warning about any softmax-policy likelihood.

## 5. Ideas to steal
- **Put the boss's noise in its program, not in a softmax over actions.** The boss is a scripted program, but bosses with reactive laws ("if player heals, punish") have the same issue SIPS fixes: a softmax likelihood over moves will spuriously favor "nearer" explanations. Model stochasticity as explicit latent choices (which branch, which delay) so an off-pattern move is explained by a latent, not by noise. Keep a small epsilon "glitch" likelihood only as a floor.
- **Replan-only-when-surprised = the filter's incremental computation.** Per hypothesis, only recompute the HSMM forward pass / planner rollouts when the observed move falls outside what that hypothesis predicted (SIPS's "plan does not cover state"). Report MC vs C0 vs AC cost per step exactly as Table 1a does, for E5/E6 speed-accuracy curves.
- **Two rejuvenation kernels, ported:** (a) data-driven proposal from the *latest* observation (SIPS's heuristic-distance goal proposal) = intuition-net proposal conditioned on the last k moves, used as independence MH so pruned boss programs can come back; (b) **error-driven rejuvenation**: find the first timestep where the particle's predicted move/timing diverges from the log and resample the program fragment (the conditional law or duration parameter) active there. This is cheaper and more targeted than whole-program resampling and fits split/merge.
- **ESS/k < 1/4 resampling threshold** as default; log ESS per step as the surprise signal that also drives the metareasoning gate.
- **Quartile metrics**: report P(true boss program component) and top-1 at Q1/Q2/Q3 of each fight and across deaths. Gives a clean E1/E6 learning curve that is comparable to SIPS.
- **Robustness grid (Table 1b) as an E6 sub-experiment**: test the agent when the boss is "more random than assumed" (lower q-analog = higher timing variance) and when it is more deterministic. Prediction from SIPS: degrades under extra randomness, improves under more determinism.
- **Human comparison protocol**: pause the replay every few moves and ask "what will the boss do next / which pattern is this?" with slider probabilities; correlate with the agent's posterior predictive (SIPS got r = 0.89 vs 0.51 for BIRL). Cheap to run with 5 friends and directly supports E4.
- Include irreversible-failure cases (e.g., a boss phase that only triggers after a mistake) where non-planning baselines break.

## 6. Structures and models of thought
- **Agent as a program with bounded resources** (budgeted search, replanning triggers): the same template fits a boss AI with reaction delays and limited lookahead, and fits the *agent's own* self-model.
- **SMC with data-driven + error-driven MH rejuvenation**: the inference scheme the slow mind already uses; SIPS shows rejuvenation kernels should be targeted at the divergence point.
- **Incremental, surprise-triggered recomputation**: a cognitive mechanism (keep executing until the world surprises you) that unifies the filter's cost savings with the event-segmentation story.
- **Top-down filtering of bottom-up proposals** begins here and becomes explicit in the follow-up (infinite-ends.md).
