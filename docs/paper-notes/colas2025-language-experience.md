# Language and Experience: A Computational Model of Social Learning in Complex Tasks

## 1. Citation and coverage
Cedric Colas, Tracey Mills, Ben Prystawski, Michael Henry Tessler, Noah Goodman, Jacob Andreas, Joshua Tenenbaum (MIT, Stanford, Google DeepMind). 2025. arXiv 2509.00074 (venue not named on the page). https://arxiv.org/abs/2509.00074
Coverage: arXiv HTML processed through a web-fetch extraction tool (direct download blocked). The first ~100k of ~108.6k characters were covered (the unread tail is most likely references/late appendix). So: main text and most appendices via a detailed extraction, not verbatim reading.

## 2. Core idea and method
- Social learning = **joint Bayesian inference over executable world-model programs** from experience E and linguistic advice L: P(T | E, L) proportional to P(E|T) P(L|T) P(T).
- **Theories T are VGDL programs**: object types, pairwise interaction effects, rewards, win/loss conditions. Prior: object type uniform; each object pair interacts w.p. 0.25 (uniform effect type); each object's death contributes to win/loss w.p. 0.1; favors simpler theories. Theory space >1e20 for 5-object games (1e30 in appendix; inconsistent).
- **Experience likelihood**: experience segmented into discrete *events* (interactions), assumed conditionally independent given T; P(e_i | T) estimated by **20 simulations per theory**.
- **Language likelihood**: P(L|T) approx P_LM(L | prompt(T)) with LLaMA-3.1-70B as a speaker model; only relative values across theories matter.
- **Language-guided proposals**: the LM answers multiple-choice questions about each VGDL rule given the message; answers averaged with the base proposal distribution.
- **Inference**: particle filter, M = 20 particles; each inference step = 1 resampling + 5 Metropolis rejuvenation steps; run every 20 environment steps, and additionally **whenever a new object type appears or the agent dies**.
- **Planning**: subgoals = object-object interactions; **exploration value = disagreement across the 20 theories**; exploitation values: 10 win condition, 8 protects essential resources, 6 creates tools, 2 collects resources/reward, 0 otherwise. Plans = 10-step action sequences refined by a genetic algorithm (3 generations); +100 win / -100 loss; ten 3-step lookaheads trigger replanning to avoid deaths.
- **Advice generation**: LM conditioned on the agent's MAP theory writes a message for the next learner.

## 3. Experiments
- 10 VGDL games (beesAndBirds, aliens, jaws, preconditions, missileCommand, pushBoulders, avoidGeorge, plaqueAttack, relational, portals), 4 levels each; agent sees symbolic state.
- Humans: 122 Prolific, 120 kept (40/condition), 5 random games each, 15 lives per game; conditions experience-only / + human advice / + model advice; wrote advice after each game; median 49.3 min.
- Simulations: 20 runs per condition; iterated learning chains of 10 generations, 2 lives each, passing one message.
- Metrics: normalized area under learning curve (nAUC), Delta nAUC; lives needed per level; coded message content.
- Baselines: oracle (true rules), Double DQN, pure LM (LLaMA-3.1-70B ReAct + scratchpad + CoT).
- Key numbers: experience-only, median human solved 9/10 games, model 10/10 within 10 lives; DDQN solved 0 levels in 10 lives; pure LM solved 0 levels in 7/10 games. Human advice: median attempts 4 -> 2.25 (humans), 2.5 -> 1.25 (model); Delta nAUC 0.12 humans, 0.04 model. Model advice: Delta nAUC 0.15 humans, 0.088 model; models learn better from model advice than human advice (Delta 0.052). Warnings cut deaths 37-67%; wrong advice suppressed useful exploration (avoidGeorge). Removing language-guided proposals: Delta = -0.058. Iterated learning: Delta nAUC 0.44-0.57 vs generation 1, with regressions in plaqueAttack and relational. Human-model correlation of advice benefit only r = 0.17.

## 4. Failure modes and limitations
Stated: VGDL is coarse; LM is an approximate speaker; no metacognitive check of advice reliability; human messages contain strategies/analogies the model cannot use; **motor-heavy games (portals, plaqueAttack, aliens, missileCommand) benefit little from advice**; single-teacher chains brittle; only one LLM.
Noticed: symbolic state access sidesteps perception; 20 particles in a 1e20 space means proposals do almost all the work; per-event independence ignores timing/sequential structure (exactly what bosses have); the model beats humans on several games, so it is not a tight human model; the ablation sign inconsistency (-0.030 vs experience-only) should be checked.

## 5. Ideas to steal
- **Inference schedule**: run SMC (1 resample + 5 MH) at fixed intervals AND at surprise events: new object type appears, or agent dies. Direct analogue: rejuvenate on new move primitive (ESS collapse) and on death. Their schedule validates the plan's "slow mind after death + surprise-gated early rejuvenation".
- **Exploration value = disagreement across particles** for scouting: choose actions (bait a specific move, stand at a range) that maximize disagreement among boss-program particles about the outcome. Simple, cheap alternative to full expected information gain, and with 20-100 particles it is directly computable.
- **Likelihood by simulation (20 rollouts per theory) for events lacking closed form**; use only for conditional laws where the HSMM filter is not exact.
- **"Advice" as a structured prior channel**: a cheap E3 extension, give the agent a one-line hint ("the boss punishes healing", "the third swing is delayed") as a likelihood/proposal bias over program slots, and measure deaths saved. Their numbers (attempts halved) give a target. Also: **wrong advice suppresses exploration**, so test a misleading hint.
- **Agent-to-agent transmission**: the MAP boss program from a finished fight is itself a message; iterated learning across bosses of a family = the family CRP prior. A chain experiment (each "generation" gets 2 deaths + the previous MAP) is a neat E3 variant.
- **Metric**: nAUC of the learning curve over lives, and lives-per-level, both comparable to humans; adopt nAUC for E1.
- **Warning**: advice/inference helps least in motor-heavy games. Souls bosses are motor-heavy, so separate "knows the pattern" from "can execute the dodge" in evaluation (e.g., oracle-program agent vs ours, as their oracle baseline does).

## 6. Structures and models of thought
- **Program-like theories with explicit interaction laws** (VGDL) = boss programs with conditional laws.
- **Multiple evidence channels combined in one posterior** (experience + language): an LM is used as a likelihood and proposal, not as the learner.
- **Uncertainty-directed exploration from particle disagreement**: a lightweight metacognitive signal.
- **Cultural accumulation via compressed messages**: MAP program as a transmissible abstraction; supports family libraries and transfer.
