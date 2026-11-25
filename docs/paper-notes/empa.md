# Human-Level RL through Theory-Based Modeling, Exploration, and Planning (EMPA)

## 1. Citation and reading status
- Pedro A. Tsividis, João Loula, Jake Burga, Nathan Foss, Andres Campero, Thomas Pouncy, Samuel J. Gershman, Joshua B. Tenenbaum. arXiv 2107.12544v1, 27 Jul 2021 (preprint; course reading). https://arxiv.org/abs/2107.12544
- **How I read it:** I read the **full text** from the local copy /mnt/project-files/cog-ai/text/2107.12544v1.txt: main text, Methods, Supplementary game descriptions, heuristics, Extended Data captions and references. Figures were not visible; only their captions were.

## 2. Core idea and exact method
- **Theory-based RL.** Learn a symbolic, object-oriented, simulatable game model by Bayesian inference. Explore to resolve uncertainty about that model. Plan in it with theory-derived heuristics.
- **Input.** Symbolic: objects, classes, positions, collision events. Fully observed.
- **Things it is told.** The avatar's identity and type, its projectile, and the identity and dynamic type of walls (walls' interaction rules must still be learned).

### Model space (VGDL subset)
- θ = (θ_S sprites, θ_I interactions, θ_G termination).
- **θ_S.** Each class has a dynamic type: vgdlType (static, random, chaser or avoider, missile, ...) plus speed and orientation. Positions come from small programs, e.g. `nextPos(Missile, speed=2, Right)`.
- **θ_I.** Rules (c_j, c_k, ξ, π). The predicate ξ comes from 14 types: stepBack, bounceForward, reverseDirection, turnAround, wrapAround, pullWithIt, undoAll, collectResource, changeResource, changeScore, teleport, clone, destroy, transform.
  - Preconditions π are count(agentState, item) < N or ≥ N.
  - **No action at a distance:** events happen only on contact.
- **θ_G.** Win or loss when count(class) == 0.
- **Size.** Over 3.6×10³⁷ descriptions with only 10 classes.

### Inference
- p(θ | s, a) ∝ p(s | θ, a) p(θ), with a **uniform prior**.
- The likelihood factorises (Eq. 3) into three parts:
  - **termination** (indicator that the predicted Win/Loss/Continue matches);
  - **interactions** (Eq. 5): 1−ε if every observed event is *explained* by some rule (η) *and* every *implied* event for that class pair occurred (υ), else ε;
  - **free movement** (Eq. 6): product of per-object next-position likelihoods under the dynamic type.
- **Factorised posterior:**
  - dynamic types: an independent posterior per class, by **enumeration**;
  - interactions: **MAP only** (they are deterministic). On a violation, it **proposes the minimal conditional rule** that explains it (quantifiers over agentState);
  - termination: keeps a **superset** of candidate win conditions that the planner tries to satisfy together.

### Exploration (theory-based curiosity)
- It approximates expected information gain. Because interactions are class-determined and deterministic, **one contact between a class pair is almost fully informative**.
- So the exploratory goals are: **contact every class pair whose interaction has not been observed**.
- Avatar-object goals are **reset whenever agentState changes** (e.g. after picking up a key: maybe now walls are passable?).
- To learn termination, if a class's count can be reduced, set **"drive its count to 0"** as a goal.
- Exploration is non-myopic: long plans whose only purpose is to produce informative contacts.

### Planning
- **Search.** Best-first search over states imagined with the **MAP model** (Algorithm 1), with **Iterative Width (IW)** novelty pruning.
  - Atoms are (object, present), (object, x, y) and (avatar, x, y, orientation).
  - Atoms are not generated for random objects, short-lived objects or the agent's own projectiles.
  - Repeated-position penalty α·count²: α = −10 if other objects move, else −1.
- **Intrinsic value** V = R_G + R_SG + R_GG:
  - **R_G:** +∞ for win, −∞ for loss (prune).
  - **R_SG** (Eq. 7): ρ Σ_g (N_g − |c(g)|)/|c(g)|² · (−1)^{[g = Win]}, with ρ = 100.
  - **R_GG** (Eq. 8): distance between the nearest instance of the target class and the nearest instance of a class that can destroy, pick up or transform it, normalised by |c(g)|². Goal gradients are derived automatically from the MAP theory.
- **Modes, run by a metacontroller:**
  - **Long-term:** returns the first plan that reaches a win or exploration goal. Budget 1,000 nodes, **doubling on repeated failure**.
  - **Short-term:** also accepts any subgoal. Budget drawn from {200, 500, 1000}.
  - **Stall:** returns any short non-losing plan, 50 nodes.
  - The game starts in short-term mode. The agent counts as **"stuck" if it lost twice the same way, or no other objects move**; then it switches to long-term. A non-stuck failure triggers one stall cycle. Stuck in long-term mode means restart the level and double the budget.
- **Prediction-error monitoring.** Execute the plan while the observed state matches the predicted state. **Replan** if the avatar's position is off, or a dangerous object within 3 squares is not where predicted.
- **Observe before acting.** Wait 15 steps at the start of a new game and 5 at a new level to learn the motions of objects.
- **Extra heuristics.**
  - Picking up resources counts as a subgoal.
  - Flicker objects (swords) transfer goal gradients to the avatar.
  - When firing projectiles, simulate random-action rollouts.
  - While a projectile is in flight, take no-op unless within 3 of danger.
- **Compute.** Unoptimised Python VGDL engine; the planner may take as long as needed, up to **24 hours per game**.

## 3. Experiments
- **Games.** 90 games: 27 base games (17 from GVGAI, 10 new) plus 1–4 variants each that change objects, dynamics or win/loss rules. 4–6 levels per game; levels share rules but differ in layout.
- **Humans.**
  - 300 MTurk participants, paid $3.50 plus up to $1 bonus. Each played 6 games, **20 people per game**, no one played two variants of the same game.
  - Told only the keys and that games have levels. They could restart a level or forfeit.
  - Games previously played were excluded.
  - **Color-only mode** (uniform coloured blocks, colours randomised per participant), so object semantics carry no prior.
- **Fairness.** Humans play in real time; models only step when they act. The x-axis is **agent steps**.
- **Runs.** EMPA 10 seeds; ablations 5; DDQN and Rainbow 3 seeds each, up to 1M steps.
  - DDQN ε annealed over 100k steps (the best of 1k/10k/100k).
  - Rainbow uses the data-efficient settings, ε annealed over 150k steps.
- **Metric: learning efficiency** κ = (levels completed / levels in game) × (levels completed / steps to completion), measured over the first 1,000 steps (10,000 for Avoidgeorge). Confidence intervals come from the efficiency ratio of the mean EMPA run to the mean human, with 10,000 bootstrap resamples per game.
- **Results:**
  - EMPA is **within 0.1×–10× of humans on 79/90 games**. It is better than humans on about 2/3 of games and worse on about 1/3. On one game, no EMPA run passed level 1.
  - DDQN is more than 100× worse on 67/90, more than 1,000× worse on 45/90, and more than 10,000× worse on 22/90. Rainbow is no better on average, because humans finish within about 1000 actions, while deep RL is still exploring randomly.
  - **Examples:**
    - Bait: EMPA solves 5 levels in under 1000 steps; DDQN needs 250k steps for 2 levels.
    - Zelda: humans and EMPA under 500 steps; DDQN 100k steps for 4 levels.
    - Butterflies: DDQN fails the last level because of a geometry change.
  - **Ablations.** ε-greedy exploration (2 variants), and no subgoals, no goal gradients, both, no IW, or all three. Each fails to win even one level on a subset of games, reaching DDQN-like failure rates.
- **Fine-grained human comparison:**
  - occupancy **heat maps** (time per board cell, normalised per level);
  - early levels diffuse, later ones targeted;
  - **object-interaction profile** over 78 games: positive, instrumental, neutral or negative (defined in Methods). EMPA's profile matches humans better than DDQN's (**L1 distance between interaction distributions**), especially in slow-paced games. DDQN mostly hits walls;
  - **contacts with deadly objects:** humans and EMPA only a handful of times; DDQN hundreds to thousands.
- **Human-EMPA differences.**
  - EMPA moves faster and is always aware of every action.
  - EMPA lacks high-level priors (e.g. "a game has one win condition"), so it keeps exploring other ways to win after it has found one.

## 4. Limitations
- **Reported by the authors:**
  - Assumes **determinism and class uniformity** of interactions. Suggested fix: hierarchical Bayes for probabilistic rules and subtypes.
  - No social or linguistic learning.
  - Does not pursue novel self-chosen goals.
  - Unknown whether it works when the environment falls outside the hypothesis space.
  - Weak high-level priors.
- **Noticed by me:**
  1. **Symbolic input plus privileged knowledge** (avatar, walls).
  2. **MAP-only interactions and MAP planning:** no planning over uncertainty, so a fooled MAP leads straight into danger. This is precisely the feint and delayed-attack failure the project targets.
  3. Exploration targets only *contact* events. Timing and stochastic patterns (which are what a boss is) are learned passively.
  4. **The 24-hour compute budget per game and unlimited thinking time** make the comparison with real-time humans generous.
  5. κ ignores deaths. An agent can die many times cheaply as long as total steps are low.
  6. Only 20 humans per game, from MTurk.
  7. Uniform prior: no transfer across games.

## 5. Ideas to steal
1. **Adopt κ, adapted to deaths.**
   - κ_boss = (phases cleared / total phases) × (phases cleared / attempts-to-clear), or with ticks as the denominator.
   - Report the "within 0.1–10× of human on X/Y bosses" style summary, plus "PPO more than 100× / 1000× worse on N bosses." This makes E1 directly comparable to the course reading.
2. **"Contacts with deadly objects" becomes "hits taken per move type."** Count how many times the agent and humans get hit by each move before they stop getting hit by it (their Fig. 6B). Pair it with an **interaction profile**: time spent in punish windows, safe zones, neutral zones and danger zones. Compare with **L1 distance to the human profile**. This is a concrete E4 metric.
3. **Scouting = contact goals over unobserved (boss move × player response) pairs.**
   - EMPA's rule "one contact per class pair is enough, reset goals when agentState changes" gives an explicit scouting policy:
     - each attempt, target eliciting and testing (dodge left / right / roll-through / block) every boss move whose response outcome is still unknown;
     - **reset when the player's state or the boss phase changes** (low HP, phase 2 — "maybe phase 2 changes the combo").
   - This is a cheaper and more interpretable alternative to full expected-information-gain scouting, and a good ablation: EIG scouting vs EMPA-style contact goals vs ε-random.
4. **Propose the minimal conditional law on a violation.** When a learned transition is violated, propose the minimal precondition (HP below x%, distance, phase, combo count) that explains it. This is a natural SMC split/rejuvenation move for the plan's "conditional laws."
5. **Prediction-error replanning as the gate's trigger.** Replan when the observed state diverges from the predicted one near danger (their 3-square radius). This is a hand-coded baseline for the learned metareasoning gate (E5). Also copy "observe for 15 steps before acting": an opening "watch the boss" phase.
6. **Planner modes and a "stuck" detector.**
   - "Lost twice the same way" switches to deeper planning or a different strategy. Mapped to bosses: same move kills you twice, so escalate deliberation or switch to a scouting attempt.
   - The node budget **doubles on repeated failure**. That is a simple resource-rational schedule to compare against the learned gate.
7. **Planning heuristics from the theory.** Derive subgoals and goal gradients from the inferred boss program, e.g. "punish window after move X," "distance to safe zone during move Y." IW novelty pruning over (player position, boss move, stage) atoms keeps a laptop planner tractable.
8. **Human protocol to copy.**
   - **Color-only / semantics-free rendering** with colours randomised per participant (bosses as abstract shapes), so humans rely on learning rather than genre priors.
   - Exclude anyone who has seen a boss before. Use agent decision steps vs real-time ticks carefully.
   - 20 per condition was enough for them. The plan's 5 friends × 30 min supports direction of effect only, and the write-up should say so.
9. **Report their ablation pattern.** Ablate exploration and each planner heuristic, and show a fraction of bosses with *zero* wins (not only mean efficiency). They show failures concentrate where multi-step sequences are needed. Boss analogue: multi-hit combos and delayed attacks.

## 6. Structures and models of thought
- **Program-like, object-oriented causal theories** (types, contact laws, termination) with a factorised likelihood. This is the direct precedent for boss programs (move primitives, conditional laws, phase termination). Factorising the posterior by component (enumerate timings, MAP for deterministic laws, superset for goals) is a practical recipe for keeping SMC tractable.
- **Core-knowledge priors** (no action at a distance, determinism, class uniformity) explain fast learning. Our analogues: telegraph precedes hit, same move gives the same hitbox, timing is scalar-noisy. EMPA's own limitation (determinism) is exactly where the plan's probabilistic HSMM and hierarchical prior extend it.
- **Curiosity as goal generation for a planner** (non-myopic epistemic goals): a cognitive mechanism for directed scouting.
- **Goal decomposition into subgoals and gradients** (means-ends analysis). Hierarchical intrinsic value derived from the theory is what makes sparse-reward planning tractable.
- **Metacontrol** (mode selection, stuck detection, prediction-error-triggered replanning): a simple, explicit form of the metareasoning the plan learns. It is a good hand-designed baseline and a source of human-like hesitation predictions.
