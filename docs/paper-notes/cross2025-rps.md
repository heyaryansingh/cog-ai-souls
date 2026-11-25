# Understanding Human Limits in Pattern Recognition: A Computational Model of Sequential Reasoning in Rock, Paper, Scissors

## 1. Citation and coverage
Logan Cross, Erik Brockbank, Tobias Gerstenberg, Judith E. Fan, Daniel L. K. Yamins, Nick Haber (Stanford). 2025. arXiv 2508.06503 (venue not named on the page). https://arxiv.org/abs/2508.06503
Coverage: full arXiv HTML processed through a web-fetch extraction tool (direct download blocked). I read a detailed extraction of the full text (main + tables), not every sentence verbatim.

## 2. Core idea and method
- Question: when people fail to exploit a patterned opponent, is the bottleneck **hypothesis generation, hypothesis evaluation, or execution**?
- Human data: Brockbank & Vul (2024): N = 218, 300 rounds each vs one of 7 bots; payoff win +3, tie 0, loss -1. Each bot follows its rule with 90% probability (10% random).
  - Transition bots (4): self-transition(+), self-transition(-), opponent-transition(+), opponent-transition(0) (copies player's last move).
  - Outcome bots (2): W0L+T- (stay after win, up after loss, down after tie), W+L-T0.
  - Combined bot (1): depends on previous outcome AND previous transition (nine condition-action pairs).
  - Complexity = how many previous events are needed to predict the move.
- Model: **Hypothetical Minds (HM)**, LLM agent (GPT-4o main; GPT-3.5, Llama 3 compared) with memory, a Theory-of-Mind module, and decision reflection.
  - Generation: LLM proposes natural-language strategy hypotheses from interaction history, refining the top-valued previous ones.
  - Evaluation: top_h = 5 hypotheses each predict the next opponent move; value update V <- V + alpha (r - V) with r = +1/-1, alpha = 0.3; a hypothesis is "validated" at V > 0.7 and used until it falls below.
  - Reflection: CoT maps chosen hypothesis -> predicted move -> counter.
  - Ablations: no hypothesis evaluation (regenerate every round), ReAct (no ToM module), base LLM (no CoT).

## 3. Experiments and numbers
- Humans: transition bots 57.9-66.5% wins (chance 33%); outcome bots 41.3%, 39.7%; combined bot 34.1% (n.s.).
- Fit to humans (Table 1):
| Metric | Human-human ceiling | HM | No hyp. eval | ReAct | Base LLM |
|---|---|---|---|---|---|
| Win-rate L2 (lower better) | 0.17 | 0.53 | 0.74 | 0.94 | 0.91 |
| Trajectory L2 (10-round bins) | 1.40 | 2.95 | 4.27 | 5.17 | 4.96 |
| Trajectory corr r | 0.81 | 0.38 | -0.23 | -0.27 | -0.02 |
- HM matched humans on 6/7 bots, 74-77% vs simple transition bots (ReAct 14-50%). HM (GPT-4o) failed on opponent-transition(0), misreading copying as opponent-transition(+), below chance; Llama 3 handled it.
- **Give Hypothesis** (true strategy in words): >80% wins vs 6/7 bots. **Choose Hypothesis** (pick among the 7 true strategies): nearly as good. Both still fail on the combined bot (352-word, nine-contingency description): nested contingencies are hard to *apply* even when known.
- Hypothesis memory: top_h 0 -> 1 is a big gain; 1 -> 9 changes little. Temperature 0.2 -> 1.0 hurts (search is in the wrong region, not too narrow). **Attention scaffold** (highlight relevant contingency, e.g., "look at outcomes") raises outcome bots to 67%/72% and opp-transition(0) to 47%; analogical scaffold less (47%/58%, 38%).
- 3 seeds per experiment in several analyses (SEM).

## 4. Failure modes and limitations
Stated: scaffold effects not tested in humans; one dataset; GPT-4o dependence (opp-transition(0) failure); "Bayesian" is a verbal approximation; local refinement limits diversity; few seeds; qualitative analysis of human strategy reports.
Noticed: trajectory correlation 0.38 vs ceiling 0.81 means learning dynamics are poorly captured; the 90% rule-following rate is fixed, so noise vs complexity are not dissociated; no formal Bayesian model over a grammar of dependencies as a comparison, which would separate "space contains the hypothesis but prior is low" from "space lacks it".

## 5. Ideas to steal
- **Bottleneck-localization design for E2/E4/E6**: run the agent in three conditions per boss: (a) normal, (b) **Choose-hypothesis**: the true boss program is guaranteed to be in the particle set / proposal support, (c) **Give-hypothesis**: posterior fixed at the true program (oracle). Differences (a)-(b) = generation/proposal bottleneck, (b)-(c) = evaluation bottleneck, (c)-optimal = execution bottleneck. This is a clean decomposition of the amortization gap and of where the agent fails on novel bosses.
- **Difficulty axis exactly as in the plan**: boss transitions conditioned on (i) own previous move, (ii) the player's previous action ("opponent-transition"), (iii) previous outcome (did the last attack hit / did the player heal), (iv) combined outcome x transition. Use the 90% rule-follow noise level. Prediction from humans: success drops from ~60% to chance by (iv). The agent with full-hypothesis-space SMC should not drop, unless proposals miss; the human-like agent (small N, low-order prior) should.
- **Reactive bosses are the important case**: the hardest human condition was conditioning on the *opponent's* (player's) behaviour and on outcomes. Include boss laws conditioned on player action ("if player rolled last, thrust"), not only self-transitions.
- **Metrics**: win-rate L2 distance to humans per opponent, trajectory L2 over 10-round bins, and trajectory correlation, with a human-human split-half ceiling. Port to deaths/hit-rate per move per attempt bins for E4.
- **Attention scaffold = conditioning features**: what helped most was pointing at the right *context variable*. For the intuition net, explicitly expose candidate context features (last player action, last outcome, phase, HP) as inputs/slots; for the program prior, a hyperprior over which context variables laws depend on (learned across bosses = transfer of "what to pay attention to").
- **Hypothesis memory matters, quantity doesn't**: keeping 1 good hypothesis across steps is most of the gain; supports small particle counts plus persistent best-program memory across deaths.
- **Execution limit for complex laws**: even with the right model, nested contingencies are hard to act on; for our agent, measure whether the reflex policy can execute a 9-contingency law vs the planner.

## 6. Structures and models of thought
- **Generate / evaluate / act decomposition** of opponent modeling, each separately ablatable.
- **Hypotheses as programs over context variables** (transition, outcome); dependency order as a complexity measure, i.e., a prior over program size/context arity.
- **Value-tracked hypothesis memory** (RL-style running accuracy, validated threshold): a cheap approximation to posterior weights, usable as the human-like baseline.
- **Attention to the right context variable** as the key learned inductive bias; maps to hierarchical priors over law structure.
