# AI GameStore

## 1. Citation and reading status
- **Title:** AI GameStore: Scalable, Open-Ended Evaluation of Machine General Intelligence with Human Games
- **Authors:** Lance Ying, Ryan Truong, Prafull Sharma, Kaiya Ivy Zhao, Nathan Cloos, Kelsey R. Allen, Thomas L. Griffiths, Katherine M. Collins, José Hernández-Orallo, Phillip Isola, Samuel J. Gershman, Joshua B. Tenenbaum
- **Affiliations:** MIT, Harvard, UBC, Princeton, Cambridge, UPV
- **Date and venue:** arXiv preprint, submitted 2026-02-20; no venue stated
- **Links:** https://arxiv.org/abs/2602.17594 · aigamestore.org
- **What I read:** The full arXiv HTML through Appendix I and the Figure 14 caption (the end of the document), via a fetch tool that summarizes pages, so I saw it in paraphrase, not verbatim. I also checked author names on the abstract page. Exact per-capability values exist only in figures, which I could not see.

## 2. Core idea and method
- **The claim.** General intelligence should be measured as the ability to learn and play the "Multiverse of Human Games", meaning games humans made for humans.
- **The platform** has four pipeline stages.
  1. **Sourcing.** 7,500 games were sampled from Apple App Store Top-100 lists (action, adventure, casual, puzzle and board categories, across 15 countries), plus 500 top Steam indie games. A game passes the popularity filter with at least 10k reviews and a rating above 4.5. Gemini 2.5 Flash then scores suitability out of 100: playable in minutes, expressible in p5.js, has a quantifiable score, and needs no game-specific expertise. 100 games were kept.
  2. **Generation and refinement.** Claude Sonnet 4.5 writes each game in p5.js:
     - keyboard only, pausable, with a score;
     - multiple levels of increasing difficulty, lives, and resets.

     First, an LLM-written test script finds bugs and the LLM fixes them. Then humans in the loop play the game and give natural-language feedback; each refinement step takes about 2 minutes, with 4.7 steps per game on average. One game takes about 30 minutes end to end. The coauthors did all the refinement.
  3. **Annotation.** Each game is rated on 7 capabilities from 0 to 5:
     - Visual Processing (VP)
     - Spatial-temporal coordination (ST)
     - Memory (ME)
     - Planning (PL)
     - World-Model learning (WM)
     - Physical reasoning (PH)
     - Social reasoning (SO)

     Three author-annotators used rubrics (Appendix F) and resolved disagreements by deliberation. No inter-rater agreement statistic is reported.
  4. **Evaluation.** Humans and models get the same 120-second game-time budget. 10 games are public and 90 are a private test set.
- **Model harness.**
  - The game pauses every second (1 call = 1 second of game time, so a game is about 120 calls).
  - Each call receives:
    - the game description and controls;
    - a scratchpad memory that the model rewrites each call;
    - the actions and screenshots since the last call;
    - the available keys.
  - Each call outputs:
    - the updated scratchpad;
    - 5 action lists, one per 0.2-second segment;
    - reasoning of 10 sentences or fewer.
  - Actions: UP, DOWN, LEFT, RIGHT and SPACE as presses; HOLD_ versions of each; NOOP; and R to restart.
  - Seven VLMs were tested: GPT-5.2, GPT-5-mini, Gemini-2.5-Pro, Gemini-2.5-Flash, Claude-Opus-4.5, Qwen-3-VL-32B and Llama-4-Maverick. Each played 3 runs per game, with scores averaged.

## 3. Experiments and numbers

### Human study
- **Participants:** 106 people from Prolific (mean age 38.8; 58 male, 46 female, 2 non-binary), under MIT IRB approval.
- **Design:** Each person played **10 random games in sequence, 120 seconds each**, so each game had about 10 human plays.
- **After each game:** fun and challenge ratings on 0–100 sliders.
- **Logged:** score every 30 frames, gameplay video, and action sequences.
- **Not reported:** compensation, instructions, and exclusion criteria.

### Metric (exact)
- **Per-game normalized score:** clip(100 · RawScore / HumanMedianScore, 1, 10000).
- **Aggregate:** the **geometric mean** over games, chosen because scores are heavily skewed. A median gives similar results.
- **Uncertainty:** 95% confidence intervals, presumably bootstrapped. The top 6 models are *not* significantly different at p < 0.05.

### Table 9: geometric mean of normalized score [95% CI]

| Model | Public 10 | All 100 |
|---|---|---|
| GPT-5.2 | 6.71 [2.67, 16.22] | 8.26 [5.93, 11.28] |
| Claude-Opus-4.5 | 5.91 | 7.74 [5.50, 10.68] |
| Gemini-2.5-Pro | 8.99 | 7.49 |
| Gemini-2.5-Flash | 6.27 | 7.07 |
| GPT-5-mini | 5.01 | 6.13 |
| Llama-4-Maverick | 4.74 | 5.91 |
| Qwen-3-VL-32B | 2.53 | 4.68 [3.39, 6.41] |

The best models therefore reach about 8% of the human median on the geometric mean.

### Other findings
- **Distribution of scores:**
  - About 2/3 of games show some progress, typically 10–30% of the human median.
  - 30–40% of games score below 1%.
  - A few simple casual games score above the human median.
- **Speed:** Models need more than 1,200 seconds of wall clock for 120 seconds of game time, about 12–18× slower than humans, with many taking 20 minutes or more.
- **Capability effects:**
  - The weakest capabilities are Memory, Planning and World-Model learning; Social and Planning degrade further at high demand (Fig. 7).
  - Performance falls as the number of capabilities a game demands increases (Fig. 8).
  - Humans' cumulative score rises steadily over time; models plateau early or flatline (Fig. 9).
- **Latency control:** Restricting to games with ST ≤ 2 does not change the conclusions (App. I), so real-time demand alone does not explain the gap.
- **Ratings (all 100 games):** fun 52.4 ± 13.2, challenge 59.3 ± 13.9.
- **Mean demand (all 100 games):** VP 2.23, ST 2.04, ME 0.20, PL 0.95, WM 0.52, PH 0.96, SO 0.50. Memory and WM demands are low on average, so few games stress them.

## 4. Failure modes and limitations

### Reported by the authors
- Losing track of information across calls despite the scratchpad.
- Failing to infer underspecified rules.
- Shallow lookahead.
- Early progress followed by a stall, or total failure.
- Slow reasoning.
- Games are short and casual; NPCs are naive, with no opponent modelling; LLM-generated levels are often trivial or impossible.
- Results depend on one harness design.

### What I notice
- **Internal inconsistencies:**
  - the abstract says "<10% of human average" while the introduction says "<30%";
  - Table 10's geometric means match the public-10 column, not the all-100 column;
  - Figure 5 says 8.5 while Table 9 gives 8.26.
- **Annotation:** done by the authors, with no agreement statistic.
- **Human sample:** about 10 humans per game, so the human median is noisy.
- **Normalizer:** the human median is a weak reference with no human learning curve over repeated plays. Each human plays each game once for 120 seconds, so this measures *first-contact* skill and says nothing about learning across attempts.

## 5. Ideas to steal

1. **Use their exact normalized score.**
   - Per boss: HNS = clip(100 · agent_metric / human_median_metric, 1, 10000).
   - Aggregate over bosses with the **geometric mean**, plus a bootstrap 95% CI over bosses.
   - Our plan's §5 "human-normalized score" should use exactly this, with the metric = AUC of the win-rate curve over 20 attempts, *or* boss-HP-removed when wins are rare. The clipping floor of 1 matters, because a boss with zero wins would otherwise send the geometric mean to 0.
   - Report the median as a robustness check, and state significance between methods.
2. **Fix their human-design weakness.** Our human study should give each participant **repeated attempts on the same boss**, so we get a human *learning curve* (deaths to first win, AUC over 20 attempts), not only first-contact skill.
   - Copy their logistics: randomly assign each participant a subset of bosses (for example 3–4 bosses × N attempts), log the full action and event stream, and collect fun and challenge sliders after each boss.
   - Aim for 10 or more humans per boss, so the median normalizer is stable.
   - Cover the T1 to T4 splits in a counterbalanced order.
3. **Use equal game-time budgets, but log wall-clock separately.** Pause the game for agent decisions as they do. Report the **wall-clock ÷ game-time ratio** for every method; frontier VLMs are at about 12–18×. This supports the compute-normalized score, and it shows the metareasoning gate as resource-rational: the gated agent should approach real time (a ratio of about 1) while the planner-only agent does not.
4. **Annotate every boss with a capability profile.** Rate each boss 0–5 on axes adapted to bosses:
   - timing-inference demand (delayed attacks);
   - memory demand (long combos, phase history);
   - cue ambiguity (confusable pairs, low cue informativeness);
   - planning demand (stamina and heal management);
   - rule novelty (T4).

   Then regress performance on demand level, as in their Fig. 7, and on the *number* of high-demand axes, as in Fig. 8.
   - Prediction: amortized-only drops with novelty or ambiguity demand, and SMC much less.
   - Because our generator is programmatic, the demand levels can be **computed from the boss program** (for example the entropy of transitions, timing variance, and the number of confusable pairs) instead of hand-rated, which avoids their inter-rater problem.
5. **Cumulative-score-versus-time plots,** human against agent (their Fig. 9), within a fight and across attempts. Diagnose "early plateau" agents.
6. **Latency control analysis (their App. I).** Re-run the analysis restricted to bosses with low reaction-time demand (high cue informativeness). This shows whether agent failures are about inference or reflex speed, which separates E1 effects from perception speed.
7. **A public/private split.** Hold out the T3/T4 bosses entirely from all development, including the hyperparameter tuning of the gate and intuition net, to avoid overfitting.

## 6. Structures and models of thought
- **Game-time pausing** makes deliberation free in game time. Humans cannot pause, so the relevant cognitive structure is **resource-rational metareasoning**: knowing when thinking is worth its time cost. Our gate is exactly this, and should be evaluated *without* pausing as a stretch condition.
- **The scratchpad as externalized working memory** fails for VLMs. This suggests that a structured belief state (posterior over latent move, stage and duration) is a better memory than free text. It also supports explicit HSMM filter state as the agent's working memory.
- **The capability decomposition** (WM learning, memory, planning) is a coarse cognitive architecture: the axes on which models fail are the ones that need an internal, updatable world model. This is the same argument as Ying et al. 2025, and it supports program-like hypotheses updated online.
