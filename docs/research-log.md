# Research log: Reading the Boss

A running, dated log of experiments, results, decisions and new ideas. Newest entries at the bottom. Every number is reproducible from the script named in its entry (default seeds unless stated). Win rates carry 95% Wilson intervals.

---

## 2026-11-25 · Entry 1: M0, environment and exact inference

- **Built:** MiniSouls (1D lane, 20 ticks/s, ~190k ticks/s on one core), the boss generative program (an explicit-duration HSMM over 9 primitives), and the exact forward filter (about 1k states).
- **Verified:** the filter equals brute-force enumeration of every latent path, computed with the simulator's own representation, to 1e-9 in both posterior and log-evidence (`tests/test_filter.py`).
- **Decision:** model and simulator share one `Boss` object, so model/simulator mismatch is impossible by construction. Mismatch will only ever be introduced deliberately, by giving an agent a different believed program.

## 2026-11-25 · Entry 2: difficulty must create a learning gap

- **First result (easy settings):** a planner that knows only the generic prior won 100%, the same as the true-model planner. Learning a boss would have bought nothing.
- **Decision:** raise difficulty until boss-specific knowledge matters. New defaults:
  - between-boss timing spread 0.35;
  - 6-tick tell before a move's visual group is readable;
  - damage ×1.4;
  - 4 i-frames;
  - 2 heals;
  - dodge cost 15, stamina regen 1.0/tick.

## 2026-11-25 · Entry 3: the ceiling ladder (M1)

Script: `rtb/evaluate.py`. 40 training bosses × 2 attempts = 80 fights per row.

| Agent | What it knows | Win rate [95% CI] | Deaths |
|---|---|---|---|
| State cheat | true boss latent state (move, stage, exact remaining wind-up) + heuristic | **97.5%** [91–99] | 2.5% |
| Oracle planner (12 rollouts, 45-tick horizon) | true boss *program*; state inferred by the exact filter | **72.5%** [62–81] | 27.5% |
| Prior-model planner | right move list, typical timings, uniform transitions | **46.2%** [36–57] | 53.8% |
| Generic-model planner | only Θ*: one move per primitive, typical timings | **37.5%** [28–48] | 62.5% |
| Dodge-everything / random | nothing | 0% | 100% |

**Findings:**
1. **More planning compute does not raise the oracle.** 12 against 24 rollouts, and horizons of 30, 45 or 60, all land at 71–73% on 80 fights. An early 24-fight sweep suggested 92% for 12×45; that was noise, and is the reason all comparisons now use at least 80 fights with CIs.
2. **The 97.5% → 72.5% drop is perceptual and timing ambiguity, not a weak planner.** With the true program, the agent still cannot tell a slam from a delayed slam during the shared wind-up, and wind-up lengths vary from use to use. Death coding on 16 bosses (`scripts/death_taxonomy.py`):
   - most hits are taken **while blocking**, the rational hedge when the remaining wind-up is too uncertain to time a 4-tick dodge;
   - next most common are hits while **healing**.

   This is the "averaging beats committing" regime the project is about.
3. **A concave HP utility hurt (61.3%).** It over-valued heals. Reverted to the linear env reward.
4. **The room for learning is 72.5% vs 46.2% / 37.5%.** That gap is what the slow mind (M2) has to close from deaths alone.

**Revised M1 criterion:** the original "oracle ≥ 90%" target assumed full observability. The right ceiling for a learner is the *oracle* (true program, inferred state), and the state cheat is the perception ceiling. Both are reported.

## 2026-11-25 · Entry 4: realism audit (what could inflate results)

Prompted by advisor review. Simplifications compared with real souls-like fights, and how each will be handled:

| Simplification | Real games | Risk to results | Plan |
|---|---|---|---|
| Zero reaction latency: the agent sees the cue the tick it appears | Human visual-motor latency is ~200–250 ms (4–5 ticks) | **High.** Favors any agent that reacts late in the wind-up | Add `latency` ticks to observations; report results at 0 and 4 ticks. Human-comparison runs use 4 |
| Symbolic cues (a token per animation group) | Pixels, camera angle, lock-on, occlusion, partial view | Medium. Perception is assumed solved | State it explicitly. Cue noise and the 6-tick tell stand in for perceptual ambiguity. Stretch goal: a tiny renderer plus a learned cue classifier |
| 1D lane | 3D movement, spacing, rolling *through* attacks, positioning behind the boss | Medium | Keep 1D for the science; note it as a limitation. A 2D arena is a stretch goal |
| One impact tick per move | Multi-hit combos, hitboxes with duration, tracking | Medium | Add a multi-hit combo primitive (impacts on several ticks) in the next environment revision |
| Boss reacts only to distance and phase | Real bosses read player inputs: heal punishes, input reading, aggro | **High** for transfer claims | Add **reactive laws**, e.g. "if the player heals within reach, grab is likely". These are exactly OneLife-style conditional laws, and make the opponent an agent, not a timer |
| Same rules every boss | Different games tune i-frames, stamina, poise | Covered by T4 | — |
| No poise or stagger, no hyper-armor | Trading hits depends on poise | Low–medium | Possible later |
| Deterministic damage | Mostly deterministic in real games too | Low | — |

**Real-game validation path.** Run the *inference* side, the boss model plus the filter, read-only on SoulsGym's ground-truth state for Iudex Gundyr (Dark Souls III, Windows only, needs the purchased game). The test is whether the HSMM boss model predicts real attack timings and whether the slow mind recovers Gundyr's moveset from a few logged fights. This needs the game on the Windows laptop; it is noted as a later milestone.

## 2026-11-25 · Entry 5: reactive laws and reaction latency

- **Built:** two realism fixes from the Entry 4 audit.
  - **Reactive boss laws.** The boss's next-move distribution now also depends on what the player just did (context: heal, attack, other). Each boss draws Poisson(0.8) laws such as "if the player heals, boost the grab/slam row by ×1.5–3". The exact filter conditions on context, and still matches brute force to 1e-9 with laws present.
  - **Perception latency.** The agent sees cues `latency` ticks late. The planner predicts its belief forward by `latency` steps before acting.
- **Ladder** (`scripts/ladder.py`, 40 bosses × 2 attempts = 80 fights each):

| Agent | Latency 0 | Latency 4 (~200 ms) |
|---|---|---|
| State cheat | 100% | 100% (cheat reads true state, so it ignores latency: not a fair ceiling at latency 4) |
| Oracle planner | 68.8% [58–78] | **41.2%** [31–52] |
| Prior-model planner | 43.8% | 16.2% |
| Generic-model planner | 33.8% | 15.0% |

**Findings:**
1. **Human-like latency costs every inference agent about 25–30 points.** The 6-tick tell minus a 4-tick delay leaves about 2 ticks of readable group before the decision.
2. **The value of knowing the boss grows under latency.** Oracle/prior is 1.6× at latency 0 and 2.5× at latency 4. When you cannot react, you must *predict*, and prediction needs a model of this boss. This is the strongest argument yet for the slow mind.
3. Laws barely moved the latency-0 numbers (72.5 → 68.8, within CI). They matter for transfer and for interpretability, not raw win rate.

**Decisions:**
- Latency 4 is the headline setting from now on. Latency 0 is reported as a secondary row.
- **M2 target:** close the 41% (oracle) vs 16% (prior) gap from deaths alone, at latency 4.

## 2026-11-25 · Entry 6: M2, the slow mind learns bosses from deaths

**Built** (`rtb/learner.py`, `rtb/policies.py: MixturePlanner`). Belief = 4 particles, each a full boss program. After every fight:
- **Reweight** each particle by the exact evidence of the fight.
- **Resample** when ESS < N/2.
- **Rejuvenate** with 3 sweeps of:
  - FFBS over all logged fights (the final move of each fight is right-censored);
  - one structure proposal (delete an unused move, split, merge, retype within a visual group, or add a novel move for an unexplained cue group), accepted by exact evidence ratio × structure prior;
  - Gibbs on wind-up medians (grid, exact discretised log-normal likelihood with censoring), phase-2 speed (grid), Dirichlet transition rows, and reactive laws (enumeration).

The planner averages over particles: each rollout samples a program, then a state from that program's filter. Learning takes 1–7 s between fights.

**Tests:**
- FFBS recovers the true move on 100% of ticks when given the true program.
- 3 fights move held-out evidence more than 70% of the way from prior to truth.
- Offline, 6 fights take held-out log-evidence from −788 to −457; the true program scores −442.

**Experiment** (`scripts/learn_curve.py`). 40 training bosses × 12 consecutive attempts, latency 4, 12 rollouts × 45 ticks. All four conditions use the same fight seeds.

| Condition | Attempt 1 | Attempts 2–3 | Attempts 4–12 | Bosses never beaten in 12 |
|---|---|---|---|---|
| Frozen prior (no learning) | 12.5% [5–26] | 13.8% [8–23] | 12.8% [10–17] | 20/40 |
| **Slow mind, mixture** | 12.5% [5–26] | 30.0% [21–41] | **37.2%** [32–42] | 7/40 |
| Slow mind, MAP lesion | 12.5% [5–26] | 25.0% [17–35] | 37.5% [33–43] | 5/40 |
| True-program oracle | 42.5% [29–58] | 51.2% [40–62] | 42.5% [37–48] | 3/40 |

**Findings:**
1. **Learning from deaths closes about 80% of the prior→oracle gap within 3–4 attempts** (12.8% → 37.2% vs 43–44%). Median deaths-to-first-win is 2. Per-boss late win rates correlate with the oracle's at r = 0.77, so what remains is mostly boss difficulty, not learning failure.
2. **Recovered programs are interpretable and mostly right.**
   - The move count is exact for 19/40 bosses and within one move for 34/40.
   - Reactive laws are discovered, e.g. train00 "heal → slam" with the right target.
   - Errors are extra near-duplicate copies (train02 gets two lunges and three slams), which are nearly harmless for acting.
3. **No mixture-over-MAP benefit yet** (37.2 vs 37.5). With 4 particles that agree after a few fights, posterior averaging has little to average. The "averaging beats committing" effect lives *within* a fight (state uncertainty), not across programs, at least at this particle count. This is still open: test more particles, early attempts only, and T3 novelty, where hypotheses genuinely disagree.
4. **The learner now sits near the ceiling, so the ceiling is the bottleneck.** The oracle itself only wins about 44% at latency 4.

**Next (agenda updated):**
- (a) RL² baseline (GRU-PPO, memory across attempts) on the same protocol.
- (b) Empirical-Bayes transfer: fit the hyperprior from the 40 learned training programs, and test whether attempt-1/2 performance on T1–T4 improves (learning to learn).
- (c) Raise the latency-4 acting ceiling: predictive commitment, better continuation policy.
- (d) Generalization splits T2/T3 (novel move) and T4 (rule change) for the slow mind.
- (e) Stronger Occam pressure on duplicate moves.

## 2026-11-25 · Entry 7: RL² baseline, learning to learn, and what the prior is (not) for

**RL² baseline** (`rtb/rl.py`, `scripts/train_rl.py`).
- Setup: GRU-PPO with memory carried across 4-fight trials, trained on fresh Θ* bosses at latency 4, for 1,500 updates (≈5,100 training fights, 25 min).
- Evaluated on the same 40 bosses × 12 attempts: **2.5–7.5% at every attempt**. There is no in-context improvement across attempts, whether sampled or greedy.
- Comparison: the slow mind reaches 37% after 3–4 fights *with zero pretraining*. That is consistent with the DSLE finding (≤0.33% wins for PPO/DQN on real souls bosses).
- Caveat: this is a short run. A 10× longer run is queued as a fairness check before the final claim.

**Learning to learn (empirical Bayes).**
- Step 1: run the slow mind on the 40 training bosses with a *vague* timing prior (all medians 15 ticks, log-sd 1.0). It still learns: attempts 4–12 win 40% [34–46] on the second half.
- Step 2: fit per-primitive hyperparameters from the 40 learned programs (`scripts/fit_prior.py`).

  The fitted means recover Θ* almost exactly:

  | Primitive | Fitted | Θ* |
  |---|---|---|
  | slam | 13.8 | 14 |
  | delayed slam | 25.6 | 26 |
  | sweep | 10.2 | 10 |
  | lunge | 11.9 | 12 |
  | feint | 10.4 | 10 |
  | aoe | 21.5 | 20 |
  | jab | 6.6 | 6 |
  | idle | 12.8 | 12 |

  Fitted sds are 0.29–0.46 against a true 0.35. The hyperprior is learnable from the agent's own fights.
- Step 3: 30 held-out T1 bosses × 8 attempts with each prior:

| Prior | Attempt 1 | Attempts 5–8 | Never beaten |
|---|---|---|---|
| Vague | 10.0% | 35.8% [28–45] | 9/30 |
| Empirical Bayes | 23.3% | 30.8% [23–40] | 10/30 |
| True Θ* | 13.3% | 29.2% [22–38] | 12/30 |

**Finding:** the timing hyperprior barely matters. All rows overlap, and the 23% at attempt 1 for EB is within noise of the others. This is informative, not disappointing: one fight holds 20+ wind-ups, so the likelihood swamps any reasonable timing prior after a single death. *The binding constraint early on is that the belief does not change during attempt 1 at all.* Particle weights move within a fight, but the programs only change between fights.

**New idea → experiment:** *within-fight learning.* `OnlineMixturePlanner` re-runs one rejuvenation sweep every 200 ticks (10 s) on past fights plus the fight so far, then rebuilds its filters by replaying the fight. Humans visibly adapt mid-fight ("oh, he delays that one"). Prediction: attempts 1–2 rise toward the attempt-4 level. Queued on the 40 training bosses. Caveat: simulated time stops while it thinks, about 1 s per refresh.

**Also built:** the T4 self-model (`RulesBelief`).
- Stamina cost, regen and heal duration are read off proprioception.
- The i-frame window is inferred on a grid from every in-reach impact the agent experienced mid-roll: a hit at roll tick τ excludes τ, a clean dodge includes it.
- Planner rollouts imagine with the believed body. Runs comparing a body model fixed at training rules, a learned one and the true one on T4 are in progress.

## 2026-11-25 · Entry 8: T4, a different game's body rules

T4 bosses come from the training distribution, but the *player rules* change, as when moving to another souls-like:
- i-frames go from ticks 2–5 to ticks 1–3;
- dodge cost 15 → 22;
- regen 1.0 → 0.8;
- heal 30 → 40 ticks.

30 bosses × 8 attempts, latency 4. The slow mind learns the boss in every condition. Only the body model differs:

| Body model in the planner's imagination | Attempt 1 | Attempts 5–8 | Never beaten |
|---|---|---|---|
| Fixed at training rules (no self-model learning) | 3.3% | 20.0% [14–28] | 14/30 |
| **Learned self-model** (`RulesBelief`) | 3.3% | **42.5%** [34–51] | 9/30 |
| True rules (oracle body) | 3.3% | 40.0% [32–49] | 6/30 |

**Findings:**
1. **A wrong self-model halves the win rate**, even with a correctly learned boss: 20% vs 40%. The agent's imagined rolls succeed when the real ones fail.
2. **Learning the self-model recovers all of it** (42.5% vs 40% for the oracle body) in 1–2 fights. This is the T4 claim: transfer to a new game's rules by inference, with no retraining.
3. **Identifiability bug, now fixed.** The learned i-frame windows were often (start 0, length 4), which is *observationally identical* to the true (1, 3), because roll tick 0 never coincides with an impact. The grid now starts at 1. Many particles stayed at the training prior (2, 4) through attempt 8: discriminating evidence (an impact landing exactly on roll tick 1 or 4–5) is rare.
4. Hypothesis to test: much of the gain comes from the proprioceptive quantities (dodge cost, regen, heal time), not the i-frames. Ablation queued: learn proprioception only vs i-frames only.

## 2026-11-25 · Entry 9: within-fight learning doubles first-attempt wins (suggestive)

`OnlineMixturePlanner`, 40 training bosses × 12 attempts, same seeds as Entry 6. It runs one rejuvenation sweep every 200 ticks (10 s) on past fights plus the fight so far.

| Condition | Attempt 1 | Attempts 2–3 | Attempts 4–12 | Never beaten |
|---|---|---|---|---|
| Between-fight learning only (Entry 6) | 12.5% [5–26] | 30.0% [21–41] | 37.2% [32–42] | 7/40 |
| **+ within-fight learning** | **25.0%** [14–40] | 27.5% [19–38] | 37.8% [33–43] | 3/40 |

**Findings:**
- On attempt 1, the paired comparison over the same bosses and seeds gives 5 bosses newly beaten and 0 lost (sign test p = 0.06). Boss HP left drops from 0.28 to 0.25.
- Later attempts are unchanged, as expected: by then the between-fight learner already has the program.
- So the early-attempt deficit was "beliefs frozen during the fight", not "bad prior" (cf. Entry 7).
- Caveats:
  - the pairing is only partial, because trajectories diverge after the first refresh;
  - simulated time stops for about 1 s per refresh;
  - n = 40 on attempt 1. Needs a replication on more bosses (T1 with 30 more is queued) before it becomes a claim.

**Interpretation for the paper:** this is the cleanest demonstration of the "slow mind / fast hands" split. Fast hands (the exact filter) track state at 20 Hz. The slow mind revises the *program* on a slower clock, and running that clock inside the fight, not only between deaths, is what lets the agent beat a quarter of bosses on sight.

## 2026-11-25 · Entry 10: generalization splits and the self-model ablation

All runs: 30 held-out bosses × 8 attempts, latency 4.

**T3, a never-seen move (grab: unblockable, its own visual group, absent from the prior's move set).**
- Slow mind: 16.7% on attempt 1, rising to **71.7%** [63–79] on attempts 5–8.
- True-program oracle: 80.0% [72–86].
- Grab was added to the program by the novelty proposal for **30/30 bosses**: 25 after the first fight, the rest by attempt 4.
- The CRP-style "explain an unexplained cue group with a new move" step works as designed. (T3 bosses are easier than training bosses for everyone, since the oracle is at 80%: grab has 1.5 reach and is avoidable by spacing.)

**T2, unseen move combinations.** Slow mind 13.3% → 25.8% [19–34] on attempts 5–8, with 12/30 never beaten. Oracle and frozen controls for T2 are queued; T2 can't be read without them.

**T4 self-model ablation** (re-run with the i-frame identifiability fix):

| Learned part of the body model | Attempts 5–8 | Never beaten |
|---|---|---|
| Nothing (training rules) | 20.0% [14–28] (Entry 8) | 14/30 |
| i-frames only | 15.8% [10–23] | 15/30 |
| Proprioception only (dodge cost, regen, heal time) | 46.7% [38–56] | 6/30 |
| Both | 48.3% [40–57] | 6/30 |

**Finding:** the whole T4 benefit comes from proprioceptive quantities. Learned i-frames add nothing; alone they slightly hurt, since an agent that thinks rolls are safer but has the wrong stamina budget over-rolls. The i-frame window is rarely probed at discriminating ticks. That itself suggests an **active-learning experiment**: deliberately roll early or late against weak attacks to identify the window (information-gain scouting, from the plan).

**Within-fight learning, replication on T1** (30 bosses × 4 attempts, same seeds):

| Condition | Attempt 1 | Attempts 2–3 |
|---|---|---|
| Between-fight only | 13.3% | 15.0% |
| + within-fight | 13.3% | 31.7% |

The attempt-1 doubling from Entry 9 did **not** replicate. The benefit appears in attempts 2–3 instead. Pooled over both experiments (210 fights per condition in attempts 1–3): 26.2% vs 20.0%. **Revised claim:** within-fight learning gives a modest speed-up of early learning, and its timing is not robust. It stays an option, not a headline.

## 2026-11-25 · Entry 11: generalization summary with matched controls

Latency 4. Win rate on the late attempts: 4–12 for train, 5–8 for the test splits. Same seeds within a row.

| Split | What is new | Frozen prior | Slow mind | Oracle | Gap closed |
|---|---|---|---|---|---|
| train (40 bosses) | nothing; same distribution as training | 12.8% | 37.2% | 42.5% | 82% |
| T2 (30) | move combinations never seen together | 11.7% | 25.8% | 34.2% | 63% |
| T3 (30) | a never-seen move (grab) | 13.3% | 71.7% | 80.0% | 88% |
| T4 (30) | the player's body rules | 20.0%* | 48.3% | 40.0%† | ≥100% |

\* T4 "frozen" = slow mind with the body model fixed at training rules. † T4 "oracle" = slow mind with the true body. Gap closed = (slow − frozen) / (oracle − frozen).

**Reading:**
- The model-based learner transfers to every kind of novelty tested.
- T3 shows why structure learning matters: without grab in the program the agent wins 13%, and after the novelty proposal adds it, 72%.
- T2 is the weakest (63%). Hypothesis: unseen combinations include more overhead-group pairs (slam + delayed slam share a wind-up cue), which take longer to separate. Checked: the correlation between the per-boss oracle−learner gap and the number of overhead-group moves is r = 0.14 (n = 30). This is **not supported**; the two bosses with 4–5 overhead moves have large gaps, but most of the gap is spread out. Hypothesis dropped for now.

## 2026-11-25 · Entry 12: M3 intuition net, and the amortization gap as a structural trap

**Ceiling probe** (`scripts/ceiling_probe.py`; oracle, latency 4, 40 bosses × 2):

| Rollouts / horizon | 12 / 45 | 24 / 45 | 48 / 45 | 24 / 90 |
|---|---|---|---|---|
| Win rate | 41.2% [31–52] | 47.5% [37–58] | 50.0% [39–61] | 40.0% [30–51] |

Unlike latency 0 (Entry 3), more rollouts seem to help under latency, presumably because the projected belief is wider and needs more samples. The CIs overlap, so this is a trend, not a result. A longer horizon does not help. The 12-rollout setting used everywhere else is therefore conservative for every agent alike.

**Intuition net** (`rtb/intuition.py`, `scripts/train_intuition.py`).
- Architecture: a 3-layer transformer (d = 96) over run-length cue tokens (about 90 per fight). For each library primitive, including grab, it predicts the move count (0/1/2+) and log wind-up.
- Training: inference compilation on 40k fights simulated from the broad META prior. 20% of those bosses may use the full library. 15 epochs take 74 min on 3 cores.
- Validation: count accuracy 0.80; log-median MAE 0.19, about 20% timing error.
- Speed: one proposal takes 5 ms; one SMC update takes 1.7 s.

**Amortization gap** (`scripts/amortization_gap.py`). After observing **one** fight, the held-out log-evidence per tick on 3 unseen fights, relative to the true program (0 = as good as truth). 30 bosses per row; ± is the s.e.m., about 0.06.

| Bosses | Prior | Net only | SMC from prior | SMC seeded by net | Mixture, resample first | Mixture, rejuvenate then weigh |
|---|---|---|---|---|---|---|
| T1 (in distribution) | −0.20 | −0.21 | **−0.13** | −0.20 | −0.18 | −0.18 |
| T3 (novel move) | −0.70 | −0.29 | −0.43 | −0.28 | −0.28 | **−0.24** |
| OOD (all timings ×1.8) | −0.56 | −0.25 | **−0.13** | −0.25 | −0.25 | −0.20 |

**Findings:**
1. **Amortized intuition wins under *structural* novelty and loses under *parametric* shift.**
   - On T3 the net, which saw grab during pretraining, is far better than SMC after one fight (−0.29 vs −0.43). SMC must first discover grab through the novelty proposal.
   - On timings stretched beyond the meta-prior, SMC from the generic prior is best, and anything seeded by the net is worse.
2. **The mechanism is a structural trap.** Case-by-case diagnosis (6 OOD bosses): the net *under-proposes* moves. It merges two lunges, or drops a slow slam it never saw at that tempo. Local SMC moves can easily *delete* unused moves but cannot easily *add back* a missing copy, so a net-seeded chain stays wrong. Lowering the split threshold from 4 to 2 observed wind-ups did not help.
3. **Mixture proposals only work if weighing happens after rejuvenation.** With the standard reweight → resample → move order, prior-seeded particles are killed before they can adapt; the "resample first" column equals net-only. Rejuvenating all particles first and then weighing them on all logs recovers part of the gap: T3 is best overall, and OOD improves from −0.25 to −0.20. It is still behind pure SMC, because weighing on the single fight that was fitted favours the net's compact programs.

**New ideas from this:**
- (a) **Predictive weighting.** Weigh particles by evidence on the *later* part of a fight after fitting only the *earlier* part (prequential / cross-validated evidence). This penalizes overfit structure.
- (b) A **birth move guided by residuals**: propose a new copy where the filter's surprise (per-tick −log z) is concentrated.
- (c) The amortization gap as the paper's E6 figure: net vs SMC across a novelty axis (structural vs parametric). This matches the plan's prediction that amortization fails under novelty; but the *direction* depends on the kind of novelty, which the plan did not anticipate.

## Agenda (living)

1. ~~Add reaction latency and reactive laws; re-measure at latency 4.~~ Done (Entry 5).
2. **Now, M2:** the slow mind. Resample-move SMC over boss programs from fight logs:
   - FFBS with censoring at death;
   - conjugate wind-up and transition updates;
   - split/merge with a CRP;
   - error-driven rejuvenation.

   Metric: deaths-to-first-win and the win-rate curve over 20 attempts, with CIs, against the prior-model and generic planners.
3. **Hold out boss families** for generalization (new split T2f).
4. PPO and recurrent-PPO baselines (Gym wrapper), and the planner with a linear opponent model (FightingICE-style).
5. **M3:** intuition net; amortization-gap experiment.
6. SoulsGym validation, once the game is available on the laptop.
