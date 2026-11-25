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

## 2026-11-25 · Entry 13: net-seeded learning in play, RL² fairness run, a negative result

**Net-seeded slow mind in actual fights** (`learn_curve.py --mode netmix`).
- After fight 1, half the particles are seeded from the intuition net's proposal and half from the generic prior; weighing happens after rejuvenation.

| Bosses | Condition | Attempt 2 | Attempt 3 | Attempt 4 | Mean, attempts 2–4 |
|---|---|---|---|---|---|
| T3 (30) | slow mind | 40.0% | 46.7% | 66.7% | 51.1% |
| T3 (30) | + net seed | 43.3% | 66.7% | 73.3% | **61.1%** |
| train (40) | slow mind | 30.0% | 30.0% | 42.5% | 34.2% |
| train (40) | + net seed | 27.5% | 37.5% | 27.5% | 30.8% |

- This matches the evidence-level result of Entry 12: amortized intuition speeds up learning when the novelty is a *move the net has seen in pretraining but the prior excludes*, and does nothing in distribution. The effect is small relative to the CIs (n = 90 fights per T3 condition); it needs a larger replication before it becomes a claim.

**Prequential weighting (idea (a) from Entry 12): negative.** Fitting on the first half of the fight and weighing by the conditional evidence of the second half was no better than net-only: OOD −0.255 vs −0.247, T3 −0.250. Half a fight is too little data to fit, so the score is noisy. Dropped.

**RL² fairness run.** Same architecture, 8,000 updates (~29k training fights, 2.4 h):
- greedy: 13.3% [11–17] over all 480 evaluation fights;
- sampled: about 7%;
- no improvement across attempts (attempt 1: 20%; attempt 12: 15%).

5× more training took RL² from about 4% to about 13%, which is *the frozen generic-prior planner's level* (12.8%), but it shows **no in-context adaptation** across attempts. The slow mind, with zero pretraining, reaches 37% after 3–4 fights. Next RL steps, if pursued: longer trials (12 fights) and a larger GRU. Model-free meta-RL is known to need far more tasks (Algorithm Distillation used >1000 tasks and millions of steps).

## 2026-11-25 · Entry 14: residual-guided birth move (idea (b))

- **What it does:** a new structure proposal. Wind-ups that the current program finds very unlikely (p(D) < 0.01 under their move's pmf) propose a new copy of that move, at the outliers' tempo. It is up-weighted 2× among the proposals.
- **Amortization-gap re-run** (held-out log-evidence per tick vs the true program, 30 bosses per row):

| Bosses | SMC from prior, before → after | Net + SMC, before → after |
|---|---|---|
| T1 | −0.130 → −0.118 | −0.201 → −0.199 |
| T3 | −0.431 → −0.452 | −0.279 → −0.259 |
| OOD | −0.128 → −0.118 | −0.253 → −0.245 |

- **Findings:**
  - Birth slightly helps SMC from the prior on T1 and OOD, and is neutral elsewhere.
  - It does **not** release the structural trap. In the diagnosed cases the missing move's wind-ups are absorbed by a sibling move whose median shifts to cover both, so few wind-ups are outliers by the time the birth step looks.
  - The trap is a *mode*: escaping it needs a joint move (birth + reassign wind-ups + re-fit), e.g. a split-merge Gibbs step with restricted reassignment (Jain–Neal style). Added to the agenda.
- **Re-check:** the main train-40 learning curve is being re-run with birth enabled, to confirm it does not hurt.

## 2026-11-25 · Entry 15: active self-identification costs more than it is worth (information ≠ value)

**Birth-move check.** The train-40 × 12 curve re-run with the birth proposal gives attempts 2–3 at 30.0% and attempts 4–12 at 37.2% [32–42]. These are identical to Entry 6, so birth stays in.

**Active self-identification** (T4, 30 bosses × 8). Planner rollouts that start with a roll get a bonus of 15 × (expected bits about the i-frame window) from the first impact landing mid-roll.

| T4 self-model | Correct i-frame window identified (of 30), at attempt 2 / 4 / 8 | Attempts 1–2 wins | Attempts 5–8 wins |
|---|---|---|---|
| Learned, passive | 0 / 5 / 11 | 3.3%, 33.3% | 48.3% |
| **Learned, active** | **4 / 10 / 11** | 0.0%, 3.3% | 46.7% |

**Findings:**
1. Information-seeking rolls **double the speed of self-identification** (10 vs 5 bosses with the correct window by attempt 4).
2. They **cost the early fights**: attempt 2 drops from 33% to 3%. Deliberately rolling into attacks to probe the window gets the agent killed, and Entry 10 already showed the i-frame window is worth almost nothing for winning.
3. **Lesson:** an entropy bonus values *information*, not *value of information*. The right quantity is VOI: the expected improvement in the decisions the belief feeds, which here is small for i-frames and large for stamina. This is a clean metareasoning point for the paper, and it argues for VOI-based scouting (estimated by planning under posterior samples of the rules), not an IG bonus. Added to the agenda.

## 2026-11-25 · Entry 16: the "trap" is Occam on one fight, not a search failure

**Regroup move** (Jain–Neal-flavoured joint split-merge). It re-clusters every wind-up of one visual group into c tempo clusters and rebuilds the group as c moves, adding or removing several at once. Accepted by exact evidence.

| Bosses | SMC from prior | Net + SMC |
|---|---|---|
| T1 | −0.157 (was −0.118) | −0.201 |
| T3 | −0.504 (was −0.452) | −0.264 |
| OOD | −0.113 (was −0.118) | −0.237 (was −0.245) |

It does not rescue net-seeded chains, and it *hurts* SMC from the prior on T1/T3, because it takes structure-proposal slots away from delete/add. **Disabled** (`USE_REGROUP = False`), with the code kept.

**Reinterpretation.** After two failed "escape the trap" moves (birth, regroup), the evidence points elsewhere. With *one* fight, a compact program like the net's is a genuinely competitive explanation of what was seen: few wind-ups per move, so splitting is not yet supported by the evidence. The generic-prior chain wins on held-out fights *because it keeps unused, plausible moves alive*: delete is only probabilistically accepted. Those extra moves act as a hedge, covering behaviours the boss has but did not show in fight 1.

That is "averaging beats committing" at the level of **structure**, not state. The posterior should keep, and the planner should prepare for, moves it has not yet seen. That leads to:
- **Prediction:** an Occam-heavier learner (higher deletion pressure) does better on the observed fight and worse on held-out fights and win rate early on.
- **Design rule:** keep a "dormant move" reservoir: unused library moves with low but non-zero transition mass, rather than deleting them.
- **Test:** compare delete-prior strengths on early attempts (queued).

## 2026-11-25 · Entry 17: the hedging prediction is falsified

Test of Entry 16's prediction (train 40 × 4 attempts, same seeds):

| Deletion prior | Attempt 1 | Attempt 2 | Attempt 3 | Attempt 4 | Attempts 2–4 | Program size error at attempt 4 (moves) |
|---|---|---|---|---|---|---|
| Default (log 6 per move) | 12% | 30% | 30% | 45% | 35.0% [27–44] | +0.42 |
| +3 extra toward deletion | 12% | 30% | 30% | 45% | 35.0% [27–44] | +0.42 |
| Never delete (dormant reservoir) | 12% | 18% | 32% | 40% | 30.0% [23–39] | +2.40 |

**Findings:**
- Deletion of unused moves is already *saturated* at the default prior: the +3 run is identical down to every fight, so every proposed delete was already accepted.
- Keeping all unused moves alive makes the program 2.4 moves too large and **hurts** early learning (attempt 2: 18% vs 30%).
- So the hedging explanation of Entry 16 is wrong for acting. Dormant moves dilute the transition model and make the planner over-prepare for attacks that never come.
- The OOD gap between net-seeded and prior-seeded chains (Entry 12) therefore remains *unexplained*. Remaining candidates:
  - (i) the net's proposed *medians*: no rejuvenation of medians happens before selection beyond 3 Gibbs sweeps;
  - (ii) idle-median or transition initialisation;
  - (iii) phase-2 speed.

  Next diagnostic: swap one component at a time between net and prior initialisations.
- Program-size numbers are reported as MAP move count minus true move count.

## 2026-11-25 · Entry 18: the fix is a *superset* proposal (structure from the net, coverage from the prior)

**Component swap** (`scripts/swap_diag.py`). After one fight, held-out log-evidence per tick vs the true program, 30 bosses per row, s.e.m. 0.03–0.08.

| Initial program for SMC | T1 | OOD (timings ×1.8) | T3 (novel move) |
|---|---|---|---|
| Generic prior | −0.118 | **−0.118** | −0.452 |
| Net proposal | −0.199 | −0.245 | −0.259 |
| Net structure + prior medians | −0.220 | −0.232 | −0.247 |
| Prior structure + net medians | −0.128 | −0.134 | −0.472 |
| Net, 12 sweeps instead of 3 | −0.184 | −0.202 | −0.238 |
| 50/50 net/prior particles, weighed after rejuvenation | −0.180 | −0.200 | −0.229 |
| **Superset: net moves + every library move it omitted** | **−0.115** | −0.127 | **−0.156** |

**Findings:**
1. **The net's move *structure*, not its timings, explains both its T3 win and its OOD loss.** Swapping medians changes almost nothing. Swapping structure moves each row to the other side.
2. **Slow mixing, not a hard trap.** 12 sweeps recover about a third of the OOD gap. Adding moves is slower than deleting them, so an under-complete start is costly, while an over-complete start is cheap to prune. (Entry 17: an over-complete program that *never* prunes is costly too.)
3. **Superset proposal = best of both.**
   - It takes the net's structure, including moves the prior excludes (grab) and duplicate copies, and adds one copy of every library move the net left out. Ordinary SMC then deletes what is unused.
   - It matches the prior on T1/OOD and is far better than anything on T3 (−0.156 vs −0.259 for net-only and −0.452 for prior-only).
   - Design principle: **amortized proposals should over-propose structure and let inference prune.** Proposals are cheap to be wrong in the direction inference can fix quickly.

**In play** (`--mode netsup`), attempts 2–4:

| Bosses | Slow mind | + net (50/50 mixture) | **+ net superset** |
|---|---|---|---|
| T3 (30) | 51.1% | 61.1% | 60.0% (attempt 2: **53%** vs 40%) |
| train (40) | 34.2% | 30.8% | 33.3% |

Superset seeding helps on novel-move bosses and costs nothing in distribution, so it becomes the default M3 configuration. The win-rate gains are within CIs at n = 90; the evidence-level result is the robust one.

## 2026-11-25 · Entry 19: ablations, and the FightingICE-style baseline

Lesion flags (`RTB_LESION`) switch off one part of the slow mind's learning. Train 40 × 6 attempts, same seeds:

| Learner | Attempt 2 | Attempts 3–6 |
|---|---|---|
| Full slow mind | 30% | 33.8% [27–41] |
| − laws (no reactive-law inference) | 25% | 40.6% [33–48] |
| − timing (wind-ups fixed at prior) | 22% | 30.6% [24–38] |
| − structure (move set fixed to generic) | 20% | 30.0% [23–38] |
| − transitions − laws | 12% | 26.9% [21–34] |
| **Counts-only opponent model** (− timing − structure − laws; FightingICE-style move-frequency learning) | 20% | **16.9%** [12–23] |
| Frozen (no learning) | 12% | 13.8% [9–20] |

**Findings:**
1. **A move-frequency opponent model (the standard fighting-game AI approach) buys almost nothing:** 16.9% vs 13.8% frozen. What matters is learning *when* and *what*: timing and structure. "Which move comes next" alone is not enough.
2. **No single component is essential, but they are synergistic.** Each single lesion costs 3–7 points; removing timing, structure and laws together costs 17. Timing and structure partly substitute for each other: an extra copy of a move can absorb a tempo, and a wider timing can absorb a missing copy.
3. **Laws do not help win rate here.** Without law inference the agent does slightly better (40.6 vs 33.8, CIs overlap).

   Law recovery at attempt 12:

   | Law contexts | Count |
   |---|---|
   | Correctly found | 17 of 28 |
   | Missed | 11 |
   | Spurious | 3 |

   Precision is 85% and recall 61%. A plausible cost: contexts that carry a law are excluded from the transition counts, which throws away data. Laws matter for *interpretability* and for punishing heals, but the planner rarely heals into a law. To test: law inference with pooled counts.

## 2026-11-25 · Entry 20: net-superset seeding replicates (p = 0.007)

T3 (never-seen move) re-drawn with 80 bosses × 4 attempts (`--n_test 80`). These are new bosses, independent of Entry 18.

| Condition | Attempt 1 | Attempt 2 | Attempt 3 | Attempt 4 | Attempts 2–4 |
|---|---|---|---|---|---|
| Slow mind | 11% | 39% | 55% | 71% | 55.0% [49–61] |
| **+ intuition-net superset seed** | 11% | **50%** | **62%** | **81%** | **64.6%** [58–70] |

- Paired over the same boss, attempt and seed: 45 fights won only with the net and 22 only without it (sign test **p = 0.007**).
- This is the first M3 claim that holds at the win-rate level. Amortized structural intuition, used as an *over-complete* proposal that SMC prunes, speeds up adaptation to a novel move by about 10 points per attempt. In distribution it costs nothing (Entry 18).

**Pooled law counts (Entry 19 follow-up): no gain.** Attempts 3–6 win 30.0% [23–38] vs 33.8% for the default. The law-context exclusion is not what makes laws unhelpful; dropped.

## 2026-11-25 · Entry 21: a playable human version (instrument for the human baseline)

`web/minisouls.html` is a faithful JavaScript port of MiniSouls: the same rules, step order, wind-up pmfs, transitions, laws and cue noise, at 20 ticks/s. It uses the first 8 training bosses, exported in `web/bosses_train0-7.json`, and is published as a private artifact.
- **Display:** cues are drawn as poses. The agent's symbolic cues map one-to-one to poses, and a toggle shows move names.
- **Controls:** keyboard, plus touch buttons on phones.
- **Ledger:** per-boss attempts, wins and first-win attempt, shown next to the slow-mind agent's first win from Entry 14. Results can be copied as JSON.

The protocol for a human comparison is to fight each boss until the first win, or for 12 attempts, and report deaths-to-first-win. Humans bring their own ~200–250 ms latency, which matches the agent's latency-4 setting. Agent reference values for bosses 1–8: first win on attempts 8, 2, 9, 2, 9, —, 7, —.

## 2026-11-25 · Entry 22: within-fight learning, now replicated at n = 80

T1 re-drawn with 80 new bosses × 3 attempts, same seeds in both conditions:

| Condition | Attempt 1 | Attempt 2 | Attempt 3 |
|---|---|---|---|
| Between-fight learning only | 8.8% | 22.5% | 30.0% |
| **+ within-fight learning** (re-learn every 10 s) | **21.2%** | 30.0% | 31.2% |

- Attempt 1, paired by boss and seed: 12 fights won only with within-fight learning, 2 only without it (sign test **p = 0.013**).
- Attempts 1–3 together: 34 vs 17 (p = 0.024).

**All three experiments on attempt 1:**

| Experiment | Between-fight only | + within-fight |
|---|---|---|
| E9, train (40) | 5/40 | 10/40 |
| E10, T1 (30) | 4/30 | 4/30 |
| E22, T1 (80) | 7/80 | 17/80 |
| **Pooled** | **16/150 (10.7%)** | **31/150 (20.7%)** |

The E10 null now looks like a small-sample miss.

**Claim:** running the slow mind's clock inside the fight roughly **doubles first-encounter wins**. Its benefit shrinks as between-fight learning catches up by attempt 3. This upgrades Entry 10's "modest, not robust" verdict.

## 2026-11-25 · Entry 23: combining the two improvements on T3 (n = 80)

| Condition (80 T3 bosses, same seeds) | Attempt 1 | Attempt 2 | Attempt 3 | Attempt 4 | Attempts 2–4 |
|---|---|---|---|---|---|
| Slow mind | 11% | 39% | 55% | 71% | 55.0% |
| + net superset (E20) | 11% | 50% | 62% | 81% | 64.6% |
| + net superset + within-fight learning | 15% | 51% | 72% | 64% | 62.5% |
| True program (oracle) | 75% | 79% | 76% | 76% | 77.1% |

**Findings:**
- On T3 the two improvements **do not stack.** Within-fight learning adds +4 points on attempt 1 (4 vs 1 paired, n.s.) and nothing later.
- On T3 the first fight is dominated by the unknown grab move. Within-fight learning *can* add grab mid-fight, but the net superset can only act after fight 1, and grab kills fast.
- The gap to the oracle on attempt 1 (11–15% vs 75%) is the clearest picture of what an unseen move costs, and of why first-encounter performance needs prior knowledge, not just fast learning.
- Best configuration for the paper: slow mind + net superset seed, with within-fight learning reported separately (E22: it doubles first-encounter wins on T1).

## 2026-11-25 · Entry 24: why the latency-4 ceiling is low, and a failed fix

**Death taxonomy at latency 4** (`scripts/death_taxonomy.py 4`): true-program oracle, 16 bosses, 109 hits taken.

| What the player was doing when hit | Hits |
|---|---|
| Blocking | 55 |
| Idle, no defensive action | 22 |
| Healing | 18 |
| Heavy attack | 8 |
| Light attack | 2 |
| Mistimed dodge | **2** |

At latency 4 the planner almost never rolls. It hedges by blocking, which costs 25% chip damage plus a stamina drain, so a fight becomes a slow bleed. The 80% "belief > 0.5" share means the agent usually knows *which* move is coming; it cannot time it.

**Failed fix, the "humble" imagined player.**
- Hypothesis: rollouts use a continuation player that knows the exact remaining wind-up, so imagined rolls always succeed. That should bias the planner, though if anything it would favour rolls. The humble player in imagination instead (i) cannot read the move during the tell + latency and (ii) predicts the wind-up end by its conditional median.
- Result at latency 4: **23.8% wins vs 41.2%**, with a quarter of fights timing out. The humble imagined player never attacks during uncertainty, so the planner learns that "nothing good happens" and turtles.
- Reverted to the default (`humble=False`).
- **Lesson:** the planner's optimism about its own future is *useful*, because it keeps the agent aggressive. A properly belief-consistent rollout needs a belief-aware *policy*, e.g. the planner itself recursively or a distilled reflex policy, not a timid heuristic.

**Next ceiling idea:** a predictive commitment roll. Roll at the predicted impact minus the i-frame start, using the conditional wind-up distribution: choose the roll tick that maximizes P(impact ∈ i-frames). With 4 i-frame ticks and wind-up sd of about 10%, this should beat blocking on most moves.

## 2026-11-25 · Entry 25: the planner's optimism is load-bearing

Variant: limit **only the roll timing** of the imagined player to prediction (conditional median wind-up); everything else stays omniscient. Oracle, 40 bosses × 2:

| Imagined player | Latency 4 | Latency 0 |
|---|---|---|
| Omniscient (default) | 41.2% | 68.8% |
| Roll timing by prediction | 20.0% (many timeouts) | 41.2% |
| Fully humble (E24) | 23.8% | — |

Any realism added to the imagined player's defence makes imagined futures dangerous. The planner then stops attacking, and fights time out.

**Conclusion:** with a one-step lookahead over a fixed heuristic continuation, optimism about one's own future skill is what keeps the agent aggressive enough to win. This mirrors the "optimism in the face of uncertainty" principle, here at the level of self-model.

Raising the latency-4 ceiling properly needs a better continuation policy, not a more pessimistic one. Candidates:
- a reflex policy distilled from planner rollouts (DAgger, as in the plan's M4);
- a two-level search.

Defaults are unchanged (`humble=False`).

## 2026-11-25 · Entry 26: averaging over programs vs committing, at 16 particles

Train 40 × 6 attempts with 16 particles (4 in all earlier runs):

| Planner over the particle set | Attempts 2–3 | Attempts 4–6 | Paired, attempts 2–6 (only mixture won / only MAP won) |
|---|---|---|---|
| Bayesian model averaging (mixture) | 31.2% | 32.5% | 35 |
| MAP lesion (best particle only) | 27.5% | 30.8% | 30 |

**Findings:**
- Even with 16 particles, averaging over boss programs gives only a small, non-significant edge (+2.5 points; 35 vs 30 discordant fights).
- 16 particles are not better than 4 (Entry 19 full system, attempts 3–6: 33.8%).
- **Conclusion for the paper:** across-program uncertainty collapses within a few fights, so committing to the best program costs little. The "averaging beats committing" effect in this domain lives in *within-fight state* uncertainty (the exact filter's belief over the current move and wind-up), which every condition uses.
- A cleaner test would remove state averaging: plan from the filter's MAP state only. Added to the agenda.

## 2026-11-25 · Entry 27: averaging over *state* is where averaging pays

State-averaging lesion. The true-program oracle plans from the single most probable filter state, not 12 states sampled from the belief. The remaining wind-up is still sampled inside rollouts. 40 bosses × 2, same seeds:

| Planner | Latency 4 | Latency 0 |
|---|---|---|
| Averages over sampled states (default) | 41.2% [31–52] | 68.8% [58–78] |
| Commits to the MAP state | 35.0% [25–46] | 61.3% [50–71] |

**Findings:**
- Committing to the MAP state costs 6–8 points at both latencies. That is consistent in direction, but each comparison alone is within CI.
- Compare program averaging, which costs 2.5 points or less (E6, E26).
- **Synthesis:** in this domain, averaging pays at the level of *what is happening right now*: which move is winding up and how far along it is. It barely pays at the level of *what kind of boss this is*, because evidence about the boss accumulates fast across fights, while state evidence is renewed every few ticks and stays ambiguous. This is the project's version of "averaging beats committing", now localized.

## 2026-11-25 · Entry 27b: review follow-up, does within-fight learning hurt late attempts on T3?

This follows the external review point that E23's attempt 4 dropped from 81% to 64%. Paired by boss and seed, net superset alone vs net superset + within-fight learning, 80 T3 bosses:

| Attempt | Only superset won | Only +online won | Sign test p | MAP K minus true K (superset / +online) |
|---|---|---|---|---|
| 1 | 1 | 4 | 0.38 | 2.41 / 2.31 |
| 2 | 6 | 7 | 1.00 | 2.08 / 1.75 |
| 3 | 7 | 15 | 0.13 | 1.31 / 1.38 |
| 4 | 17 | 3 | **0.003** | 1.05 / 0.81 |

Per boss over attempts 2–4 (Wilcoxon on win counts), p = 0.67: superset better on 20 bosses, +online better on 15.

**Findings:**
- The attempt-4 drop is significant on its own (p = 0.003, still p < 0.05 with a Bonferroni correction over 4 attempts), but attempt 3 leans the other way. The boss-clustered test over attempts 2–4 shows no difference.
- **Spurious extra moves do not explain it.** With within-fight learning the MAP program has *fewer* surplus moves at attempt 4 (0.81 vs 1.05).
- **Status:** an unexplained late-attempt cost, so E23's conclusion stands as "does not stack" rather than "harms". It is on the agenda as a diagnostic: log particle sets after each mid-fight refresh, and add a sweeps = 0 refresh control (rebuild filters without re-learning).
- **Labelling (review point 3):** E22 (pre-registered replication, n = 80, p = 0.013) is the confirmatory within-fight result. The pooled 10.7% → 20.7% table mixes E9 training bosses and test bosses and was assembled after E10's null, so it is exploratory.

## 2026-11-25 · Entry 28: a distilled reflex matches the 48-rollout planner at 1/160 of the compute

**Idea ("fast hands, learned").** The planner simulates 8 actions × n rollouts × 30 ticks for every decision. The reflex (`rtb/reflex.py`) is a 2×128 MLP over a *boss-agnostic summary of the belief*:
- **threat clock:** P(the next in-reach impact lands in t ticks), for t = 1..20 and 21+;
- **attack clock:** the same, ignoring reach;
- **punish clock:** P(the boss stays helpless for t more ticks);
- **scalars:** unread wind-up mass, idle mass, expected threat damage, unblockable share, reach margin, and the body state.

Because the inputs are a function of the filter's belief, not of a boss identity, one reflex can act on any boss program, including programs the slow mind learns at test time.

**Training.** DAgger (Ross et al. 2011). The teacher is the true-program planner with 48 rollouts, at latency 4, on the 40 training bosses, with training seeds disjoint from evaluation seeds. Each iteration runs 160 fights in which the current reflex acts (the teacher acts with probability β = 1, 0.5, 0.25, 0, …). Every decision is labelled with the teacher's 8 Q-values, the data are aggregated, and the reflex is refit with soft targets softmax(Q/2). Six iterations produced 93k labelled decisions in 40 minutes.

DAgger curve (n = 80 per point): reflex 31% → 51% → 61% → 49% → 61% → 53%. The β = 1 (pure behavioural cloning) reflex is the worst, as DAgger theory predicts.

**Paired evaluation.** 40 train bosses × 6 new seeds = 240 fights, true program, latency 4:

| Agent | Win | s / fight | vs 12-rollout planner (only A won / only B won) |
|---|---|---|---|
| Planner, 12 rollouts (all earlier experiments) | 46.2% [40–53] | 2.78 | – |
| Planner, 48 rollouts (teacher) | 56.7% [50–63] | 9.98 | 57 / 32, p = 0.011 |
| Reflex, DAgger iteration 2 | 55.8% [50–62] | 0.07 | 54 / 31, p = 0.017 |
| Reflex, iteration 4 | 61.7% [55–68] | 0.06 | 63 / 26, p = 0.0001 |
| Reflex, iteration 5 | 52.5% [46–59] | 0.06 | 52 / 37, p = 0.14 |

**Findings:**
- **The reflex reaches teacher quality.** The mean over the three checkpoints is 56.7%, identical to the teacher, at about 1/160 of the compute per fight.
- Planning noise is a real cost. Quadrupling rollouts gains 10 points (p = 0.011), so the 12-rollout planner used in every earlier experiment was leaving wins on the table. A reflex trained on noisy labels effectively averages Q over many similar belief states.
- **Checkpoint variance is large** (52–62%). Picking iteration 4 would be selection on the evaluation set, so the honest claim is "matches the teacher". Next: an ensemble of checkpoints, a metareasoning gate (plan only when the reflex is unsure), held-out T1 bosses, and the reflex acting over the *learned* mixture on T3. T3 includes grab, which the reflex never saw in training.
- **Relation to R1 (E25 confound):** the reflex is trained on its own state distribution, so it sidesteps the "rollouts don't model the executed policy" issue on the acting side. The planner's rollouts still use the optimistic continuation.

## 2026-11-25 · Entry 29: where the reflex wins, and where it loses

The ensemble (iterations 2 + 4 + 5, log-probs averaged) was evaluated in four settings, all at latency 4.

**1. Ensemble, train bosses, true program, 240 fights** (same seeds as E28):

| Agent | Win | vs 12-rollout planner (only A / only B) |
|---|---|---|
| Ensemble reflex | 60.8% [55–67] | 63 / 28, p = 0.0003 |

**2. Gate on action uncertainty (plan when the reflex's top probability is below 0.5): it fails.** The reflex is "unsure" on 79% of decisions, which then go to the 48-rollout planner, and the hybrid gets 52.1%. The reflex's soft targets are deliberately flat between near-tied actions, so its confidence does not measure how hard a decision is.

**3. Held-out T1 bosses** (80 new bosses × 2 seeds, true program):

| Agent | Win |
|---|---|
| Planner, 12 rollouts | 48.8% [41–56] |
| Planner, 48 rollouts | 45.0% [37–53] |
| Ensemble reflex | 48.1% [41–56] (27 / 28 vs planner-12) |

On held-out bosses more rollouts buy nothing, and the reflex equals the planners at 1/33 of the compute. The train-boss gains in E28 were partly specific to those bosses (more rollouts help there, p = 0.011, but not here). **Honest claim: the reflex reproduces planner quality on unseen bosses, it does not exceed it.**

**4. Reflex acting over the *learned* mixture on T3** (net superset seed, 80 bosses × 4 attempts, paired with E20/E23's planner run):

| Attempt | Planner | Reflex | Only reflex won / only planner won | Deaths (planner / reflex) |
|---|---|---|---|---|
| 1 | 11% | **41%** | 27 / 3, p < 10⁻⁵ | 71 / 47 |
| 2 | 50% | 52% | 19 / 17 | 40 / 38 |
| 3 | 62% | 48% | 12 / 24, p = 0.065 | 30 / 42 |
| 4 | **81%** | 48% | 4 / 31, p < 10⁻⁵ | 15 / 42 |

**Findings:**
- **A clean crossover.** When the model is wrong (attempt 1: the prior has no grab), the reflex nearly quadruples wins. Once the slow mind has learned the boss, the planner exploits the model and the reflex plateaus near 48%.
- **Interpretation:** the planner trusts its model completely, because rollouts never imagine an unmodelled move. The reflex was trained on the teacher's choices *across* many bosses and belief states, so it encodes generic defensive habits (stamina reserve, spacing) that are robust to model error. It cannot exploit a specific boss the way search can.
- This is the dual-process story in miniature: habits for the unfamiliar, deliberation for the familiar. Note this is the *opposite* of the classic "habits for the familiar" arbitration (Daw et al. 2005). Here the habit is the robust generic prior, and the model becomes trustworthy only after learning.
- **Next (running):** arbitrate on *model adequacy*, not action uncertainty. Plan while the mixture explains the cue stream; act on reflex while the last 40 ticks hold ≥ 4 cues the mixture found implausible (predictive p < 0.01). One noisy cue is common, a run of them is an unmodelled move. Also: reflex on attempt 1 only, and the same tests on T1 (no novel move, only timing misfit).

## 2026-11-25 · Entry 30: the reflex over the learned model dominates on T1; arbitration tests

All runs are 80 bosses, latency 4, paired by seed with the matching planner run. Learned model means the slow mind's mixture.

**T1 (new bosses, timings differ from the prior):**

| Attempt | Planner (E22 baseline) | Reflex | Only reflex won / only planner won |
|---|---|---|---|
| 1 | 8.8% | **30.0%** | 17 / 0, p < 10⁻⁴ |
| 2 | 22.5% | **42.5%** | 22 / 6, p = 0.004 |
| 3 | 30.0% | **46.2%** | 16 / 3, p = 0.004 |

The true-program planner on T1 bosses gets 48.8% (E29). **By attempt 3 the reflex acting on a learned model is within 3 points of the planner given the true program.**

**T3 (novel grab move), arbitration variants** (win %; superset seed in all):

| Attempt | Planner | Reflex | Reflex on attempt 1 only | Surprise gate (≥ 4 odd cues / 40 ticks) |
|---|---|---|---|---|
| 1 | 11% | 41% | 41% | 19% (reflex on 50% of decisions) |
| 2 | 50% | 52% | 55% | 55% |
| 3 | 62% | 48% | 62% | 70% |
| 4 | 81% | 48% | 66% | 65% |
| Mean, attempts 1–4 | 51.2% | 47.5% | **56.2%** | 52.2% |

**Findings:**
- **The reflex is robust to model error; the planner exploits a correct model.** T1's learned timing models stay imperfect over 3 fights, and the reflex wins everywhere. T3's learned models become nearly right (grab is found in 74–78 of 80), and planning wins late.
- **The surprise gate catches only one kind of model error.** It detects unmodelled cue *types* (grab) but not timing misfit, so it fires too rarely to help on T1 (15% vs reflex 30%). On T3 it hands over control too late in the first fight.
- The planner's 81% at T3 attempt 4 looks high next to its own attempt 3 (62%) and both hybrids (65–66%). Part of the late gap is probably seed luck.
- **Next (running): habit as a prior for deliberation.** Act by argmax_a Q(a) + λ·log π_reflex(a), as in the AlphaZero / PUCT idea, with λ = 3 and 10. It keeps planning's exploitation but breaks ties toward robust habits.

## Agenda (living)

Done: latency + laws (E5); M2 slow mind (E6); RL² baseline (E7, E13); empirical Bayes (E7); T4 self-model (E8, E10); T2/T3 generalization (E10–11); M3 intuition net + amortization gap (E12, E18); active scouting (E15).

Next, in priority order:
1. **Linear opponent-model planner baseline** (FightingICE-style: per-move frequency counts with no timing model). This shows how much the *timing/HSMM* structure buys over a simpler opponent model.
2. **VOI-based scouting** in place of the IG bonus (E15): plan under posterior samples of rules or program and measure the value of resolving each uncertainty.
3. **Replications at n ≥ 80 per cell**: net-superset done (E20, p = 0.007). Within-fight learning is done too (E22, p = 0.013).
4. **Hold out boss families** (T2f: a primitive pair always co-occurring in training appears split).
5. **Multi-hit combo primitive** (realism audit, E4), then re-check the ladder.
6. **Human comparison:** the playable version is built (E21). Next, collect human ledgers.
7. SoulsGym validation, once the game is available on the laptop.

**From the external review (Nov 25, 18:25), added to the queue:**
- R1. *E25 confound.* The humble variants changed only the imagined player, so rollouts stopped modelling the executed policy (strategy fusion). Re-run with the commitment roll in both the real and imagined agent, imagined rolls scored as expected success over the wind-up pmf; latency 4, n = 80, paired against 41.2%. Note: the DAgger reflex (E28) is a second route around this, since it is trained to imitate its own execution.
- R2. *Within-fight late-attempt cost* (E27b): particle logging and a sweeps = 0 refresh control; asynchronous refresh.
- R3. *Block realism:* 55 of 109 hits landed while blocking (E24). Compare chip damage and stamina drain with real shields: a 100% physical shield takes no chip, and its real cost is guard break.
- R4. *Superset controls:* a no-net full-library seed; hold one primitive out of the net's training library for a true novelty test; check prior/q importance weights on net-seeded particles.
- R5. *Statistics:* boss-clustered tests (per-boss Wilcoxon or mixed effects) for every close call; survival analysis and counterbalanced order for the human study.
- R6. ~~Verify that RTB_DELETE_BIAS reaches the sampler.~~ Done: it is added to the delete proposal's log prior (learner.py:325), which enters the MH acceptance (learner.py:521). With -inf, deletes are never accepted.
