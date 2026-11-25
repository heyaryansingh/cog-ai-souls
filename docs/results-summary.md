# Reading the Boss: results summary (as of 2026-11-25)

Every number comes from `docs/research-log.md`, which names the entry, script and seeds. Unless stated, results are at **latency 4** (~200 ms, human-like) and win rates carry 95% Wilson CIs.

## Headline: learning a boss from deaths

40 training bosses × 12 consecutive attempts, same seeds in every row (Entry 6).

| Agent | Attempt 1 | Attempts 4–12 |
|---|---|---|
| Planner with the generic prior, no learning | 12.5% | 12.8% [10–17] |
| **Slow mind** (SMC over boss programs between deaths) | 12.5% | **37.2%** [32–42] |
| Planner given the true boss program (oracle) | 42.5% | 42.5% [37–48] |
| RL² (GRU-PPO, memory across attempts, ~29k training fights) | 20% | 13.3% overall, flat (E13) |

Further results for the slow mind:
- It closes **~80% of the prior→oracle gap within 3–4 attempts**. Median deaths-to-first-win is 2.
- The programs it recovers are readable: move count exact for 19/40 bosses and within one move for 34/40, and reactive laws such as "heal → slam" are discovered.
- With zero pretraining it beats a model-free meta-learner given ~29k training fights.

## Generalization (Entry 11)

| Split | What is new | Frozen | Slow mind | Oracle | Gap closed |
|---|---|---|---|---|---|
| T2 | unseen move combinations | 11.7% | 25.8% | 34.2% | 63% |
| T3 | a never-seen move (grab) | 13.3% | 71.7% | 80.0% | 88% |
| T4 | player body rules (another game) | 20.0% | 48.3% | 40.0% | ≥100% |

- **T3:** the novelty proposal adds grab to the program for 30/30 bosses, 25 of them after a single fight.
- **T4:** a learned self-model recovers all of the loss caused by a wrong body model. The ablation shows the gain comes from proprioception (stamina, heal time), not i-frames.

## Cognitive-science findings

1. **Prediction replaces reaction under latency.** Human-like delay costs the true-model planner 30 points (69% → 41%) but makes boss knowledge 2.5× more valuable (E5).
2. **The timing prior barely matters; the structure prior does.** An empirical-Bayes hyperprior fitted from the agent's own 40 learned bosses recovers the true one, but one fight swamps any reasonable timing prior (E7). Structure (which moves exist) is where priors and intuition pay off (E12, E18).
3. **Amortized intuition helps under structural novelty and hurts under parametric shift.** The intuition net beats SMC on a novel move but under-proposes moves on out-of-range timings. The cause is the net's *structure*, and SMC repairs it slowly, because adding moves mixes more slowly than deleting them. **Over-propose, then prune:** a superset proposal (the net's moves plus everything it omitted) gets the best of both (E18). In play on 80 new novel-move bosses it lifts attempts 2–4 from 55.0% to 64.6% (paired p = 0.007, E20).
4. **Information is not value of information.** Information-seeking rolls doubled the speed of self-identification but got the agent killed in early fights, for a parameter that barely affects winning (E15).
5. **Averaging over programs ≈ committing to the MAP program**, once a few fights are in (E6). Uncertainty that matters for acting is *within-fight state* uncertainty. Keeping dormant moves alive for hedging hurts (E17).

## Ablations (E19)

A move-frequency opponent model (FightingICE-style) reaches only 16.9%, against 13.8% frozen and 33.8% for the full slow mind. Learning *timing* and *structure* carries the gains, and the two partly substitute for each other. Reactive-law inference does not raise win rate (law precision 85%, recall 61%).

## Honest negatives and caveats

- Within-fight learning: the attempt-1 doubling (E9) did not replicate; there is only a modest early speed-up (E10).
- Prequential weighting, the residual-birth move and the regroup split-merge move did not help (E13, E14, E16).
- Environment simplifications are listed in the E4 realism audit: 1D lane, symbolic cues, a single impact tick per move.
- The state-cheat ceiling ignores latency.
