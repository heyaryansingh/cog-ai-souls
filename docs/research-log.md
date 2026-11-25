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
