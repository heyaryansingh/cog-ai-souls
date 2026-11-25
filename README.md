# Reading the Boss: code

A laptop-scale souls-like testbed (MiniSouls) and a Bayesian agent that learns bosses from a few deaths.
The design docs are in `docs/`: `capabilities-and-results.md` (v4, the current plan), `architecture-and-training-plan.md`, `refined-design.md`, the original `proposal.md`, and one note per paper in `docs/paper-notes/`.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q tests            # filter == brute-force enumeration (to 1e-9), filter tracks real fights
python scripts/bench_env.py          # simulator speed
python scripts/run_oracle.py         # M1: random / dodge-everything / oracle / prior-model planners
python scripts/learning_gap.py       # how much boss-specific knowledge is worth at the current difficulty
```

## What is here (M0 + first half of M1, 25 Nov 2026)

| File | What it does |
|---|---|
| `rtb/primitives.py` | 9 move primitives (idle, slam, delayed slam, sweep, lunge, feint, jab, AoE, grab) and the 16-token cue vocabulary. The first `tell_ticks` of every wind-up show a generic "tell" cue, then the move's visual group. Slam and delayed slam look identical; lunge and feint look identical until the active window. |
| `rtb/boss.py` | The boss generative program: an explicit-duration HSMM over moves, with log-normal wind-ups, distance-dependent transitions, and a phase-2 speed-up. Includes `theta_star` (the narrow true distribution), `sample_meta_theta` (the broad pretraining prior), `make_splits` (40 training bosses plus T1 to T4), `prior_mean_boss` and `generic_boss` (what partly informed agents believe). |
| `rtb/env.py` | MiniSouls: a 1D lane at 20 ticks/s. The player has HP, stamina, 8 actions, i-frame dodges, blocking and heals; `Rules` are the player-body rules that T4 changes. The boss follows `Boss` exactly, so the model and the simulator cannot disagree. About 190k ticks/s on one core. |
| `rtb/filter.py` | The exact forward filter over (move, stage, ticks elapsed, start phase), about 1k states. It returns the belief and the running log-evidence, which is the quantity the SMC slow mind will use to weight boss hypotheses. About 12k ticks/s with the filter. |
| `rtb/policies.py` | Random, a dodge-everything heuristic, and `OraclePlanner`: model-predictive control that samples the boss state from the filter and imagines 30-tick futures. `model_boss=` makes it plan with a *believed* program instead of the true one. |
| `rtb/run.py` | Runs one fight and returns the outcome, HP, ticks and an optional text log. |
| `tests/test_filter.py` | Brute-force check: enumerate every latent path using the simulator's own representation, and compare it with the filter. They match to 1e-9 in posterior and log-evidence. |

## Results so far

At the default difficulty (between-boss timing spread 0.35, 6-tick tell, damage ×1.4, 4 i-frames, 2 heals), on 10 training bosses with 2 attempts each:

| Agent | Win rate | Notes |
|---|---|---|
| Random | 0% | |
| Dodge-everything heuristic | 0% | |
| Planner with the **true** boss program (oracle) | **85%** | acting ceiling, M1 target is ≥ 90% |
| Planner with the right moves but **typical** timings | 55% | |
| Planner knowing only Θ (generic) | ~30% (8-boss run) | |

**The gap between the oracle and the partly informed planners is the room a learner has.** At the first, easier settings every planner won 100%, including the generic one. So difficulty was raised until boss-specific knowledge mattered. This was the first real finding: the benchmark needs that gap to exist before any learning claim means anything.

## Next steps

1. **Finish M1.** Push the oracle to ≥ 90%: more rollouts, a 45-tick horizon, a smarter continuation that weighs stamina, and EMPA-style "watch before acting." Make Figure 1 (filter correctness against N) and Table 1 (deaths coded by the taxonomy).
2. **M2, the slow mind.** Resample-move SMC over boss programs between attempts:
   - FFBS segmentation with censoring at death;
   - conjugate updates of wind-up medians (log-normal) and transition rows (Dirichlet);
   - detach/attach split/merge with multiple-try proposals;
   - a CRP over moves.

   Go criterion: the true program is in the particle set by attempt 5 on T1.
3. Start the PPO baseline training overnight (needs `gymnasium` + `stable-baselines3`; a Gym wrapper around `MiniSouls` is a small next file).
