# Stafford & Dewar 2014: Tracing the trajectory of skill learning with a very large sample of online game players

## 1. Citation and access
- Tom Stafford, Michael Dewar. "Tracing the trajectory of skill learning with a very large sample of online game players." *Psychological Science* 25(2):511–518, 2014. doi:10.1177/0956797613511466.
- The journal version is paywalled. I read the open author postprint at https://eprints.whiterose.ac.uk/id/eprint/83582/1/stafford_and_dewar_postprint.pdf (White Rose eprints), through a text-extraction tool. That covered the main text and figure captions. The postprint may differ slightly from the published version; one cross-reference in it is broken ("Figure Error").

## 2. Core idea and method
- **Data.** Observational analysis of play logs from *Axon*, a browser game made for the Wellcome Trust.
  - Players click rapidly on targets to grow a neuron. Score is total branch length.
  - The game demands little strategy and is mostly fast perception, decision and motor skill.
- **Sample.** 854,064 machine-ID "players", 14 Mar – 13 May 2012, more than 3.5M plays. Most IDs played once; some played up to 1,000 times.
- **There is no computational model.** Analyses are descriptive:
  - learning curves of mean score against play number, split by groups of maximum score;
  - spacing analysis: time between first and last plays, and "resters" against "goers";
  - exploration analysis: variance in early scores against later mean score.

## 3. Results
1. **Practice.** Mean score rises with play number up to about 100 plays.
   - Among players with more than 9 plays (n = 45,672), groups with higher maximum scores already score higher on their first plays and also improve faster.
   - Practice alone does not explain the differences. The pattern held at cutoffs of 5, 8, 10 and 20 plays.
2. **Spacing.**
   - Players with higher maximum scores spread their plays over longer periods (bootstrap with 2,000 resamples, t(99) = 7.27, p < .0001).
   - Among players with more than 14 plays (n = 21,575), "resters" (first 10 plays spread over more than 24 h) averaged 47,264 against 44,050 for "goers" (t(20354) = 6.22, p < 1e−5, d = 0.11).
   - Matched-start comparison: a rest of 6 h or more between plays 6 and 15 makes learning curves diverge. The benefit is about 3,000 points, roughly 5 extra plays' worth.
3. **Exploration.** The variance of scores on plays 1–5 correlates with mean score on plays 6–10: **r = 0.59**, p < .0001. A bootstrap null interval of about ±0.009 shows the correlation is not an artifact of the score distribution.
4. No power-law or exponential fits are reported. The paper mentions the debate over the power law of practice (Heathcote et al. 2000) but does not fit curves.

## 4. Limitations
**Reported by the authors:**
- Observational design, so no causal claims.
- Machine ID ≠ person, in both directions.
- No demographics and no data on prior gaming experience.
- Confounds such as prior skill or motivation may drive both spacing and scores.
- The direction of the exploration effect is untested: bad scores may *cause* players to try new tactics.
- Thresholds were chosen by the authors, though robustness was checked.
- The optimal spacing is unknown.

**My own observations:**
- Early score variance is a weak, noisy proxy for exploration. It also partly reflects ceiling effects: players who sometimes score high have high variance *and* a high mean.
- The game is mostly motor skill, not opponent modeling. Effect sizes are small (d = 0.11 for spacing).
- "Exploration" here means varying behavior across whole plays, not information-seeking within a play.

## 5. Ideas to steal
1. **The early-variability → later-performance analysis, computed on agents.**
   - For each agent run (and each human), compute the variance of a per-attempt outcome over attempts 1–5 (boss HP removed, or survival time) and the mean over attempts 6–10. Correlate across runs and seeds.
   - Run it for:
     - the full agent with scouting (β > 0, information gain);
     - the no-scouting ablation;
     - PPO.
   - Prediction: within the scouting agent, information-seeking produces high early variance that pays off later. Since we can manipulate β directly, **we can do the causal experiment the authors could not**: sweep β and show that early variance causally improves later performance (or that it does not, past some point). This turns their correlational finding into an E2 result.
2. **Behavioral exploration metric for humans and agents.** Their score-variance proxy is crude. Use a direct one alongside it: the entropy of the action distribution per boss-move context in early attempts, or the posterior entropy reduction per attempt (information actually gained). Report both, so the human data (where only behavior is visible) and agent data are comparable.
3. **Spacing is not applicable to the agent** (it has no consolidation), but it gives a design caution for the human study. Run all human attempts in one session, so spacing effects do not confound human-versus-agent learning curves. Record inter-attempt intervals anyway.
4. **Individual differences: "start higher, learn faster."** Among agents, variants with a better prior Θ or family library should show the same pattern (better first attempt *and* steeper curve), not just a constant offset. This is a signature of a better prior combined with better inference. Test it by comparing agents after 0, 10 and 40 training bosses.
5. **Do not fit a power law naively.** If we report learning curves, fit both power and exponential functions per run (Heathcote et al. argue that group averages create spurious power laws) and report which fits individual runs. A Bayesian agent that resolves uncertainty quickly should look exponential per run.

## 6. Structures and models of thought
- **Explore early, exploit later.** This motivates the planner's information-gain term with a schedule. A principled version anneals β with posterior entropy rather than with attempt count, which is resource-rational: explore while information is valuable.
- **Variability as a learning resource.** It argues against a purely greedy reflex early on. Let the gate route to the planner, with its information-gain utility, more often in the first attempts on a new boss.
- **Large-N naturalistic play as a benchmark.** If we can get logs from players of a real souls game (or our pygame viewer posted online), the same group-by-max-score analysis gives a population-level comparison for E1.
