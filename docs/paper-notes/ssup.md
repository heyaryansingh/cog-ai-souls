# SSUP: Allen, Smith & Tenenbaum 2020

## 1. Citation and access
- Kelsey R. Allen, Kevin A. Smith, Joshua B. Tenenbaum. "Rapid trial-and-error learning with simulation supports flexible tool use and physical reasoning." *PNAS* 117(47):29302–29310, 2020. doi:10.1073/pnas.1912341117. arXiv: https://arxiv.org/abs/1907.09620 (v3, 29 Jun 2020).
- **What I read.** The main text of the arXiv v3 PDF, through a text-extraction tool. Direct download was blocked in my sandbox. The PDF does **not** contain the SI Appendix, so these details are not available to me: the GMM component count, learning rate, ε, acting threshold, noise magnitudes, number of initial prior samples and the policy-gradient equation. The main text gives no formal equations.

## 2. Core idea and exact method
**Virtual Tools game.**
- 2D physics puzzles. The player picks one of 3 unfamiliar tools and places it once per attempt so that a red object ends up touching a green goal region.
- After each attempt, the player can reset and retry.
- 30 levels:
  - 20 in Experiment 1, including 6 matched A/B pairs where a small change in the scene changes the solution;
  - 10 new levels in a validation experiment.
- Time limit: 2 minutes per level.

**SSUP = Sample, Simulate, Update.**
1. **Sample (object-oriented prior).**
   - Pick a movable object uniformly.
   - Pick x within a span slightly wider than that object.
   - Pick y above or below it.
   - Pick the tool uniformly.
   - Proposals come from the current policy with probability 1−ε and from the prior with probability ε.
2. **Simulate.**
   - A noisy physics engine (noise only in the direction and magnitude of collision forces) runs n_sims = 4 rollouts per proposal.
   - Reward = 1 − (min distance from the red object to the goal / the same distance with no tool added), averaged over rollouts.
   - **Act** if the average reward exceeds a threshold. If no proposal passes the threshold after T = 5 proposals, take the best imagined action so far.
3. **Update.**
   - The policy π′ is a Gaussian mixture over (tool, position), initialized from prior samples.
   - It is updated by REINFORCE-style policy gradient (Williams 1992), using *both* imagined outcomes and real outcomes.
   - Real failures move probability away from the failed region.
   - The dynamics model is fixed. There is no world-model learning.

**Key point.** Exploration happens in imagination. The real action is chosen from simulation results. So the model interleaves "thinking" (cheap simulated trials) with "acting" (costly real trials), and it uses real failures as evidence.

## 3. Experiments and key numbers
**Humans:**
- Experiment 1: 94 MTurk participants, 14 levels each.
  - Solution rate 81% (SD 19%), ranging 31–100% across levels.
  - Mean attempts 4.5 (SD 2.5), ranging 1.5–9.4 across levels.
- Experiment 2: 50 participants, 10 levels.

**Model fit to humans (Experiment 1).**
- Attempts per level: r = 0.71 (95% CI 0.62–0.76). Means were 4.48 for humans and 4.24 for the model.
- Accuracy per level: r = 0.86 (95% CI 0.76–0.89).

**Validation (Experiment 2, no re-tuning).** Attempts r = 0.85, accuracy r = 0.95.

**Other metrics:**
- Cumulative solution rate as a function of the number of placements (model curve against human curve).
- Kernel density estimates of human first and last placements, compared with 250 model runs.
- Matched-pair sensitivity of the first action.

**Ablations and baselines:**
- **No Prior, No Simulation, No Update.** Each fits significantly worse (bootstrap p < 0.0001 in Experiment 1). In Experiment 2, No Update was only marginally worse (p = 0.055).
- **Parameter Tuning.** Learns density, friction and elasticity instead of updating the policy. Roughly equal to No Update, so learning the dynamics was not needed here.
- **DQN + Updating.** Trained on 4,500 random levels, then updated online. The worst of the plausible models: near chance, with the most attempts.
- **Guessing.** Random tool placement.
- No model reached the noise ceiling on matching human placements.

**Compute.** Not detailed in the main text. The run is cheap: at most about 20 simulations per real attempt.

## 4. Limitations
**Reported by the authors:**
- The prior ignores context. In Falling-A, people tip the container from above, while the model places an object below it.
- The model has no fast "fit" judgement without simulating. In a hypothetical level that tests this, it solves 10%.
- Planning is forward only, with no subgoals or working backward. In a hypothetical catapult level it solves 8%, stuck at a local optimum.
- The model does no dynamics learning, which would fail for glued or very heavy objects.
- The scope is mechanical reasoning only.

**My own observations:**
- It is a one-shot action space (one placement), not sequential control under time pressure.
- The world model is given (true physics plus noise). The hard part of our problem, inferring the opponent model, is assumed away here.
- Without the SI, the acting threshold and ε look like they play the role of a metareasoning knob, but they are hand-set, not derived.

## 5. Ideas to steal
1. **Separate the posterior over *models* from the posterior over *good actions*.**
   - SSUP keeps a learned action policy (a GMM) that is updated from imagined *and* real outcomes.
   - For souls, the analog is a per-boss "where to punish / when to dodge" proposal distribution that the planner warm-starts from. It gets updated by real deaths (negative REINFORCE signal) and by simulated rollouts inside the posterior.
   - It could feed the reflex's DAgger data, or act as a cached opening plan per boss move.
2. **Act-when-good-enough thresholding with a timeout.** "Simulate until the expected reward exceeds θ, or until T proposals." This is the simplest version of the metareasoning gate, and a good **baseline gate** to compare the learned gate against (E5). Sweep θ and T to trace a speed–accuracy curve.
3. **Copy the human-comparison metrics:**
   - per-boss correlation of (agent deaths-to-win) with (human deaths-to-win);
   - per-boss correlation of win rate;
   - **cumulative win-rate curves against attempt number**, overlaid for agent and humans;
   - **distributions of first-attempt behavior** (for example, the agent's first-attempt dodge-time histogram per move against the human one);
   - **matched-pair bosses** that differ by one small rule (a delayed version against a normal version of the same move), checking that first-attempt behavior differs for both agent and humans.
4. **Ablation trio as E2 template:**
   - no prior (Θ removed);
   - no simulation (reflex only, never plan);
   - no updating (posterior frozen after attempt 1).
   - Then add "Parameter tuning only" (fit timings, but no structure learning, i.e. no CRP/laws). SSUP found that updating action beliefs mattered more than tuning parameters. Test whether structure learning or action caching matters more for us.
5. **Generalize without re-tuning.** SSUP validated on 10 fresh levels with no parameter changes (r = 0.85/0.95). Adopt the same rule: freeze all agent hyperparameters before running the human comparison on new bosses, and report it explicitly.
6. **Strong capability target from DQN + Updating.** Model-free RL pretrained on 4,500 levels plus online updates was near chance. Our PPO/RL² baselines should be expected to look similar on T3/T4, and this paper is the precedent to cite.

## 6. Structures and models of thought
- **Mental simulation as the inner loop of trial-and-error.** The "slow mind" is SSUP turned on an adversary: sample a plan, simulate it against posterior samples of the boss program, update. The difference is that we must also infer the simulator (the boss program), which SSUP gets for free.
- **Object-oriented, structured priors over actions** cut the search space. Our analog is a prior over responses conditioned on the move primitive: dodge timing tied to the wind-up stage, punish windows tied to recovery.
- **Learning from imagined and real outcomes in one update rule.** This is a Dyna-like loop. It argues for treating each death as a strong REINFORCE signal on the action-proposal distribution, *and* as likelihood evidence for the boss posterior. Use both.
- **The missing pieces SSUP names (subgoals, backward planning, model learning)** are exactly where our design adds structure. Model learning is our core. Subgoals could be "bait move X, then punish."
