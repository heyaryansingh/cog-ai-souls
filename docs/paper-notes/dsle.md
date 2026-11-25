# DSLE: A Learning Environment for Dark Souls Boss Encounters

## 1. Citation and reading status
- **Title:** DSLE: A Learning Environment for Dark Souls Boss Encounters
- **Authors:** Derin Gezgin, Jim O'Connor, Tanner Goodwin, Gary B. Parker (Autonomous Agent Learning Lab, Connecticut College)
- **Year and venue:** 2026, arXiv:2608.09902v1 (cs.AI), submitted 10 Aug 2026
- **Links:** https://arxiv.org/abs/2608.09902 · code github.com/ConnAALL/dsle
- **What I read:** The full arXiv HTML (all body sections, tables and conclusion; the page text ended in the References), in two passes through a fetch tool that summarizes pages, so I saw it in paraphrase, not verbatim. Survival box plots (Fig. 4) were described only through the medians given in the text.

## 2. Core idea and method

### Environment
- **Setup.** Unmodified Dark Souls: Remastered runs in a Docker container (Ubuntu, Wine and DXVK). Several isolated instances run per machine. The interface is Gymnasium `step`/`reset`.
- **State.** HP for player and boss, death count, boss-defeated flags and lock-on state are read from process memory (`/proc/<pid>/mem`). Resets copy a save file and navigate menus by template matching.
- **Observations.** Grayscale 1×600×800 uint8 by default; RGB and `state_only` modes are also available.
- **Actions.** Discrete(14): 4 moves, light attack, strong attack, heal, backstep, 4 rolls, and left and right click (parry). Each action is **held for 250 ms**, about 4 decisions per game-second. Lock-on is automatic.
- **Episodes.** Termination on a win or the player's death. Truncation at 7,200 steps.
- **Reward:** r_t = −0.001 + Δh_b − 0.25·Δh_p + 100·1[win] − 10·1[death], where Δh are the health *fractions* lost this step by the boss (b) and the player (p).
- **Bosses.** All 22 Dark Souls bosses are available.
  - Table 2 tags each boss with characteristics and a benchmark focus, including "Transfer".
  - The **DSLE-5** subset is Asylum Demon (melee baseline), Capra Demon (cramped arena with adds), Chaos Witch Quelaag (hazards and ranged attacks), Ornstein & Smough (two targets) and Gwyn (fast final boss).
- **Start states.** Fixed save states set build, inventory, equipment and location. DSLE-5 uses one low-level character for all five bosses. A separate "advantaged" study uses all stats at level 50.
- **Stochasticity.** 10 runs with identical inputs on Asylum Demon diverged by step 12 and lasted 134–200 steps. Seeds do not make runs deterministic.

### Methods tested
- **Random:** uniform over the 14 actions.
- **Expert:** heal below 40% of max HP observed, respecting a heal cooldown; otherwise move forward and attack.
- **PPO:** 84×84 input, Nature CNN (3 conv layers and 1 fully connected layer), shared trunk. 128-step rollouts, 4 epochs, minibatch 32, learning rate 2.5e-4 annealed, γ 0.99, λ 0.95, clip 0.1, entropy 0.01.
- **DQN:** same CNN. Buffer 100k, minibatch 32, learning rate 1e-4, ε from 1 to 0.01 over the first 10%, learning starts at 10k steps, target update every 1k.
- **Budget:** 100k environment steps per run, 5 seeds.
- **SCOPE (neuroevolution):**
  - Features: 2D DCT of each frame, keeping a low-frequency block, sparsified with k = 100 and p = 90.
  - Policy: bilinear affine, a 1,514-parameter genome.
  - Optimizer: CMA-ES, σ 0.5, population 25, 40 generations, 5 trials per boss.
  - Fitness: f = 100·(H_p + (1 − H_b)), where H are final health ratios. A flawless win scores 200. Wins are counted from the boss-death flag.

## 3. Results

### Table 4: wins / episodes (DSLE-5, standard saves)

| Method | Asylum | Capra | Quelaag | O&S | Gwyn |
|---|---|---|---|---|---|
| Random | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 |
| Expert | 63/100 | 0/100 | 0/100 | 0/100 | 0/100 |
| PPO | 5/1971 | 0/6423 | 0/1684 | 0/5309 | 0/5892 |
| DQN | 8/2469 | 0/12549 | 0/2284 | 0/8203 | 0/9942 |
| SCOPE | 1065/5000 | 0/5000 | 0/5000 | 0/5000 | 0/5000 |

- **Asylum Demon.** The best SCOPE generation wins 43%, against 35% for a prior NEAT agent. PPO and DQN stay at or below 0.33% wins, with **no upward learning trend**; reward stays near −10, the death penalty.
- **The four harder bosses.** Zero wins for every method.
- **Survival medians:**
  - Capra and Gwyn: 7–19 seconds.
  - Quelaag: 52–68 seconds of survival but almost no damage dealt, a "stalemate".
- **Damage.** The expert removes only about 7 of 771 HP from Capra.
- **Advantaged level-50 study (SCOPE).** Wins: Capra 96%, Taurus 88%, Pinwheel 16%, Bell Gargoyles 8%; the rest unwon. The authors call this an easier lower bound on solvability, not comparable to DSLE-5.
- **Compute:**
  - Wall clock per run: PPO 9–21 hours, DQN 12–41 hours. Capra is the worst, because short episodes force many reloads.
  - 100k steps is about 7 hours of game time.
  - Hardware: 5 instances per machine in the experiments; up to 30 on an RTX 5090 with 128 GB RAM. Each instance takes about 0.3 GB GPU, 2–3 GB RAM and 30 GB disk.
- **Comparisons not run:** no human comparison and no transfer experiments.

## 4. Failure modes and limitations

### Reported by the authors
- **Fast deaths** in cramped or multi-target fights.
- **Stalemates:** survival without damage.
- **Structural failures:** Bed of Chaos needs environmental targets destroyed, and Seath is invulnerable until a crystal is broken. Damage alone cannot win these fights.
- **Practical limits:** low throughput in real time, a game licence is needed, nondeterminism, a fixed 100k budget that bounds performance at that budget rather than measuring true difficulty, and most bosses unevaluated.

### What I notice
- Pixel-only RL with 100k steps (thousands of episodes) basically fails. Win counts of 0/12,549 show that **sample-inefficient learners cannot even get started**. This is the strongest motivation for model-based, few-death learning.
- Their metrics have no "deaths to first win" and no learning-curve AUC, although those are what souls players care about.
- The reward weights are hand-set, and the death penalty dominates.

## 5. Ideas to steal

1. **A failure-mode trichotomy as an automatic per-episode classifier.** Classify each death or episode as:
   - **fast death** (survival below τ_s with little boss damage);
   - **stalemate** (long survival with boss HP removed below ε);
   - **attrition loss** (substantial damage dealt but died);
   - **structural** (the win needs a non-damage mechanism, for example a phase gate or invulnerability condition).

   Report the distribution over attempts. For our agent, the predicted trajectory is fast death → attrition → win, while PPO stays at fast death. MiniSouls could include one **structural boss** (invulnerable until a condition is met) in T4, to test rule or law induction beyond timing.
2. **Joint survival and damage reporting.** For each method and boss, report median survival (in ticks) *and* the median fraction of boss HP removed, as a 2D scatter. This separates defensive skill (reading the boss) from offensive skill. Use it in E1 alongside win curves, because wins are rare early on.
3. **Their reward as the shared baseline reward.** r = −0.001 + Δh_b − 0.25·Δh_p + 100·win − 10·death, for the PPO, RecurrentPPO and AD baselines in MiniSouls. This makes our baselines comparable to the DSLE numbers, and lets us cite that the same reward in the real game gives about 0 wins in 100k steps.
4. **SCOPE-style fitness as a dense, monotone score:** f = 100·(H_p + 1 − H_b), from 0 to 200, as a secondary per-attempt score. It is useful for human-normalized AUC when win rates are 0.
5. **A stochasticity check.** Replay identical action sequences 10 times and measure divergence. Do the same in MiniSouls to *quantify* boss stochasticity per boss, which bounds the maximum achievable predictive accuracy for the calibration metric.
6. **One standard start build across bosses** so scores are comparable. Separately, an "advantaged" build as a difficulty knob or curriculum, the analogue of our self-model rule changes. A T4 manipulation could deliberately change the player's stats.
7. **Positioning and stretch goal.**
   - DSLE-5 maps to our archetypes: melee baseline, adds/cramped arena, hazards/ranged, multi-target, fast final boss. Our boss generator could include these five archetypes explicitly and report results per archetype.
   - The stretch goal is to run our agent's boss-model inference on DSLE `state_only` observations for Asylum Demon. This would need cue extraction (animation IDs), which DSLE does not currently expose.
   - Their 250 ms action hold (4 Hz) gives a realistic decision rate for the "tick" in our MiniSouls.
8. **Compute framing.** PPO needs about 2,000 to 12,000 deaths per boss for 0–8 wins. Humans need roughly 3–30. Use this as the motivating "sample-efficiency gap" figure in the write-up.

## 6. Structures and models of thought
- DSLE offers a **raw-perception, model-free baseline**, and it fails. The structures missing from their agents are exactly the ones that matter:
  - an explicit **opponent model** with temporal structure (move durations, combos), our HSMM;
  - **event-level perception** (animation cues instead of pixels);
  - **self-knowledge** (heal timing, i-frames);
  - **conditional laws** (invulnerability conditions, phase gates), which the structural failures (Seath, Bed of Chaos) show cannot be learned by damage-seeking reward alone. They need hypothesis-level reasoning about *why* damage is not landing.
- Their "Transfer" tag and the proposed curriculum (Asylum → Stray Demon) point to **family-level transfer**. That is our CRP family library: Stray Demon is effectively Asylum Demon's family with altered parameters.
