# How our arenas compare with real souls games and existing research environments

*Nov 25, 2026. Sources are listed at the end. "Souls Arena 100 Trials" (Steam, Jan 2026) is a commercial souls-like game with no research API. It is noted for completeness, but it is not an RL environment.*

## Side by side

| | **Real game** (Elden Ring / DS3) | **SoulsGym** (DS3) | **DSLE** (Dark Souls Remastered) | **Skill-graph agent** (DS3, 2026) | **MiniSouls** (ours, ch. 1) | **Arena** (ours, now) |
|---|---|---|---|---|---|---|
| What the agent sees | 3D video + sound | Game-memory state vector, or a 90×160 image (WIP) | 600×800 grayscale pixels | 25-dim memory-read state | 21 symbolic cue tokens | 17-joint 3D skeleton keypoints (noisy, dropout, 200 ms delay) + projectiles + 3 sound events + body (88 floats) |
| Space | 3D, camera, terrain | 3D (real game) | 3D (real game) | 3D (real game) | 1D lane | 2D plane, lock-on frame, no terrain |
| Bosses | Hundreds, hand-made | 1 (Iudex Gundyr) | 22 | 1 (Iudex) | Procedural, 1D | Procedural, unlimited: 4 archetypes, about 7 move templates each, random timing, combos, laws |
| Boss mechanics | Combos, delays, tracking, AoE, projectiles, grabs, phases, hyper-armor, adds | Real game | Real game | Real game | Single moves, timing, reactive laws | Combos, log-normal delays, tracking, leaps, arcs, lines, discs, rings, projectiles, grabs, phase 2, input reading, posture |
| Player body | Full moveset, weapons, builds, camera | Real game, 1 build, no blocking | 14 actions held for 250 ms | Macro skills | 8 actions | Free movement, directional rolls with i-frames, light/heavy, guard + parry, jump, heal, poise, 3 builds |
| Timing | Real time, 60 fps | Game paused between steps | Real time (not paused) | Real time, key macros | Discrete, 20 Hz | Discrete, 30 Hz, deterministic |
| Reproducible | No | No (seed has no effect) | No (diverges by step 12) | No | Yes | Yes |
| Speed | 1× | About 1× (game speed) | About 4 decisions/s per instance, 2–3 GB RAM each | 1× | ~50k ticks/s | ~18k ticks/s per core |
| Needs | Console/PC | Windows + DS3 | Linux + NVIDIA GPU + Docker + Wine + game copy | Windows + DS3 | Laptop | Laptop |
| Reported RL results | n/a | About 45% win on Iudex (state input); pixel-input goal of > 10% not yet met | PPO/DQN ≤ 0.33% on the easiest boss, 0% elsewhere, after 100k steps (9–41 h per run) | 44% (phase 1, HP-threshold "win") vs 0% for end-to-end DQN | Bayesian agent 37% by attempt 4, RL² 13% | Recurrent PPO 42% on random bosses after ~15k fights; cheat-dodger reference 64% |

## Where Arena is now closer to real games than other research setups
- **Perception is of a body, not of labels or RAM.** SoulsGym and the skill-graph work read game memory: boss animation IDs, positions and timers handed over directly. Arena gives a noisy, delayed skeleton, closer to what a human sees.
- **Many bosses with shared structure.** Every real-game environment has 1 boss (SoulsGym, the skill-graph agent) or a fixed 22 (DSLE). Arena samples unlimited bosses from archetype hyperpriors. That is what is needed to study *learning a new boss* and *transfer*, which is the question Aryan cares about.
- **Ground truth.** We know the true moves, segments, holds and hitboxes, so we can score perception (NMI, boundary F1) and inference directly. A real game cannot provide this.
- **Laptop-scale and reproducible.** DSLE needs a GPU server and real-time play. Arena runs about 18k ticks/s per core, deterministic by seed.

## Where Arena is still clearly *not* a real souls game (honest gaps)
1. **2D, not 3D.** No camera control, no vertical play, no terrain, no line-of-sight or field-of-view limits.
2. **Synthetic animation.** Poses are hand-built keyframes with eased interpolation, not motion capture. Real tells are subtler and richer: weapon trails, VFX, sound design, camera shake.
3. **Keypoints, not pixels.** Real players see pixels. Keypoints are a deliberate middle step (what a pose estimator would output), and a pixel track is planned.
4. **Thin move library.** About 28 templates across 4 archetypes, against hundreds of hand-animated moves per real boss roster. There are no weapon movesets, no adds or multi-boss fights, no hyper-armor, no status buildup.
5. **Simplified physics of harm.** 2D hitboxes evaluated at one impact tick per strike. Real games use per-frame 3D hit volumes, multi-hit swings and trades.
6. **RL may find Arena easier than DSLE does.** Recurrent PPO reaches 42% here, while DSLE reports ≤ 0.33% under 100k steps. The reasons are our ~1000× more samples, a cleaner observation and a simpler world. So Arena is not yet hard in the way real games are hard for RL.

## What this means for the plan
- **Arena is the science testbed:** many bosses, ground truth, generalization and human comparison.
- **SoulsGym is the realism check.** It runs on Aryan's Windows laptop with DS3 and works paused, which suits our planner. Its state includes the boss's animation ID, so our boss-program learner can run on real DS3 animations (animation IDs as syllables) against Iudex Gundyr. Target: compare with SoulsGym's 45% PPO and the 44% skill-graph agent in deaths-to-first-win. **Needs: DS3 installed on the laptop.**
- **DSLE** is the strongest real-game benchmark (22 bosses, pixels), but it needs Linux, an NVIDIA GPU and real-time play, so it is out of laptop scope. We can cite its baselines as the "deep RL fails on real bosses" anchor.
- **Arena v1 upgrades queued to close the gaps:**
  - field of view and camera;
  - hyper-armor and trades;
  - multi-hit swings with per-tick hit volumes;
  - a pixel renderer (egocentric 64×64) for a pixel-input track;
  - a larger move library, plus a held-out archetype for transfer.

## Sources
- [SoulsGym docs: Iudex Gundyr](https://soulsgym.readthedocs.io/en/dev/envs/iudex.html), [SoulsGym README](https://github.com/amacati/SoulsGym/blob/master/README.md)
- [DSLE: A Learning Environment for Dark Souls Boss Encounters (arXiv 2608.09902)](https://arxiv.org/pdf/2608.09902)
- [Learning Transferable Skills in Action RPGs via Directed Skill Graphs (arXiv 2601.17923)](https://arxiv.org/pdf/2601.17923)
- [Souls Arena 100 Trials (Steam)](https://store.steampowered.com/app/4255480)
