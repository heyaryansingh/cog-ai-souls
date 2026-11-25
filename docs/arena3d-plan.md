# Chapter 3: Arena3D, a motion-capture souls arena (plan, Nov 25 2026)

Aryan's request: a 3D world, motion capture, pixels and keypoints read in a human-like way, an open move
library like a real souls game, and hitboxes that are not simplified. He owns Elden Ring and Hollow Knight.

## What changes from Arena (2D) to Arena3D

| | Arena (ch. 2) | **Arena3D (ch. 3)** |
|---|---|---|
| World | 2D plane | 3D, MuJoCo physics engine, floor, lighting, shadows |
| Boss body | 17 hand-keyframed points | full CMU humanoid (31 bodies, 56 joints), scaled 1.2–2.6×, holding a weapon |
| Animation | hand-built keyframes | **real motion capture**: 836 CMU clips, 3.3 h |
| Move library | ~28 templates | **open**: all mocap cut into 14,203 motion units, 4,148 of which can strike; any unit can be a boss move |
| Combos | hand-set matrix | motion-matching cost between the end pose of one move and the start of the next |
| Delays | log-normal holds | log-normal windup freeze just before the weapon's fastest frame |
| Hitboxes | 2D arcs/discs, one impact tick | **per-substep 3D contact** of the weapon capsule and boss limbs with the player's 19 body capsules, 4 substeps per 30 ms frame, damage by contact speed and body part |
| Blocking | rule | geometric: a shield box on the forearm must be touched first |
| Agent sees | 88 floats of noisy keypoints | **rendered pixels** from a lock-on camera, 3 sounds, proprioception |
| Test bosses | new random draws | bosses built only from **held-out mocap subjects** (motions never seen) |

## Human-like perception (pixels → body → syllables)
1. **Peripheral motion channel** (magno): low-res frame differencing gives motion energy, where it is, and onsets.
2. **Foveal pose channel** (parvo): an off-the-shelf human pose estimator (MediaPipe Pose: 33 landmarks plus
   metric 3D landmarks) on a crop around the attended boss. Attention is steered by the last pose and by
   peripheral motion.
3. **Smoothing and coasting** through dropouts (Kalman-style), human-like latency (~200 ms).
4. **Syllables**: the existing sticky-HMM/HSMM syllable learner, now on 3D landmarks, scored against the true
   motion units (NMI, boundary F1).
5. **Prediction**: a learned forward model of the body (next-k-frame landmark forecast); prediction error drives
   attention and the surprise gate from chapter 1.

The same pipeline runs on **real Elden Ring video**, since the estimator only needs pixels. Test so far: it
finds both the player and the boss mannequins in an Arena3D render.

## Elden Ring and Hollow Knight
- **Elden Ring, step 1 (no risk):** Aryan records a few boss fights (e.g. Margit, Godrick, Malenia: humanoid
  bosses) with OBS. We run the perception pipeline on the video: boss landmarks, syllables, and a human-behaviour
  log (when he rolled, when he was hit). This tests the perception on real game pixels and gives human
  comparison data.
- **Elden Ring, step 2 (optional, offline only):** a live agent through screen capture and keyboard input, as
  community projects do. It must run offline with anti-cheat disabled, never online.
- **Hollow Knight** is 2D, so it is not the main track. It has a mod API and existing RL setups, so it could serve
  later as a fast real-game check.

## Agents on Arena3D (in order)
1. Scripted baselines: random, aggressive, and an oracle dodger (reads true contact predictions).
2. Oracle planner on cloned world states (acting ceiling).
3. Recurrent PPO on pixels (the deep-RL baseline, like DSLE).
4. The Bayesian agent: perception → syllables → hazard and 3D hit-volume model → belief-space planner, with
   cross-boss priors. Main measure: fights to first win on held-out bosses, compared with Aryan.

## Known gaps, honestly
- The player body is animation-driven (walk cycle, crouch roll, one strike clip, raised-arm guard), not a full moveset.
- CMU mocap is not combat-specific; many units are everyday motions with a weapon in hand. A combat mocap set
  (Bandai Namco dataset 1 has slash, kick and punch) would add true attacks. Its download was blocked here.
- No camera control, no terrain yet. Rendering is CPU-bound in the cloud; it will be faster on the laptop's GPU.
