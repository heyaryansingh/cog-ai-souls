# Revamp: from MiniSouls to Arena — realistic bosses, rich input, a real body

*Nov 25, 2026. Motivation: Aryan's review. MiniSouls bosses are too unlike Elden Ring bosses. Learning should look like real-world learning: lots of raw input and a rich, diverse embodiment.*

## 1. What MiniSouls gets wrong

| What a real boss fight has | What MiniSouls has |
|---|---|
| **Raw perception.** The tell is a body animation seen from an angle, shared in part across moves, with no label | 21 clean symbolic cue tokens ("overhead", "a_sweep") |
| **2D/3D space.** Spacing, circling, attack arcs and lines, tracking that turns toward you, rolling *through* or *around* a swing | One lane: distance only |
| **Combos.** Branching strings (1→2→3 or 1→end), delayed hits that bait early rolls, mix-ups | Single moves |
| **Hitbox geometry.** Sweeps cover an arc, thrusts a line, slams a disc, and projectiles fly | A reach number |
| **Movesets that change.** Phase-2 moves, enrage, gap closers, jumps, grabs, AoE, projectiles | A phase-2 speed-up |
| **A real body.** Free movement, facing and lock-on, directional rolls, light/heavy chains, guard and parry, jump, heal commitment, poise and stagger, posture breaks on the boss | 8 discrete actions in 1D |
| **Learning across bosses.** Animation semantics generalize ("big wind-up = heavy hit") and archetypes recur (knight, beast, caster) | A shared timing prior |

## 2. New environment: **Arena** (`arena/`)

The arena is a continuous 2D top-down world running at 30 Hz, simulated in NumPy and vectorized so a laptop can run thousands of fights.

**Boss body.** An articulated skeleton with 15 joints (pelvis, spine, head, two arms × three joints, two legs × three joints) and a root position and facing. Every move is a sequence of **motion clips** built from a primitive library:
- *raise*, *hold* (the delay bait), *swing-arc*, *thrust*, *overhead-slam*, *spin*, *leap*, *stomp*, *cast*, *grab*, *recover*, *step*, *strafe*.
- Each clip has parameters: speed, hold duration (log-normal, the timing uncertainty), turn-tracking rate, and travel distance.

So the tell is a *pose trajectory*. Different moves can share their first frames (a sweep and a spin both start with a wind-back), and only the later frames disambiguate, as in the games.

**Attacks.** Each active window carries a **hitbox shape** in boss-local coordinates: an arc sector (sweep), a line or capsule (thrust), a disc (slam, stomp, AoE ring with an inner safe zone), or a projectile entity (bolt, homing orb). It also has damage, poise damage, a guard-break value, and whether it is unblockable (grab).

**Boss program.** A semi-Markov graph over **moves**. A move is a combo: a chain of clips with branching continuations, chosen with probabilities that depend on distance, angle (behind the boss vs in front), the player's action (input reading: punish heals, react to rolls) and phase. Phase 2 adds new moves and speeds up. The boss's posture fills when it is hit; a full bar staggers it, opening a critical-hit window.

**Archetypes** (procedural generator with hyperpriors):
- **knight:** sword arcs, thrust, delayed overhead;
- **beast:** pounces, multi-bite combos, tracking;
- **caster:** projectiles, AoE rings, teleport-retreat;
- **giant:** stomps, slow sweeps, grabs.

Bosses are sampled from a hierarchical prior over archetype, then moves, then clip parameters, so families share structure. That makes "held-out family" tests possible.

**Player body** (the agent's embodiment):
- 2D position, facing, lock-on toggle;
- continuous movement (8-way × walk/run);
- **directional roll** with i-frames and recovery, where weight class changes roll distance and i-frames;
- light-attack chains of 1–3 hits and charged heavy;
- guard and a parry window;
- jump, which avoids low sweeps;
- heal with commitment;
- stamina, HP and poise. Getting hit with low poise staggers the player.

Body parameters vary across "builds" (light, medium, heavy), so the agent must learn *its own body*, as in the T4 self-model.

**Observation** ("lots of inputs"), about 70 continuous numbers per tick:
- boss skeleton keypoints (15 × 2, relative to the player, with Gaussian jitter and occasional dropout);
- boss root velocity;
- up to 4 projectile positions and velocities;
- proprioception: own stamina, HP, poise, action phase, facing;
- the boss health and posture bars.

An optional egocentric 64×64 rendered frame gives a pixel track, used for the deep-RL baselines.

## 3. New agent: same cognitive core, learned from raw input

The 1D results carry over as design principles: Bayesian program learning, state averaging, habit plus deliberation, uncertainty-based arbitration. They are now applied where the hard part is perception and space.

1. **Perception: discover the moveset from raw pose.** An unsupervised **sticky HDP-HSMM / autoregressive HMM over boss keypoint trajectories** segments animation into reusable *syllables*, as MoSeq (Wiltschko et al. 2015) does for mouse behaviour. Syllables replace hand-given cue tokens. The agent sees the raw skeleton, learns "this motion pattern" as a unit, and its boss model is built on those units. A small learned pose encoder (amortized: inference compilation on simulated bosses) proposes syllable labels online; the HSMM keeps them calibrated.
2. **Boss program learning.** The existing slow mind (resample-move SMC over semi-Markov programs with timing, laws and structure moves) runs on syllables. It is extended to **combo graphs**: branching continuations and conditioning on angle and distance.
3. **Spatial threat model.** For each syllable phase, a Bayesian estimate of *where* it hurts in boss-local coordinates. Hit and miss events give labelled points, giving a kernel density or Gaussian-process danger map with uncertainty. This is the 2D version of "reach".
4. **Self-model.** Roll distance, i-frame window, stamina costs and attack timings are learned from proprioceptive events per build (extending RulesBelief).
5. **Fast hands:**
   - a **planner** (MPPI over continuous action sequences in the learned world model, with belief states sampled from the filter);
   - a **reflex** distilled by DAgger over *belief features*. The threat clock becomes a **threat field**: P(an in-hitbox impact at offset Δx, Δy within t ticks) on a small polar grid around the player;
   - **uncertainty-scaled arbitration** between the two (from E31 and R11).
6. **Intuition across bosses.** An amortized proposal network trained on thousands of simulated bosses (the archetype prior) maps the first seconds of a fight to likely moves and combo structure. Its proposals are superset-seeded into SMC (from E18/E20).

## 4. Baselines and evaluation

- **Deep RL on raw input:**
  - recurrent PPO on keypoints;
  - recurrent PPO on pixels;
  - RL² across bosses.
  Each gets a large training budget (millions of steps; vectorized env).
- **World-model RL**, DreamerV3-style, if time allows.
- **Metrics:**
  - win rate by attempt and deaths-to-first-win, with boss-clustered tests;
  - transfer to held-out bosses, held-out *archetypes* and held-out *moves*;
  - a new body build (self-model);
  - human-reaction latency (200 ms) as the headline setting;
  - human comparison through a playable browser port of Arena.

## 5. Bridge to the real game (stretch)

Real Elden Ring footage can test the perception module directly. Run an off-the-shelf pose estimator (RTMPose or MediaPipe) on boss-fight video, then ask whether syllable discovery recovers the boss's documented moveset (wiki move lists as ground truth) without labels. This needs video files on the laptop, so it is the stage-3 goal.

## 6. Order of work

1. Arena env, boss generator, player body, keypoint observations; scripted-player sanity checks; true-program oracle planner (the ceiling at latency 6 ticks = 200 ms).
2. Recurrent PPO baseline on keypoints, started early because it trains for hours.
3. Perception: syllable discovery from keypoints, scored by recovery of true move segments (NMI, boundary F1).
4. Slow mind on syllables with combo graphs and the spatial threat model; learning curves vs oracle and PPO.
5. Reflex, arbitration, intuition net; transfer splits; playable port; real-footage bridge.

The 1D MiniSouls results (Entries 1–33) stay as the controlled first chapter. The P1 replication and the queued runs finish in the background.
