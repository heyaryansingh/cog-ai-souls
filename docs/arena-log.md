# Arena research log (chapter 2)

The MiniSouls log (`research-log.md`, Entries 1–33 and P1) is chapter 1. This log follows the revamp (`foundation.md`). Everything here is on **Arena**: a 2D arena, 17-joint skeletal bosses, unlabelled pose input, and a rich player body.

## 2026-11-25 · A1: Arena v0 runs; perception from raw pose works out of the box

**Environment.**
- 4 boss archetypes (knight, beast, caster, giant) with 6–8 move templates each.
- Moves are built from motion clips: wind-ups, log-normal holds, strikes with arc, line, disc, ring or projectile hitboxes, combos, leaps, tracking, phase-2 moves, input reading and posture.
- The player body has free movement, directional rolls with i-frames, light/heavy attacks, guard with a parry window, a jump over low attacks, and heals, plus poise and stamina. There are 3 builds.
- Each tick gives 88 inputs: noisy keypoints with dropout, projectiles, 3 audio events, and body state.
- The single-thread simulator runs at about 18k ticks/s.

**Scripted references** (20 bosses × 3 fights):

| Player | Win rate |
|---|---|
| Random | 3% |
| Mash attacks, ignore the boss | 0% |
| Cheat dodger (sees the true boss state, rolls just before impacts, punishes recoveries) | 64% overall: knight 60%, beast 47%, caster 60%, giant 100% |

So Arena is hard without reading the boss, and beatable with it.

**Perception (MoSeq-style syllables).** The pipeline:
1. Raw keypoints → boss-centric pose (pelvis-centred, rotated by the hip line).
2. PCA to 12 dimensions plus velocities.
3. A sticky Gaussian HMM with K = 25, fitted on 2 fights of one boss and scored on a third fight against the true animation segments (segment kind × target pose).

16 bosses, 4 per archetype:

| Archetype | NMI (segments) | NMI (moves) | Boundary F1 (±3 ticks) | True segments / states used |
|---|---|---|---|---|
| Knight | 0.62 | 0.51 | 0.77 | 15.8 / 22.8 |
| Beast | 0.61 | 0.40 | 0.78 | 13.3 / 22.3 |
| Caster | 0.65 | 0.51 | 0.78 | 12.8 / 22.0 |
| Giant | 0.59 | 0.52 | 0.61 | 15.8 / 22.8 |

**Findings:**
- **Without any labels, two fights of raw pose already give syllables aligned with the true animation segments.** About three quarters of segment boundaries are found within 100 ms.
- Move-level NMI is lower, as expected. Moves share wind-ups (spin vs slash, double stomp vs stomp), so *moves* are sequences of syllables. That is the job of the program learner on top: the slow mind's semi-Markov grammar, now over syllables.
- **Next for perception:** explicit durations (HSMM), autoregressive dynamics (AR-HMM, as in MoSeq), and a cross-boss syllable library learned on many bosses, so a new boss starts with a prior vocabulary.
