# Arena3D research log (chapter 3)

Plan and design: `docs/arena3d-plan.md`. Chapter 1 (MiniSouls) is in `docs/research-log.md`; chapter 2 (2D Arena) is in `docs/arena-log.md`.

## B1 · Nov 25 · The world
- MuJoCo 3.x. The CMU humanoid (dm_control 2020 asset) is used twice: a boss (scaled 1.2–2.6×) and a player (1×). Motion is kinematic: set qpos from mocap, then compute contacts.
- Motion capture: dm_control's CMU 2020 retarget, 836 clips, 3.3 h, 30 ms frames.
- Open move library (`arena3d/library.py`): clips are cut at lulls of whole-body speed into 14,207 units. 4,148 units have weapon-tip speed ≥ 6 m/s (median 9.4 m/s, p90 18 m/s; real sword tips run 10–20 m/s). There are no hand labels. 11 mocap subjects are held out for test bosses.
- Checks that caught real bugs:
  - The CMU root frame faces +90° from its x-axis, so the facing was corrected against the hip line.
  - The weapon axis was fixed so the blade points out of the fist.
  - Strike targeting: "attack warping" rotates and lunges each move so the weapon's fastest hittable frame lands on the player, as game animation systems do.
- Contact rate: a standing player is touched by 5–20% of strikes per boss and dies in 2–7 hits.

## B2 · Baselines (12 → 6 held-out bosses)
| Player | Win | Boss HP removed |
|---|---|---|
| random | 0% | ~6% |
| aggressive (walk in, attack) | 0% | 11–16% |
| oracle dodger (simulates the true future, same RNG) | 17% (1/6) | 75% |

## B3 · Perception on pixels
The pipeline is in `arena3d/perception.py`. MediaPipe Pose runs on a fovea placed by the lock-on reticle, with self-suppression. Scored against projected ground-truth joints on 2 bosses × 150 frames:

| Variant | Boss found | PCK@0.1 |
|---|---|---|
| full frame, no fovea | 16% | 0.75 |
| fovea | 46% | 0.69 |
| fovea, 1.6× size | **64%** | **0.73** |
| fovea without self-mask | 40% | 0.86 |

Lesson: foveation is what makes an off-the-shelf pose net usable on a small, far boss. On simulated video the video tool finds the boss in 73% of frames at 27 fps.

## B4 · Agent v1 (2D agent ported) fails
- Setup: syllables (sticky HMM on boss-centric 3D pose), hazards timed by swing sound, polar hit map, body-rollout planner.
- Result on training bosses: 0/72 wins, ~10% boss HP removed.
- Syllable quality on held-out bosses: NMI(syllable, true unit) is 0.17 with sensor noise, 0.31 without noise, 0.22 with a Kalman smoother. Chapter 2 had 0.62.
- Reading: real mocap motion is far more varied than keyframed templates, so a 40-state HMM fragments each move. The hazard and hit map then have little to condition on, and the agent ends up turtling behind its guard.

## B5 · Realism fixes driven by B4
- Bosses get recovery after combos: an idle mocap unit is played with probability 0.5–0.9. Aggression is now 0.35–0.8. Real bosses leave punish windows.
- Guard follows the Elden Ring rule: a raised guard blocks hits from the front half, costs stamina by impact speed, and breaks at 0 stamina. Before, only blade-on-shield contact counted, and just 1 in 3 guarded hits were blocked.

## B6 · Agent v2: learned threat model
- A network predicts P(unprotected hit in [t+k, t+k+2] | last 10 frames of perceived pose, my position in the boss's frame at t+k), for k = 1…24.
- It is trained on exploratory play against 24 training bosses. A per-boss Bayesian logistic last layer adapts across attempts.
- This replaces "syllable → hazard → hit map" with one amortised appraisal. It is the 3D counterpart of chapter 1's intuition net plus slow mind.
- Threat net trained on 465k samples from 24 bosses: held-out AUC 0.71 (AP 0.03 vs base 0.015).

## B7 · Attacks verified to connect
- Diagnostic: with unverified bosses, a standing player was touched by only 5–20% of strikes. Most mocap "strikes" are everyday motions that sweep past the target. Real souls attacks are tuned to hit unless you respond.
- Fix (`arena3d/aim.py`): when a boss is sampled, each candidate strike is played against a standing player in the real 3D world, and only strikes that connect are kept, the way animators tune attacks. Afterwards 76% of strikes hit a standing player.

## B8 · Why perception does not yet buy prediction (diagnosis)
Exploratory play, label = an unprotected hit in a 3-frame window k frames ahead (k = 1–24), held-out bosses. Gradient-boosted classifiers, AUC:

| Features | AUC |
|---|---|
| my position relative to the boss (distance / local xy) | 0.72–0.76 |
| + perceived pose (noisy, delayed) | 0.73–0.74 |
| + perfect pose, no delay | 0.74 |
| + weapon-tip keypoint (perfect) | 0.76 |
| **true move state** (striking? frames to danger frame, hold) + position | 0.79 |
| motion-matching anticipator (130k remembered frames) alone | 0.51 |

Coarser label (any hit in the next 24 frames), perfect pose, 8 training bosses:

| Features | AUC |
|---|---|
| position | 0.725 |
| raw pose only | **0.43** (below chance) |
| raw pose + position | 0.65 |
| invariant cues (hand speed, hand-to-me distance, closing velocity, windup rise, body speed) + position | **0.75** |

**Reading.**
1. Exact hit timing is weakly predictable even from the true move state (0.79): log-normal windup holds and random combo choices make souls timing uncertain *by design*. An agent must hedge (roll late, keep distance), not predict exactly.
2. Raw pose does not transfer: a model fitted on a few bosses memorises each boss's look and is worse than chance on new ones. This is the cross-boss generalisation problem in its purest form. Humans rely on invariant cues ("the arm draws back", "the weapon closes on me"), and those cues transfer a little.
3. The next steps follow from this: train on hundreds of bosses (cheap here), use invariant cue features, and model timing as a distribution (a hazard over windup duration) rather than as a point prediction.

## B9 · Agents v2/v3 and the information ceiling
| Player (12 held-out bosses, verified attacks) | Win | Boss HP removed |
|---|---|---|
| random | 0% | 7% |
| aggressive | 0% | 20% |
| cue reflex (rolls when a perceived hand closes fast) | 0% | 7% |
| agent v2 (threat net + per-boss Bayesian head), best attempt | 0% | 38% |
| **state oracle**: perfect view of the boss's current state, imagines 3 futures with fresh randomness | **0%** | 29% |
| clairvoyant oracle: same logic, but sees the true future | 50% | 80% |

- Threat net v3 (cue features only, 150 bosses, 727k samples): AUC 0.72. More bosses did not help.
- The state-oracle control is the key result. Even perfect perception wins 0/12, so the bosses were too hard for *any* perceiving player. My agents were not the only problem.
- Rebalance: boss HP 350 + 150 × size, player hit 50, 4 flasks. After it: clairvoyant 75%, state oracle 8%, aggressive 0%.
- Requiring windups of at least 12 frames (360 ms, above human reaction plus roll start-up): clairvoyant 58%, state oracle 0%.
- Check: imagined futures do track the truth. Over the next 14 frames, mean imagined risk is 0.70 before real hits and 0.05 otherwise. So the state oracle's failure is its decision rule (false alarms drain stamina; rolls into follow-ups), not missing information. Re-running with 5 futures and a 0.4 threshold.
- Next: PPO restarted on the rebalanced bosses. A hit-risk planner with proper stamina accounting is needed before agent comparisons mean anything.
