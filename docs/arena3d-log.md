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

## B10 · Gap audit: Arena3D vs Elden Ring, mechanic by mechanic (main finding of this entry)
"Have" = implemented and checked; "Partial" = present but simplified; "Missing" = absent.

| Mechanic | Elden Ring | Arena3D now | Status | Plan |
|---|---|---|---|---|
| Space / movement | continuous 3D, free movement, spacing decides everything | continuous 3D plane, 8-direction lock-on movement at walk speed | Partial | add run / sprint, 16 directions, free (non-lock-on) mode |
| Terrain / arena bounds | walls, pillars, slopes, fog gate | flat ±15 m square | Missing | add walls (corner traps) and a round arena |
| Lock-on and camera | lock-on toggle, camera lag, boss leaves the screen when close or large | always locked, camera behind the player | Partial | camera lag, field of view, boss partly off-screen when close (pixel track) |
| Stamina economy | regenerates; roll, attack and block drain it; guard break at 0 | same structure, values hand-set | Have (values unvalidated) | calibrate against footage |
| Roll timing and i-frames | ~13 i-frames at 30 fps-equivalent for medium roll (to be measured), recovery depends on load | i-frames on frames 2–11, recovery 16–20 frames | Partial | measure from footage; light / medium / heavy load |
| Hit detection | per-frame 3D hit volumes on weapons and limbs, multi-hit swings | per-substep 3D contact (weapon capsule vs 19 body capsules); one hit per move | Have / Partial | multi-hit per move (spins hit 2–4 times) |
| Poise and stagger (player) | hits stagger unless poise holds; hyper-armour on some attacks | hits interrupt attacks and heals only | Partial | poise meter and stagger animation |
| Poise and stance (boss) | stance break after enough damage, then a critical hit | none | Missing | stance meter, stagger window, riposte |
| Parry | timed parry window (e.g. 8 startup / 12 active frames for medium shields at 60 fps, from a community guide), two parries break Margit | none | Missing | parry with a window; boss parryability per move |
| Delayed attacks | delayed swings, feints, variable windup | log-normal windup holds before the fastest frame | Have | calibrate delay distributions from footage |
| Mixed-up / branching strings | fixed strings with branch points (e.g. staff combo → dagger or hammer) | combo matrix with peaked rows ("favourite follow-up") | Partial | explicit named strings with branch points (boss programs) |
| Combo enders and openings | some moves always end the string (Margit's hammer slam), leaving a punish window | idle recovery after a string with probability 0.85–1.0 | Partial | per-move "ender" flag and authored recovery |
| Tracking | turns during windup, often stops tracking before the swing | yaw tracking up to the danger frame, capped per frame | Have | per-move tracking strength from footage |
| Leaps / gap closers | leap slams, lunges, backsteps with throws | lunges warped to reach; no leaps | Partial | airborne leap units (CMU jumps) with slam AoE |
| Projectiles / ranged | dagger throws, magic, punishes healing | none in Arena3D | Missing | projectile moves; heal-punish rule |
| AoE / shockwaves | hammer shockwave rings | none in Arena3D | Missing | ring hit volumes |
| Phases | phase 2 at ~60–65% HP with new moves | +20% speed below 50% | Partial | new move set in phase 2 |
| Input reading | heals and attacks get punished | none in Arena3D (was in Arena 2D) | Missing | port input-reading laws |
| Input buffering | buffered inputs during recovery | none (actions only when free) | Missing | short input buffer |
| Animation source | hand-authored combat animation | real human motion capture, mostly non-combat | Partial | combat mocap (blocked download; retry) and retiming to frame data |
| Boss variety | hundreds of authored bosses | unlimited procedural bosses from 14k mocap units | Different (advantage for generalisation studies) | keep as training distribution |

**Grounding source check.** Public boss frame data barely exists. The Fextralife Margit page gives a move list and structure but no frame counts: phase 1 opens with staff strings, then a delayed spinning windup, dagger throws (often punishing heals), a leaping stab and a tail swipe; phase 2 adds hammer moves that always end the string; HP 4,174 (NG). A community guide gives parry windows only. So the timings must come from **footage**: frame-by-frame annotation of windup, active and recovery per move. The perception tool (`scripts/video_perception.py`) can mark windup onsets and contact times semi-automatically once Aryan's recordings arrive.

## B11 · Margit-like program boss works; state oracle fixed; how early can a move be read?
- **State-oracle control fixed.** Using a 30-frame horizon for the attack decision (an attack commits the body for ~30 frames) moved the state oracle from 0–8% to **50% wins** on the 12 held-out bosses (6 wins, 4 timeouts, 2 deaths), against 75% for the clairvoyant oracle. So the gap between "knows the boss's current state perfectly" and "knows the future" is 25 points: that is the part of the problem that is genuinely unpredictable (hold lengths, next-move draws).
- **Boss programs (`arena3d/program.py`).** Margit-like boss built from held-out mocap: named moves (staff sweep ×2, staff thrust, delayed overhead with a 24-frame median delay and mix-up, dagger throw, leaping stab), strings with a branch point (after the sweeps: thrust or delayed overhead), ender moves followed by recovery, a heal punish (dagger throw with p = 0.7), and a phase-2 set (hammer slam with a 4 m ring, sword spin, hammer leap with a 5 m ring). Checked with a kiting, healing player and infinite HP: phase 1 fired 22 throws (20 projectile hits), phase 2 fired hammer leaps (4 shockwave hits) and the new moves. Multi-hit per move now works through swing ids.
- **Move-reading study (`scripts/margit_reading.py`, 442 strikes, 6 attempts, aggro player, perceived keypoints with 6% noise, dropout, 180 ms latency, leave-one-attempt-out 5-NN).** Results:

| frames since onset | move identity acc. (chance .36) | danger-time MAE, frames (no reading: 13.7) |
|---|---|---|
| 4 | .34 | 15.2 |
| 8 | .40 | 14.9 |
| 12 | .66 | 6.4 |
| 16 | .70 | 4.6 |
| 24 | .61 | 4.8 |

  Danger offsets by move: sweeps 14–15 frames (no spread), thrust 49, delayed overhead 40 (IQR 27).
- **Reading.** Identity becomes readable about 12 frames after the move starts (6 frames of seen motion plus the 6-frame perception lag). That is *after* the sweeps have already landed (14 frames), but well before the thrust (49) and the delayed overhead (40). So pose reading alone can only answer the slow and delayed moves. The fast openers must be **predicted before they start**, from context: distance, the position in a string, the rhythm since the last swing. This is the same split a human player makes: you react to the delayed slam, you learn the timing of the fast string.
- Next: an episodic reader that learns this boss's timing between deaths from context + pose (`arena3d/reader.py`), compared with a memoryless control.

## B12 · Learning one boss between deaths: thresholds fail, episodic control learns to survive
- **Balance.** Margit-like at 1,100 HP: clairvoyant oracle 0/8 wins (timeouts, 44% damage in 169 s), state oracle 0/8, aggro dies in 10 s. At 700 HP and a 4.5-minute limit: clairvoyant 2/8 (26% HP left on average), state oracle 0/8 (33% left, 205 dodges, 93 hits taken). Lowered to 450 HP for the agent studies so the ceiling is informative; rerun pending.
- **Danger predictor calibration (offline, 30k frames, 6 attempts, leave-one-attempt-out kNN).** Label = a contact the player feels (hit, blocked hit, or passing through i-frames), base rate 10% for the roll window. AUC for "danger lands in frames 3–10": timing context alone (time since whoosh / grunt, distance, healing, phase) **.785**; perceived pose 78-d .66; pose PCA-6 .53; pose + context .70. Perceived pose at frame level adds nothing (noise 6% of height swamps per-frame motion); consistent with the earlier threat-net result. Identity is readable only when the pose window is aligned to the move onset (B11).
- **Threshold reader (`EpisodicReader`).** kNN predictor + roll / attack / heal thresholds, 10 attempts, roll thresholds .15/.2/.25, contact or near-miss labels, with or without counterfactual spacing ("if I were in reach now, would I get hit?"): no trend in boss damage or survival in any variant. Thresholds tuned on one base rate break when labels change (near-miss labels made it stop attacking entirely).
- **Episodic control (`EpisodicControl`, after Blundell et al. 2016 / Lengyel & Dayan 2007).** Per action (4 moves, light attack, heal, 4 roll directions), remember (context, own hp / stamina / flasks) → discounted return over the next 60 frames (damage dealt − damage taken − death); act by the best kNN-estimated return, ε = 0.1, optimism for untried actions. 25 attempts, one series each:

| | survival frames, attempts 1–5 | 6–15 | 16–25 | boss damage 1–5 | 16–25 |
|---|---|---|---|---|---|
| learn (memory kept across deaths) | 1,984 | 1,948 | **2,507** | 3.4% | 7.4% |
| forget (memory wiped each attempt) | 949 | 1,021 | 900 | 0.0% | 1.1% |

  Memory across deaths makes it survive 2.5× longer than the memoryless control by the end, and it starts to deal damage; it has not learned to win yet. One series per condition, so this is a pilot; the replicated run (2 series × 30 attempts, 450 HP) is running.

## B13 · Replication at 450 HP: memory across deaths nearly triples survival
- **Controls (8 fights each, 4.5-min limit):** clairvoyant oracle **75%** wins (1% HP left), state oracle **25%** (19% left), aggro 0% (dies in 10 s). The ceiling is now informative, and the 50-point gap between knowing the present and knowing the future is the boss's real unpredictability (delays, branches).
- **Episodic control, 2 series × 30 attempts per condition:**

| | survival frames, att. 1–5 | 11–20 | 21–30 | boss damage 1–5 | 21–30 | wins |
|---|---|---|---|---|---|---|
| learn (memory kept) | 1,274 | 2,542 | **2,832** | 5.3% | **14.2%** | 0 |
| forget (wiped each attempt) | 943 | 987 | 998 | 0.9% | 3.6% | 0 |

  The pilot replicates: keeping memory across deaths gives 2.8× the survival and 4× the damage of the memoryless control by attempts 21–30, and the curve is still rising. It has learned to survive more than to punish. Next: 60-attempt runs with damage weighted 1× and 3× in the return.

## B14 · 60 attempts: survive-or-punish trade-off; stacking the hazard model
- **Episodic control, 60 attempts, 2 series each** (means per block of 10 attempts):

| return weight on damage dealt | survival frames: 1–10 | 21–30 | 51–60 | boss damage: 1–10 | 21–30 | 51–60 |
|---|---|---|---|---|---|---|
| 1× | 1,568 | 2,832 | 2,480 | 5.8% | 14.2% | 21.8% |
| 3× | 1,769 | 943 | 849 | 14.2% | 22.7% | 37.3% |

  No wins in either. With damage at 1× the agent learns to survive and slowly deals more damage, plateauing near 22%. At 3× it trades its life for damage: 37% damage but dies 3× faster. The state oracle reaches 81% damage at 25% wins, so the agent is about a quarter of the way to the perfect-present-knowledge ceiling.
- **Diagnosis.** Episodic control's state holds only timing context and own body. It cannot tell *which* move is coming, so the same context leads to a good outcome after a sweep and a bad one after a delayed slam. The value estimates average across moves, and the safe answer is to back off.
- **Running:** a stacked agent that adds denoised arm-motion energy and the hazard model's forecast (learned from contacts and near misses) to the episodic state, damage weight 1× and 2×.
- **Stacked agent result (60 attempts × 2):** damage 1×: survival rises to 3,620 frames (attempts 11–20) but damage only reaches 12%. Damage 2×: 21–28% damage, survival falls back to ~950 frames. No wins. Hazard features make it more cautious, not better at punishing. Episodic control is plateauing well below the oracle, so I'm switching to a decomposition of the gap instead of more tuning.

## B15 · Where does the oracle's advantage come from? A perceptual-limits oracle
- **Bug found and fixed:** `Arena3D.clone` shared the program boss's string queue with the real boss, so the oracles' imagined futures *consumed the real boss's planned moves*. Projectiles and rings were also not copied. Every Margit oracle number before this entry is affected; rerunning all of them.
- **New control `perceptual_oracle`:** the state oracle, except imagined futures cannot see what perception can't: (a) the remaining length of a held windup is resampled from the boss's delay distribution (including the no-delay mix-up), and (b) for the first 12 frames after onset (the readability limit from B11) the move's identity is resampled from the boss's moves. Ablation `perceptual_oracle_delay_only`: only (a).
- This splits the gap into three parts: future randomness (clairvoyant → state), hidden delays (state → delay-only), and late move identification (delay-only → perceptual). Whatever is left between the perceptual oracle and a learning agent is the cost of learning the boss model.
- **Result (8 fights each, Margit-like, 450 HP, queue bug fixed):**

| control | what it knows | wins | boss damage | survival (s) | hits taken / fight |
|---|---|---|---|---|---|
| clairvoyant | the true future | 1/8 | 70% | 105 | 10.3 |
| state oracle | exact current state, future resampled | 0/8 | 61% | 145 | 11.6 |
| delay hidden | + remaining windup delay unknown | 0/8 | 49% | 149 | 11.8 |
| perceptual | + move identity unknown for 12 frames | 0/8 | 29% | 150 | 11.4 |

  With the bug fixed the boss really runs its strings, and it is harder: the clairvoyant's win rate fell from 75% to 12%. Earlier Margit oracle numbers (B12, B13) were inflated by the bug. The episodic-control runs never used `clone`, so they are unaffected.
- **Decomposition of the 41-point damage gap between clairvoyant and perceptual:** future randomness 9 points, hidden delays 12 points, late move identification **20 points**. Reading the move earlier is the biggest single lever. That points at anticipation from context (string position, distance, rhythm) rather than better pose estimation.
- **Agent vs ceiling.** Episodic control after 50+ attempts deals 22% (damage 1×) to 37% (3×). The perceptual oracle, which has a perfect simulator but realistic perception, deals 29%. So the learning agent already matches the perception-limited planner on damage, though the planner survives 5× longer. The headline question becomes: can a learned boss model plus planning beat the perceptual limit by anticipating moves before they are readable?
- Caveat: the oracles share a crude hand-written action layer (fixed roll and attack thresholds) and take ~11 hits per fight even when clairvoyant. Their absolute numbers are a weak ceiling; the differences between them are the result.

## B16 · Knowing the boss's strings recovers most of what late reading costs
- **`grammar_oracle`** (perceptual oracle + the true string grammar: mid-string moves known except at branch points, openers drawn from the strings valid at this distance and phase): **46% boss damage**, 0/8 wins, 145 s survival.
- Against the decomposition in B15 (delay-hidden 49%, perceptual 29%), knowing the program recovers **17 of the 20 points** lost to late identification. Anticipating from the grammar almost fully substitutes for reading the move. This is the quantitative case for a learned boss *program* model: perception can't get the move in time, but the program can.
- **Running:** `LearnedGrammarPlanner` learns those strings between deaths. Each identification (12 frames after onset, 70% accurate as measured in B11) updates Dirichlet counts over (previous move or opener, near/far, phase) → move, and the planner imagines unreadable moves from the counts. Counts are kept or wiped across 8 attempts, 2 series each. Prediction: damage climbs from ~29% (uniform) toward ~46% (true grammar) with memory, and stays flat without it.
- **Result, learned grammar (8 attempts × 2 series):** damage by attempt, learn .27 .13 .36 .13 .18 .31 .27 .40 (mean .26); forget .27 .27 .44 .18 .18 .40 .31 .49 (mean .32). No wins, and no difference. Two reasons it can't show one yet: (1) "forget" still learns *within* each fight (13–24 contexts filled per attempt), so it isn't memoryless; (2) one player hit takes 8.9% of the boss's HP, so damage per fight moves in coarse steps and 16 fights per arm can't resolve a ~15-point effect. B16's grammar-oracle gap (46% vs 29%, 8 fights each) has the same power problem.
- **Next:** firm up B16 first with 16 more paired seeds per arm (perceptual vs grammar oracle), then test the learned grammar frozen after training, on the same seeds.

## B17 · Confirmed with paired seeds: knowing the strings is worth +12 points of damage
- 16 new seeds (8–23), same seed for both arms: perceptual oracle **21.1%** boss damage, grammar oracle **33.3%**. Paired difference +12.2 points, bootstrap 95% CI [+1.1, +22.8], one-sided Wilcoxon p = .025. Pooled with B16's 8 seeds: 23.7% vs 37.5%.
- So a planner that perceives realistically but knows the boss's strings does reliably better than one that guesses unreadable moves uniformly. The effect is smaller than the 8-fight pilot suggested (+17), which is what you'd expect from a small pilot. Neither wins: the shared hand-written action layer still takes ~11 hits per fight.
- **Running:** strings learned by *watching* (2 × 9,000 frames of a circling player, near and far, phases 1 and 2; identifications 70% accurate at 12 frames), then frozen and used by the same planner on the same 16 seeds. If it lands near 33%, the program is learnable from noisy perception alone.

## B18 · Strings learned by watching are as good as the true program (main result of this chapter so far)
- **Learning by watching.** 18,000 frames of a circling player (near and far spacing, phases 1 and 2), each move identified 12 frames after onset with 70% accuracy, Dirichlet counts over (previous move, near/far, phase) → move. The learned table recovers Margit's structure without being told it: after `staff_sweep` → `staff_sweep_b` (14 of 23 near, phase 1); after `staff_sweep_b` → `staff_thrust` (7) or `delayed_overhead` (6), the branch point; far openers → `dagger_throw` / `leaping_stab`; phase-2 far → `hammer_leap`.
- **Planning with the frozen learned strings, same 16 seeds:**

| planner (all with realistic perception) | boss damage | vs perceptual |
|---|---|---|
| perceptual (unreadable moves guessed uniformly) | 21.1% | |
| **learned strings (watched, frozen)** | **32.8%** | +11.7, CI [+1.1, +22.2], p = .032 |
| true strings (grammar oracle) | 33.3% | +12.2, p = .025 |

  Learned vs true: difference 0.6 points, p = .95, CI [−13, +15].
- **Reading.** Strings learned from noisy, late, 70%-accurate perception give the same benefit as knowing the true program. The bottleneck identified in B15 (moves can't be read in time) is solvable by learning the boss's program from observation, and this boss's program is learnable from about 10 minutes of watching. This is the concrete version of the project thesis ("reading the boss" = inferring its program), now on a 3D mocap boss with real hitboxes and realistic perception.
- **Limits.** Physics and hitboxes of each move are taken from the simulator (only the *choice* of move is learned). The action layer is hand-written and weak (no wins). One boss. Next: (1) learn move kinematics too, by imagining with the agent's own library of observed move trajectories instead of the true simulator; (2) more bosses (a Godrick-like program); (3) a better action layer (search over own action sequences) so the ceiling stops being the bottleneck.

## B19 · Closest published comparison; phase-shift recovery experiment
- **Najar (2026), "Learning Transferable Skills in Action RPGs via Directed Skill Graphs and Selective Adaptation", arXiv 2601.17923** (verified from the abstract and PDF). Trains live in Dark Souls III against Iudex Gundyr from a 25-d memory-readout state (no pixels). Five DQN skills (camera, lock-on, movement, dodge, heal/attack), trained in a sequential curriculum and run concurrently. Phase 1: 44% wins (25 episodes) after ~230k steps; an end-to-end DQN baseline learned no reliable combat by ~250k. Phase 2 treated as a second domain: 33% (mid-range starts) / 12.5% (long-range) zero-shot, 52% after fine-tuning only dodge and heal/attack with a "limited" budget (no step count given). No runtime phase detection.
- **How we differ.** Their agent reads game memory and adapts by gradient fine-tuning over many steps. Ours perceives from pose keypoints with realistic noise and latency, and adapts by updating an explicit boss-program model from a handful of observations. Their result is the right foil for "how many deaths to recover after a phase change". SoulsGym / DS3 is not available (Aryan owns Elden Ring and Hollow Knight only), so the real-game timing check stays on Elden Ring footage.
- **Our phase 2 already switches the moveset** (not only timing): hammer slam with a 4 m shockwave, sword spin, hammer leap with a 5 m shockwave, and new strings (sweep → spin → spin, sweep → slam).
- **Running (`scripts/margit_phase2.py`):** strings learned by watching phase 1 only; then 6 fights that start in phase 2. Conditions: frozen phase-1 counts; online updating from perception across attempts (the slow mind); true program (ceiling). Metric: share of phase-2 HP removed per attempt. A PPO fine-tune arm is next: PPO from scratch made no progress in 1M frames here (0 wins), so the comparison will start PPO from a phase-1-trained policy.
- **Result, phase-shift recovery (fights start in phase 2; one player hit = 18% of phase-2 HP, so the metric is coarse):**

| condition | fights | phase-2 HP removed, by attempt | mean |
|---|---|---|---|
| frozen (phase-1 strings only) | 6 | 0 .18 0 0 0 0 | **3.0%** |
| online (slow mind: keeps updating) | 2 × 6 | .18 .54 .09 .36 0 .18 | **22.7%** |
| true program | 6 | .18 0 .36 0 .36 0 | 15.1% |

  Online > frozen, Mann-Whitney one-sided p = .026; online vs true program p = .62 (no difference). The slow mind collected 26–66 phase-2 identifications during its *first* phase-2 fight and was already at ceiling level in that fight. **Deaths to recover ≈ 0–1**: it adapts to the new moveset within the fight where it first sees it. For comparison, Najar (2026) recovered phase 2 by gradient fine-tuning two skills with an unreported "limited" budget. Small samples; a 3-series replication and the PPO fine-tune arm are next.

## B20 · Replacing hand-set thresholds with search over own actions: first wins against Margit
- **`mpc_planner`** keeps the same perception limits (hidden delays, move identity unknown for 12 frames) and the same imagined futures, but instead of fixed roll/attack thresholds it evaluates 9–10 macro-actions (wait, approach, retreat, strafe ×2, light attack, roll ×3, heal when low). Each gets 3 imagined futures of 24 frames, scored as damage dealt − 2.5 × damage taken − 100 for death. Movement is held for 6 frames.
- **Pilot, true strings, seeds 8–11: 4/4 wins** (81 s average, 2.75 hits taken per fight, 24 dodges). The threshold planner with the same knowledge lost all 4 seeds (22–44% damage). The action layer, not the boss model, was the bottleneck since B15.
- **Running:** 12 paired seeds × {uniform guess, strings learned by watching, true strings} with the MPC action layer. Win rate and time-to-kill now resolve differences that damage alone couldn't.
- **PPO on the same boss (phase 1, from scratch):** 1.6M frames, 263 fights, 0 wins, 0.0% average boss damage (entropy has collapsed to 1.4). Still running; it is the gradient-learning arm for the phase-change head-to-head.

## B21 · With the MPC action layer: true strings 11/12 wins, learned strings don't transfer yet; label-free discovery
- **12 paired seeds (8–19), MPC action layer, realistic perception limits:**

| boss model in imagination | wins | per seed | mean frames to kill (wins) | hits taken |
|---|---|---|---|---|
| uniform guess for unreadable moves | 7/12 | `....WWWWWW.W` | 2,339 | 48 |
| strings learned by watching (frozen, 70% labels) | 7/12 | `..WWW..WWWW.` | 1,782 | 51 |
| true strings | **11/12** | `WWWWWWW.WWWW` | 2,182 | 38 |

  True strings vs uniform: Fisher one-sided p = .077 (12 per arm). The learned table, which matched the true program under the threshold planner (B18), gives no gain under MPC. MPC exploits the program more sharply, so errors in the learned table matter more. Suspects: 30% wrong labels corrupt the "previous move" context, the opener/continuation split rarely fires (the 45-frame gap rule), and the watching player's spacing differs from MPC's close-range play. **Running:** learned with perfect labels (isolates label noise) and learned with in-fight updating.
- **R1 label-free move discovery (offline, 442 perceived strikes, `scripts/r1_discovery_offline.py`).** DP-means on onset-aligned perceived keypoint windows (PCA): NMI .16–.31 with true moves, and it over-splits. Adding *when the swing was heard* (onset → whoosh) as a second modality: **NMI .50**, purity .73, 7 clusters for 4 true moves (two of which are near-identical sweeps). Retrospective, multimodal labelling (vision + sound timing, after the move ends) is how R1 should work. Online prediction needs early reading, but the grammar can be learned from labels assigned after the fact.

## B22 · Timing grounded in public Elden Ring footage: gap table and calibrated Margit
- **Source.** Aryan doesn't want to record, so I used public no-commentary videos, downloaded on his laptop because this machine can't reach YouTube: 3 Margit fights (60 fps), 2 Godrick fights (60 fps), and a Margit moveset showcase (30 fps, the player mostly passive), all 720p. Sources and URLs are in `data/footage/sources.csv` on the laptop; the videos are not committed.
- **Automatic HUD/motion timelines (`scripts/footage_timeline.py`)** were too noisy to time moves. The HP bar is semi-transparent and compressed, and whole-screen motion mixes camera, player and effects (peaks every 26–110 frames depending on the video). So timing comes from **frame-by-frame annotation** of filmstrips (`scripts/filmstrip.py`, every 4th frame at 30 fps, so ±4 frames ≈ ±0.13 s).
- **Annotated (moveset video):** staff overhead slam, raise → impact ≈ 16–20 frames (0.55–0.65 s); staff combo, hit-to-hit ≈ 16–20 frames (0.55–0.65 s); leap slam, takeoff → impact ≈ 39 frames (1.3 s), then ≥45 frames (1.5 s) on the ground before acting; summoned golden-hammer sweep, glow → swing ≈ 36 frames (1.2 s), recovery ≈ 44 frames (1.5 s).

| move | real (s) | sim (s) | ratio | flag (>2× off) |
|---|---|---|---|---|
| staff swing windup | 0.55–0.65 | 0.45 | 0.7–0.8 | |
| staff combo hit-to-hit | 0.55–0.65 | ~0.88 | 1.4–1.6 | |
| overhead slam (no delay) | 0.55–0.65 | 0.58 | ~1.0 | |
| summoned-hammer sweep windup | ~1.2 | 1.16 | ~1.0 | |
| leap slam takeoff → impact | ~1.3 | 0.63 | **0.48** | **yes** |
| recovery after hammer moves | ~1.5 | ~0.5–0.9 | **0.3–0.6** | **yes** |

- **Calibration (`MARGIT_MOVES_CAL`, `MARGIT_CAL=1`):** hammer leap delay 8 → 30 frames (total ≈ 1.3 s); new per-move frozen recovery (the punish window): 25 frames after the hammer moves, 12 after the other enders. Unflagged moves are unchanged. Everything before this entry used the uncalibrated table.
- **Limits.** One annotator (me), one moveset video, a handful of instances per move, ±0.13 s precision. Staff thrust, dagger throw and spin were not found cleanly in the strips. Godrick is not annotated yet.

## B23 · R3: planning with a learned world model instead of the simulator
- **Model (`arena3d/worldmodel.py`).** Per identified move, a spatio-temporal danger map p(contact | move, frames since onset, my position in the boss's onset frame), learned only from felt contacts (hit, blocked, passed through i-frames) and exposures. Bins: 3-frame time × 0.5 m rings × 8 bearings. Planning imagines only the player's own body (self-model: walk, roll with i-frames 2–11, attack/heal commitment) against the maps. Move identity comes from learned strings until readable (12 frames, 70%). **No boss simulator in the loop**, and decisions are ~100× cheaper than simulator rollouts.
- **Learning:** 2 × 9,000 frames of a probing player (spacing 1.5 / 3 / 5 m, random rolls, phases 1 and 2): 10,954 exposures, 167 contacts. The maps and strings keep learning during fights.
- **v1, 12 fights (seeds 8–19, the paired set): 0/12 wins, 32.6% boss damage**, deaths after 17–108 s. That matches the threshold planner that had the *true simulator and true strings* (33.3%, B17) without either, but is well below the simulator-based MPC (7–11/12 wins, B21). Failure mode: it walks into openers, because it only imagined danger *during* a move it had already seen start.
- **v2 (running):** opener anticipation (imagine the next move starting within 0–15 frames, identity drawn from the strings given the current context) and a higher cost on being hit.
- **v2 result:** 0/12 wins, 32.6% damage again, with more variance (three fights at 80%). **Diagnosis (3 fights instrumented):** of 33 hits taken, 20 were dagger throws, and 29 came while standing or walking. The danger maps stopped at 6 m, and throws come from ≥ 4–5 m, so a player standing farther away saw **zero** modelled danger and was free to be shot. (Heals also trigger the throw: input reading.) v3: maps out to 12 m (0.75 m rings), and the probing player also holds 8 m spacing.
- **v3 (maps to 12 m):** 0/12, 17.0% damage, but survival up from ~1,600 to ~2,550 frames.
- **v4 (kernel-smoothed maps, so sparse contacts generalise to neighbouring times, distances and bearings):** σ = 1 bin: **40.0% damage**, survival ~4,500 frames (2.8× v2), 0/12 wins. σ = 2: 35.6%. Smoothing helps; over-smoothing blurs hit volumes.
- **Running: experience scaling.** Same model, watching budget 2,250 / 9,000 (done) / 18,000 frames per phase: does performance scale with experience, and how fast?
- **Experience scaling (σ = 1, 12 fights each):**

| watching per phase | exposures | contacts | boss damage | mean fight length (frames) |
|---|---|---|---|---|
| 2,250 | 3,333 | 55 | 30.4% | 3,343 |
| 9,000 | 13,528 | 202 | 40.0% | ~4,500 |
| 18,000 | 27,182 | 385 | 35.6% | 3,789 |

  It improves from little to moderate experience, then plateaus around 35–40% (±~7 points of noise at 12 fights). More data of the same kind doesn't fix it, so the bottleneck is the model class: a per-move map in the onset frame can't represent hit volumes that move with the boss's lunge and tracking. Next modelling step: maps in the boss's *current* frame (perceived hip each frame) and per-move lunge displacement.

## B24 · Cross-boss library on procedural bosses (R5), running
- `scripts/general_learner.py`: 6 held-out procedural bosses (random moves, strings, branches, enders, rings, projectiles, phase-2 switch, random weapon and scale). Per boss: scout 1,500 frames per phase, then 2 fights with the learned-world-model planner (no simulator). Conditions: K = 0 (learn from scratch) vs K = 12 training bosses pooled into a library of danger maps by move prototype (shape, ender, ring, projectile, leap), injected as pseudo-counts. Question: does a library make a new boss faster to learn?
- **v5 (maps in the boss's *current* frame + learned mean boss travel per move, σ = 1):** 0/12 wins, 35.6% damage, but mean fight length **5,976 frames** (3 timeouts at 9,000) against ~4,500 for v4 and ~1,600 for v2. Modelling the boss's travel mostly bought survival. The planner is now too cautious: its hit cost (200) vs damage gain (40) is about 5:1. A rough value-of-a-hit calculation (one boss hit ≈ 9% of the win, one player hit ≈ 35% of life, softened by 4 flasks) suggests ≈ 1.5–3:1. **Running:** hit cost 80 and 120.
- **Result (v4 world model, hit cost 200, 6 bosses × 2 fights):** K = 0: 1/12 wins (**the first win without any simulator in the loop**, procedural boss 3), 29.9% damage, 3,582 frames. K = 12 library: 0/12, 23.0% damage, 4,294 frames. The library made the agent survive longer but not deal more damage. No transfer benefit at this sample size, and bosses 4–5 were never damaged under either condition. Likely cause: maps are in metres, but bosses differ in scale (1.4–2.4×) and weapon reach, so pooled hit volumes are misaligned. Next: maps in boss-height units, so a sweep is a sweep regardless of size, and retest with v5 and the tuned hit cost.
- **Hit-cost sweep (v5, 12 fights each):** cost 80: 23.7% damage, 2,078 frames, 12 deaths; cost 120: 35.6%, 4,192 frames; cost 200: 35.6%, 5,976 frames. Being less cautious doesn't convert into damage, it just dies sooner. The limit is the accuracy of the learned danger model at close range, not the risk preference. Keeping 200.
- **Running:** library retest (B24 design) with the v5 model and boss-size units.
- **Library retest (v5, boss-size units, 6 bosses × 2 fights):** K = 0: 25.2% damage, 5,397 frames, 0 wins. K = 12: 17.8%, 4,616 frames, 0 wins. Still no transfer: the library helped on boss 0 (80–98% vs 62–89% damage) and hurt or did nothing elsewhere. Matching moves across bosses by authored prototype flags (shape, ender, ring...) is too coarse, because two "sweeps" from different mocap clips have different hit volumes. Transfer will need matching by *observed* kinematics (R1 clusters), not labels. Recorded as a negative result.

## B25 · Learning from what you see, not only what hits you (R3 v6), running
- Felt contacts are sparse: ~200 in 18k frames. A player also learns from *watching the weapon*: every swing that passes near you shows its reach and timing. v6 (`WM_GEOM=1`) stores each move's perceived weapon-tip trajectory (tip keypoint with 6%-of-height noise: a weapon-aware detector, which the MediaPipe pipeline doesn't have yet) relative to the boss. Hazard = the fraction of remembered trajectories whose tip passes within reach of the imagined player position at that moment, combined with the felt-contact map.
- **v6 result (tip trajectories + felt contacts, 12 fights):** 0/12 wins, **42.9% damage**, the best simulator-free result so far (v5: 35.6%), with shorter fights (3,156 frames). Seeing the weapon makes it more willing to engage where swings don't reach.
- **v7 (running):** hazard from the distance to the whole *blade segment* (perceived hand → tip), not only the tip point, reach 0.4 m.
- **v7 result (blade segment, reach 0.4 m):** 12.6% damage, 6,844 frames (3 timeouts). Far too cautious: the hand is always near the boss's body, so the whole approach zone looks dangerous. Negative; v6 (tip only) stays the best simulator-free model.

## B26 · Learned strings that keep updating during the fight match the true program under MPC (main result)
- Same 12 paired Margit seeds (8–19), MPC action layer, realistic perception limits:

| boss model in imagination | wins | per seed |
|---|---|---|
| uniform guess | 7/12 | `....WWWWWW.W` |
| learned by watching, frozen, 70% labels (B21) | 7/12 | `..WWW..WWWW.` |
| learned by watching, frozen, perfect labels | 10/12 | `WdWWWdWWWWWW` |
| **learned by watching + keeps updating in the fight, 70% labels** | **11/12** | `WWWWWWdWWWWW` |
| true strings | 11/12 | `WWWWWWW.WWWW` |

- The frozen learned table failed under MPC (B21) for two reasons this run separates. Label noise costs about 3 wins (perfect labels: 10/12). Context mismatch between watching and fighting is the rest: updating the counts from what the boss does *in this fight*, with the same noisy 70% labels, reaches the true-program level. Online vs uniform: Fisher one-sided p = .077 per arm. Pooled, the three program-aware arms win 32/36 vs 14/24 for uniform + frozen-noisy (Fisher one-sided p ≈ .008).
- This is the clearest evidence yet for the project's thesis: an agent that keeps inferring the boss's program from noisy perception, while fighting, does as well as one handed the true program.

## B27 · Expectation shapes perception: the boss's strings sharpen label-free move discovery

**Question.** Without any move labels, can the agent learn the boss's strings over moves it discovered itself (R1)? And can knowing the strings help it see the moves better?

**Setup.** `scripts/r1_grammar_offline.py`, data/margit_reading.pkl (442 perceived strikes, 4 true moves, labels used only for scoring). Discovery: DP-means over perceived keypoints + swing timing (the B16 config, NMI .50). Strings: Dirichlet bigram (prev discovered move or "open", phase), leave-one-fight-out information gain over a unigram baseline.

**Result 1 (negative).** Strings over clusters found by kinematics alone carry almost none of the structure: best 0.07 bits vs 0.57 bits with true labels. Splitting finer makes it worse (overfit, negative gain). Cluster noise wipes out the sequence signal.

**Result 2 (positive).** Expectation-guided discovery: hard-EM where each swing's label maximises kinematic fit + g·[log P(label | previous swing) + log P(next swing | label)], i.e. perception that uses what the boss just did and did next.

| expectation weight g | K | NMI vs true moves | purity |
|---|---|---|---|
| 0 (kinematics only) | 7 | .499 | .729 |
| 0.5 | 7 | .559 | .736 |
| 1.0 | 6 | **.605** | .759 |
| 2.0 | 5 | **.605** | .763 |
| control: shuffled sequence links, g = 0.5–2 | 6–7 | .45–.48 | .65–.67 |

The real sequence lifts label-free move identification by +.11 NMI; a fake sequence of the same density lowers it. So the boss's grammar is a genuine perceptual prior, not extra parameters. (The in-sample info gain of these labels, .65 bits, is biased upward because the labels were fit using transitions; NMI is unbiased.)

**Meaning.** A label-free learner should discover moves and strings *jointly* (an infinite-HMM-style loop), not in sequence. This is the R1↔R2 link in the plan, and it is how a human learns a new boss: "that's the follow-up to the thrust, so it must be the sweep".

**Next.** Online version inside the fight (assign each new swing with the prior from the previous one), feeding a cluster-keyed world model; the general-suite runs for the from-scratch string learner are queued (B28).

## B28 · Generalisation suite: the planner stalls on unseen bosses; an engagement drive fixes the stall (running)
- Suite 1 (`general_suite.py`, 6 procedural bosses × 2 fights, 9000-frame cap, no drive), partial 18/24: bosses 2 and 3 are won by both MPC arms (true strings 4/4, guessing 3/4); bosses 0, 1 and 4 time out with 0–44% damage.
- Diagnosis on boss 0 (700 frames, mpc_uniform): the boss is mid-swing 68% of the time; the agent waits at a median 3.7 m (own reach 1.26 m), presses nothing 97% of frames, and deals 0 damage. With a 24-frame imagined horizon, closing in costs risk now and pays only after the horizon, so standing off always looks best. This is a planning-horizon failure, not a knowledge failure.
- Fix: an engagement drive. Imagined futures that end out of reach cost `w` per metre, and `w` grows with time since the agent last landed a hit (`MPC_CLOSE=3`, `MPC_HUNGER=1`: frustration turns into pressure, reset by each hit). Same 700 frames: median distance 1.0 m, 200 damage dealt, 34 HP lost.
- Suite 3 (queued): the drive on for guessing, true strings, from-scratch in-fight string learner, and the hierarchical back-off learner. Suite 1 is the no-drive ablation.
- **Result (suite 1 vs suite 3, same 6 bosses × 2 paired fights):**

| arm | no drive | engagement drive |
|---|---|---|
| guessing (uniform strings) | 3/12 wins, 40% dmg | **9/12**, 98% |
| true strings | 4/12, 42% | 8/12, 99% |
| learns strings from scratch in-fight | — | 8/12, 89% |
| from scratch + hierarchical back-off | — | **10/12**, 97% |

  Drive vs no drive, pooled: 35/48 vs 7/24 wins (Fisher one-sided p = .0005); every timeout disappears and wins take 400–2000 frames instead of 2000–6000. Among drive arms the differences are inside noise at n = 12 (hierarchical 10 vs guessing 9). The hard cases are bosses 4 and 5 (most deaths).
- **Reading.** On unseen bosses the binding constraint was motivation, not knowledge: an agent with a short imagined horizon needs a drive that turns lack of progress into pressure, much as a frustrated player starts trading hits. Program knowledge has to be tested where the drive alone is not enough: B29 runs 12 fresh bosses (7006–7017) to get the power to separate the learners.

## B29 · 12 fresh procedural bosses with the drive: the suite saturates
- Bosses 7006–7017 × 2 paired fights, `MPC_CLOSE=3 MPC_HUNGER=1`. Guessing 23/24 wins, true strings 21/24, from-scratch hierarchical learner 20/24 (all ~100% damage).
- **Reading (honest).** With a perfect per-move physics simulator inside the planner, the drive alone beats almost every procedural boss, so knowing the strings has nothing left to add. The extra stochasticity of string sampling may even cost a little (no arm differs significantly). This is a ceiling effect, and it says where the real generalisation test lies: an agent that must also *learn the physics* of each new boss (the simulator-free world-model planner, R3). B30 puts the drive into that planner.

## B30 · The simulator-free learner on unseen bosses: drive helps, shockwaves are a blind spot (running)
- `general_learner.py`, K = 0 (no library), 6 procedural bosses × 2 fights, 1,500 scouting frames, v6 world model (current frame + seen-tip geometry). The planner has to learn each boss's danger from what it sees and feels.
- Engagement drive (`WM_HUNGER=1`) vs none: damage 34.1% vs 20.7% (paired: 7 fights better, 1 worse, 4 ties at 0; Wilcoxon one-sided p = .074), 1/12 vs 0/12 wins. Bosses 4 and 5 are never damaged in either arm.
- Death diagnosis (`scripts/wm_death_diag.py`): on boss 5, 9 of 12 hits are the shockwave ring of one slam (m3, ring radius 3.8 m), landing at 2.7–6.5 m. On boss 4 the shockwave slam (m2) also dominates once the agent survives longer. Cause: the learner **stopped attending when the boss's swing ended**. Danger maps recorded contacts only while the boss was mid-strike, and the planner stopped imagining a move once its animation finished. A ring that keeps expanding after the swing, or a hit after recovery, was invisible to learning and to planning.
- Fixes, each behind a flag: `WM_PERSIST=1` keeps attending to a move's effects for its whole 90-frame window after onset, even after the body is done (object permanence for attacks). `WM_HIER=k` makes sparse places borrow the move's hazard at that moment pooled over all places (hierarchical back-off, the same idea as the string learner's). Factorial running on the same 6 bosses.
- **B30b result (same 6 bosses × 2 fights, all with drive):**

| arm | damage | wins | mean frames survived |
|---|---|---|---|
| no drive (h0) | 20.7% | 0 | 3360 |
| drive (h1) | 34.1% | 1 | 2172 |
| drive + attend after swing (p1) | 26.7% | 0 | 1685 |
| drive + attend after swing + hierarchical back-off (p1h4) | **38.8%** | 1 | 1787 |

  None of the pairwise differences is significant at n = 12 (best: p1h4 vs no drive, Wilcoxon one-sided p = .064; p1h4 vs h1, p = .23). Attending after the swing alone *hurt*: more recorded contacts at late taus make the planner see danger everywhere, and with too few data that pushes it into bad dodges. Back-off rescues it by pooling. First non-zero damage on boss 4 (27%, 18%) and boss 5 (9%). Every fight still ends in death: the simulator-free learner's bottleneck is survival, not aggression.
- **Next (B31).** The real souls test: does the learner improve *across attempts*, as a player does after each death? 6 attempts per boss, learning carried over, p1h4 config, learning curves of damage and survival per attempt.

## B31 · Learning across attempts (the souls loop) for the simulator-free learner
- 6 procedural bosses × 6 consecutive attempts, danger maps and strings carried across deaths (p1h4 config + drive). 5/36 wins, 43.6% mean damage (vs 38.8% over 2 attempts in B30b).

| attempt | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| mean damage | 28.7% | 48.9% | 42.1% | 50.2% | 47.4% | 44.5% |
| mean frames survived | 1464 | 2111 | 1874 | 2795 | 2400 | 2213 |
| wins | 0 | 1 | 1 | 2 | 0 | 1 |

- Per boss the picture splits. Bosses 0 and 1 show a human-like curve (36→53→62→107→98→98% and 53→71→18→62→98→107%, beaten on the last attempts). Boss 3 is beaten on attempts 2–4, then *regresses* (53%, 36%). Boss 2 *gets worse* (66→0%). Bosses 4 and 5 stay near the floor. No overall monotonic trend (Spearman ρ = .04); the gain is mostly first attempt → rest.
- **Hypothesis for the regressions: superstition.** Each fight adds contacts recorded under noisy (70%) move labels, plus hazard pooled by back-off, so danger leaks onto safe places and times, and the planner becomes over-cautious or mistimes rolls. Test (B32): bosses 2 and 3 with perfect move identification (`WM_ACC=1`). If regressions vanish, label noise drives them, and the fix is credit assignment (e.g. relabel past contacts once the move is identified with more confidence, or B27's expectation-guided labelling).

## B32 · Regressions are not label noise
- Bosses 0–3 × 6 attempts with perfect move identification (`WM_ACC=1`): mean damage 77/80/40/49% (vs 76/68/27/71% with 70% labels). Regressions persist: boss 0 107→53%, boss 3 107% (won on attempt 1, in 836 frames) → 27%, with late attempts ending in death after only 620–790 frames. Boss 2 improves (to 83% on attempt 5).
- So noisy labels are not why later attempts get worse. Two hypotheses remain: (a) later fight seeds are simply harder (each attempt uses a new seed), or (b) something accumulating in the learner degrades it. Control running: a fresh learner (same scouting, no carry-over) on every attempt's seed (`NOCARRY=1`). If fresh learners also fail on attempts 5–6, it's seed difficulty; if they beat the carried-over learner there, accumulation is the culprit.
- **Control result (fresh learner each attempt, `NOCARRY=1`, 6 bosses × 6 seeds):** 41.5% damage vs 43.6% when experience carries over (attempts 2–6, paired Wilcoxon p = .72). The fresh learner shows the same attempt-1 → attempt-2 jump (28.7% → 48.9%), so **the B31 "learning curve" was mostly seed difficulty**: attempt 1's seed is harder for every learner. Beyond the 1,500 scouting frames, more experience of a boss barely changes this learner (same plateau as B23 on Margit). Exceptions: bosses 0 and 1 end higher with carry-over (98–107% vs 18–53% on attempts 5–6), boss 2 ends lower.
- **Reading.** The simulator-free learner is *model-class limited*, not data-limited. Looking at how the bosses work points to one concrete gap: bosses keep **turning toward the player until the swing commits** (tracking up to 0.06 rad/frame, i.e. up to ~100° over a wind-up). The danger maps are in the onset frame, so the same move's hit volume is smeared over every angle the player might have rolled to, and a sideways roll that the boss tracks looks safe in the map.

## B33 · An aim model: imagine the boss turning toward you (running)
- `WM_TRACK=1`: positions are recorded in the onset-origin frame rotated by the boss's perceived turn since onset (body rotation, small noise). Two things are learned from watching: the boss's turn rate (90th percentile of seen per-frame turning) and each move's commit time (median last tau at which turning was seen). In imagination, for each candidate action, the boss turns toward the agent's *imagined* position at the learned rate until the learned commit time, and the danger maps and remembered blade paths are read in that turned frame. This is a small piece of theory of mind: the boss aims at where I am going to be, until it can't.
- Same 6 bosses × 2 fights as B30b (paired with p1h4, 38.8%).
- **First 6 bosses (paired with p1h4):** damage 48.9% vs 38.8%, wins 3/12 vs 1/12, frames survived 2232 vs 1787. Paired Wilcoxon one-sided p = .28 (7 better, 3 worse, 2 tied), so promising but underpowered. Replication on 12 more bosses (7006–7017), both arms, running.
- **Replication (12 more bosses) and pooled result, 18 bosses × 2 paired fights:**

| | onset-frame maps (p1h4) | aim model (track) | paired test |
|---|---|---|---|
| frames survived | 1761 | **2166 (+23%)** | Wilcoxon one-sided p = **.008** |
| damage dealt | 39.6% | 45.4% | p = .16 (19 better, 14 worse) |
| wins | 4/36 | 8/36 | n.s. |

  On the 12 new bosses alone: 43.7% vs 40.0% damage, 5 vs 3 wins. **Modelling the boss's aim makes the learner reliably harder to kill**; the damage and win gains go the same way but don't reach significance. The aim model is now part of the default simulator-free learner.
- **Where the remaining gap is.** The same learner with a perfect boss simulator wins ~90% of these fights (B29); with everything learned it wins 22%. Most deaths still come from strikes read late (12 frames) whose remembered paths are averaged over many aims and speeds. Candidate next steps: (1) condition remembered blade paths on the boss's speed in the first frames (the hold/delay variable that the footage calibration showed matters most); (2) amortise: train a small network on (pose window → future blade path) from all seen swings, shared across bosses, the R7 "intuition" component.

## B34 · Amortised intuition (R7) does not transfer from noisy pose (negative)
- `scripts/r7_collect.py`: 2,669 swings from 58 procedural bosses (train b = 100–139: 1,846 swings; test = the 18 suite bosses: 823 swings). For each swing: the perceived boss pose for the first k frames (sensor: 6% noise, 6-frame latency, dropout) and the weapon tip's future path over 90 frames, in the onset frame and boss-height units.
- `scripts/r7_train.py`, `scripts/r7_baselines.py`: mean tip-path error (boss heights) on the 18 unseen bosses.

| predictor | k = 12 | k = 20 |
|---|---|---|
| average path of all training swings | .586 | .586 |
| MLP, 40 training bosses (zero-shot) | .614 | .594 |
| ridge regression (zero-shot) | .567 | .563 |
| kNN-10 (zero-shot) | .608 | .601 |
| same boss, same move, mean of its other swings (needs identity + experience) | **.258** | **.258** |

- Even on held-out swings of the *training* bosses, ridge only reaches .49–.51 vs .52 for the average path. So with this sensor, 12–20 frames of pose say very little about where the blade will go, while knowing *which of this boss's moves it is* halves the error. Swing paths don't share a structure across bosses that noisy early pose can unlock; what transfers is the skill of building a per-boss episodic memory keyed by move identity, which is the design the learner already has.
- Running: the same with a clean sensor (no noise, latency or dropout), to separate a perception limit from a transfer limit.
- **Clean sensor (no noise, latency or dropout):** on held-out swings of *seen* bosses, kNN reaches .285–.307 (close to the same-move mean, .258) and ridge .34–.35, versus .49–.54 with the calibrated sensor. On *unseen* bosses the best is .525 (MLP, k = 20) vs .586 for the average path. So both limits are real: the calibrated sensor destroys most within-boss reading of blade paths from early pose, and even perfect pose transfers only weakly across bosses.
- **Perceptual denoising doesn't rescue it** (`scripts/r7_denoise.py`): projecting noisy poses onto a pose manifold learned from the agent's own observations (PCA 6–20 dims) plus a constant-velocity fit over the visible frames improves seen-boss kNN only from .522 to .481 (k = 20). The information is missing, not merely buried. Running: one-at-a-time sensor ablations (noise only / latency only / dropout only) to find which limit matters.

## B35 · What limits reading: keypoint noise, not latency
Seen-boss held-out swings, tip-path error (kNN-10 / best ridge), k = 20; average path .520:

| sensor | kNN | ridge |
|---|---|---|
| clean | .285 | .344 |
| latency 6 only | .297 | .355 |
| dropout only | .351 | .406 |
| **6% keypoint noise only** | **.503** | **.467** |
| calibrated (all three) | .522 | .492 |

Keypoint noise alone removes almost all of the reading; 6-frame latency costs nearly nothing; dropout costs some. Body posture carries the blade path only in fine detail (hand and forearm placement) that 6%-of-height jitter swamps, and averaging over frames (B34b) doesn't recover it. A weapon-aware detector should matter far more than a faster one: running the calibrated sensor plus a noisy weapon-tip keypoint (`SENSE_TIP=1`).
- **B36: a noisy weapon-tip keypoint doesn't help.** Calibrated sensor + tip as a 14th keypoint (same 6% noise): seen-boss ridge .486, kNN .509 (vs .492 / .522 without). Where the blade *is* during the first frames, seen through the same jitter, says no more about where it will go than the body does. Running a noise sweep (2%, 4%) to find how precise perception must be before early reading works.
- **Noise sweep (with latency and dropout), seen-boss kNN / ridge, k = 20:** 0%: .285 / .344 (clean, no latency/dropout) · dropout only .351 / .406 · **2%: .427 / .453** · 4%: .490 / .473 · 6% (calibrated): .522 / .492 · average path .520. Reading degrades steeply: even 2%-of-height jitter (≈ 4 cm on a 2 m boss) costs most of what dropout leaves. Early blade-path reading from pose needs near-motion-capture precision. With realistic perception, the agent must rely on move identity (a categorical read that survives noise, B11: 70% at 12 frames) plus a per-move episodic memory of paths, which is exactly the learner's architecture. This is a quantitative argument for that design.

## B37 · System 2 → System 1: distilling the slow planner into the fast policy
- `scripts/distill_bc.py`. Teacher: the perceptual planner with strings learned by watching phase 1 (B18/B19 slow mind), 32 phase-1 Margit fights through the PPO worker interface (same perceived observations, 14 candidate actions); teacher 0/32 wins, 44.2% damage. Student: the recurrent PPO policy trained by behaviour cloning (30 epochs, 64-decision chunks), 90.8% action agreement.
- **Student on its own (8 fights each):** phase 1 12.2% damage (teacher 44.2%; PPO from scratch 0.0% after 2.1M frames). Phase 2 (never seen in training) 24.9% of phase-2 HP, comparable to the slow mind's in-fight adaptation in B19 (22.7%) and above frozen strings (3.0%). Coarse metric (one hit = 18% of phase-2 HP) and small n, so treat phase 2 as "no worse".
- Reading. Cloning gets a reactive policy out of PPO's dead start but keeps only a quarter of the teacher's damage: 9% disagreement at the decisions that matter (when to commit an attack) compounds over a fight (the usual covariate-shift problem; DAgger would query the teacher on the student's own states). Running: PPO fine-tuned on phase 2 from the cloned policy, 300 updates, the gradient arm of the head-to-head.
- **PPO fine-tune on phase 2 from the cloned policy (300 updates, 2.4M frames, 16 envs):** recent phase-2 damage 14.1% at the start → 1.5% by 0.25M frames → 0.0–0.2% from 0.5M frames on, 0 wins. Fights stop ending in death or victory and run to the time limit. The gradient learner unlearns the cloned attacking and settles into passive survival (the hit penalty is dense and immediate; damage dealt is sparse and needs a committed approach), the same collapse PPO from scratch showed.
- **Head-to-head verdict (coordinator ask):** facing a new phase, the slow mind adapts *within the first phase-2 fight*, from 26–66 perceived identifications (22.7%, B19). The fast policy distilled from it starts at the same level (24.9%), and gradient fine-tuning then *destroys* the skill within 0.25M frames. With this reward and budget, model-based adaptation of beliefs beats gradient adaptation of a policy by a wide margin, in line with the course's sample-efficiency argument. Caveat: one PPO configuration, one seed; a KL-to-teacher penalty or reward rebalancing might stop the collapse.
- **DAgger (3 iterations × 12 student-driven fights, teacher relabels every visited state, β = .5/.25/.125):** pure-student phase-1 damage 11.1% → 10.0% → 10.0% (BC: 12.2%); phase 2 18–25%. Mixed rollouts still deal 27–46% because the teacher takes some of the decisions. So it isn't covariate shift. The teacher decides with information the student never gets: its imagined futures run on the boss's physics, and its string beliefs are explicit counts. The ~10% of decisions the student can't match from perceived features are the ones that carry the damage. Distilling a planner into a reactive policy needs the planner's *beliefs* as inputs (or a student that plans), not just more labels.

## B38 · Feeding the habit the slow mind's beliefs doesn't help (negative)
- Same distillation, but the student also sees the slow mind's beliefs each decision: the identified current move (after 12 frames), the learned strings' predicted move distribution for the current context, time since onset, and a strike flag (`BC_BELIEF=1`). During evaluation the beliefs keep updating online, as the slow mind's do.
- Action agreement 91.3% (vs 90.8%). Phase 1: 11.1% damage (vs 12.2% without beliefs). Phase 2: 0.0% (vs 24.9%): phase-2 belief vectors are outside anything seen in phase-1 training, and the habit misreads them.
- Together with DAgger (B37): what the student lacks is the planner's imagined physics (where the blade will be relative to me for each of my options), not its beliefs about which move is coming. Strings knowledge pays off only through a model that can roll it forward. That is an argument for keeping a planner in the loop, not just a habit trained from it.

## B39 · Risk trade-off sweep for the physics-learning agent
- Track config, 6 bosses × 2 paired fights, hit cost (penalty per unit of imagined contact probability) 120 / 200 / 300 / 500: damage 35.0% / **48.9%** / 39.3% / 31.9%, wins 1 / **3** / 1 / 0, frames survived 1636 / 2232 / 3069 / 2992. Higher cost survives longer but deals less; the default (200) is best for damage and wins. The risk weight isn't the lever.

## B40 · Hierarchical strings inside the physics-learning agent: no gain
- Track config with the hierarchical back-off string learner (`WM_HIERG=1`) on the same 18 bosses × 2 paired fights: 38.5% damage, 4/36 wins, 1995 frames survived, vs 45.4%, 8/36, 2166 with flat strings (Wilcoxon two-sided p = .21). Back-off helped the simulator planner in B28 (10/12 vs 8/12, n.s.); here it doesn't. The strings component isn't what limits this agent (consistent with B37–B38: physics is the bottleneck).

## B41 · Replication of B26 on 12 new Margit seeds (20–31)
| arm | B26 (seeds 8–19) | replication (20–31) | pooled 24 | vs uniform, Fisher one-sided |
|---|---|---|---|---|
| uniform guess | 7/12 | 8/12 | 15/24 | — |
| learned by watching, frozen | 7/12 | **11/12** | 18/24 | p = .27 |
| learned by watching, online in-fight | 11/12 | 10/12 | **21/24** | **p = .047** |
| true program | 11/12 | 11/12 | **22/24** | **p = .018** |

- **What holds:** knowing the boss's strings helps the planner. The true program and the online-updated learned strings both beat guessing over 24 seeds, and the learned-online agent comes within one fight of the true program (21 vs 22).
- **What doesn't:** B26's contrast between frozen and online learned strings. Frozen went 7/12 → 11/12 on the new seeds, and online vs frozen pooled is 21 vs 18 (p = .23). The "context mismatch" reading in B26 was over-read from one seed set. Results summary updated accordingly.

## B42 · 4× more scouting doesn't help the aim-model agent either
- Same 18 bosses × 2 fights, 6,000 instead of 1,500 scouting frames: 44.0% damage, 6/36 wins, 2412 frames (vs 45.4%, 8/36, 2166; Wilcoxon p = .64). Same plateau as B23 and B32: more data of the same kind doesn't move this learner.

## B43 · A motion-phase clock for delayed swings (running)
- Death diagnosis of the aim-model agent (`scripts/wm_death_diag.py`, 6 bosses, 59 hits): 76% of hits land on moves already identified, and ~60% while the agent was free to act. So knowledge of *which* move was there and the agent could have moved. Many land late in the move (40–96 frames after onset). Procedural bosses (like Margit) freeze in a log-normal windup hold before the fast part of a swing, so the same move's danger arrives at very different times since onset. Maps indexed by time since onset smear it, and the agent can't time a dodge.
- `WM_PHASEIDX=1`: index danger by **motion phase** (how far the swing has progressed through its motion; perceivable from pose progression, with the windup freeze seen as the body stopping), not by time since onset. Each move's hold lengths and hold point are learned from watching. In imagination, each of the 8 sampled futures draws its own remaining hold from the learned lengths, conditioned on how long the boss has already held. Timing uncertainty is represented while the boss is frozen and collapses when it resumes, which is the human strategy against delayed attacks ("wait for it, then roll").
- 18 bosses × 2 paired fights, against the aim-model agent (45.4%, 8/36 wins, 2166 frames).
- **Result (18 bosses × 2 paired fights):** 49.5% damage, 7/36 wins, 2237 frames, vs 45.4%, 8/36, 2166 for the aim-model agent (damage Wilcoxon one-sided p = .21, frames p = .26). Small and not significant on its own: only ~45% of procedural moves have a windup hold, and the hold matters only for moves that are dodged late. Kept as an option. Next: the whole stack vs the naive learner on all 18 bosses, to measure the total gain of B28–B43 in one paired comparison.

## B44 · Total gain of the general-learner stack on 18 unseen bosses
Same 18 procedural bosses × 2 paired fights, no simulator, no library, 1,500 scouting frames each:

| learner | damage | wins | frames | timeouts |
|---|---|---|---|---|
| naive (v6 world model, no drive) | 22.7% | 1/36 | 3945 | 4 |
| + drive, after-swing attention, back-off, aim model | 45.4% | 8/36 | 2166 | 0 |
| + motion-phase clock (full stack) | **49.5%** | **7/36** | 2237 | 0 |

Full stack vs naive: damage ×2.2 (Wilcoxon one-sided p = 2.6e-5), wins 7 vs 1 (Fisher one-sided p = .028). The cognitive additions of B28–B43 (a drive that turns lack of progress into pressure, attention to an attack's effects after the swing, hierarchical pooling of sparse evidence, a model of the boss's aim, a motion-phase clock for delays) together double what a learner can do against a boss it has never seen, using only what it perceives. The individual steps are mostly too small to resolve at n = 36 (aim model's survival gain is the exception, B33). The gap to a learner given each boss's physics (~90% wins, B29) remains.

## B45 · Out-of-sample replication of B44 on 18 new bosses (7018–7035)
Configuration frozen after B44, bosses never used during development:

| learner | damage | wins | frames | timeouts |
|---|---|---|---|---|
| naive | 27.2% | 2/36 | 5330 | 7 |
| full stack | **60.4%** | **8/36** | 2445 | 0 |

Damage ×2.2 again (Wilcoxon one-sided p = 4.5e-6), wins 8 vs 2 (Fisher one-sided p = .042). **The B44 gain replicates on held-out bosses.** Pooled over 36 unseen bosses (72 paired fights): naive 25.0%, 3/72 wins; full stack 55.0%, 15/72 wins.

## B46 · The general-learner stack on Margit: worse than plain v6 (running ablation)
- Margit (uncalibrated program), 12 fights in series, fresh watch checkpoint per config: plain v6 (current frame + seen-tip geometry, re-run) 41.5% damage, 0/12 wins; full stack (drive, after-swing attention, back-off, aim, phase clock) **28.9%**, 0/12.
- The stack that doubles damage on procedural bosses *hurts* on Margit. Margit differs from the procedural bosses in ways that could interact with each component: aggressive delayed hammer slams and dagger projectiles (the drive pushes in), long strings with branches, and a delay mechanism implemented in the program runner rather than as the boss's windup hold (phase clock). Ablation running: full minus phase clock, full minus drive, aim-only (no drive, no phase clock), and v6 + drive.
- **Ablation (12 fights each, same seeds):**

| config | damage | frames |
|---|---|---|
| plain v6 | 41.5% | 4371 |
| v6 + drive | 46.7% | 2532 |
| aim + attention + back-off, no drive, no phase clock | 47.4% | 4360 |
| **aim + attention + back-off + drive, no phase clock** | **49.6%** | 3393 |
| full stack (with phase clock) | 28.9% | 3029 |
| full stack minus drive (with phase clock) | 21.5% | 3417 |

  **The phase clock is what hurts on Margit**: both arms with it are 20–28 points below every arm without it. Without it, the stack is the best simulator-free Margit result so far (49.6% vs 41.5% for v6 in the same run, and 42.9% in B25). Margit's delays are bimodal mix-ups (the same move sometimes comes with no delay at all); one hypothesis is that sampling remaining holds from that mixture makes the agent mistime against the undelayed version, which the procedural bosses (unimodal log-normal holds) don't punish. Open.
- **Recommendation.** The general-learner default drops the phase clock: on procedural bosses it was a non-significant +4 points (B43), and on Margit it costs ~20. The B44/B45 conclusions hold without it: the aim-model stack without the phase clock already scored 45.4% vs 22.7% for the naive learner on the B44 bosses.
- **Replication on two more 12-fight series (reps 1–2, new seeds): reverses the ablation's headline.** Plain v6 57.8% damage vs stack-without-phase-clock 27.8% (24 fights each). Pooled over all 36 fights: v6 52.3% vs stack 35.1% (Wilcoxon, stack > v6: p = .997, i.e. the stack is reliably *worse* on Margit). The single-series ablation above (12 fights, one learning trajectory) was too noisy to rank configs; the 49.6% was a lucky series. **Corrected conclusion: on Margit, the general-learner stack hurts the simulator-free learner**, even without the phase clock. Recommendation withdrawn; ablation over the new series running to find which component is responsible (v6 + drive; aim + attention + back-off without drive).
- **Ablation over 36 fights (3 series), Margit:** plain v6 52.3% · v6 + drive 49.4% · aim + attention + back-off (no drive) 40.0% · all of them (no phase clock) 35.1%. The drive is roughly neutral on Margit; the aim/attention/back-off bundle costs ~12 points and the drive adds to the loss in combination. Splitting the bundle next (aim alone; attention + back-off alone).
- **Bundle split (36 fights each, Margit):** aim model alone 34.3% · after-swing attention + back-off alone 33.6% · together 40.0% · plain v6 52.3%. Each component on its own costs Margit ~18 points.
- **Conclusion for B46.** The components found on procedural bosses (B30–B33) don't transfer to Margit; there plain v6 (current-frame maps + seen-tip geometry) stays best for the simulator-free learner. Likely reasons, untested: Margit's moves were authored with long tracking-free windups and bimodal delay mix-ups, so the aim model's learned turn rate and commit times (pooled over a mixed moveset) misplace danger. After-swing attention adds the long tails of Margit's strings, whose follow-ups start inside the 90-frame window, to the previous move's map, smearing it. The procedural suite generalisation (B44–B45) stands, but **the stack is not a universal improvement**: tuning cognitive biases on one family of bosses can hurt on a designed boss. A real general learner would need to *select* these biases per boss (e.g. by comparing each model's predictive likelihood of the contacts it feels: Bayesian model selection over its own world-model variants).

## B47 · Metacognition: the agent picks its own world model per boss
- B46 showed that no single world model is best everywhere. So the agent keeps two candidates learning side by side from the same perception: plain v6 (current-frame maps + seen-tip geometry) and the aim + after-swing-attention + back-off model. Each scores how well it predicted every felt/unfelt frame of an identified swing *before* learning from it (prequential log-likelihood, per frame). Before each fight the planner switches to the variant with the higher score. Same perceptual evidence, no rewards involved: model selection by predictive accuracy.
- **Margit (3 series × 12 fights):** 43.5% damage (v6 alone 52.3%, stack alone 40.0%). The choice starts on the stack (its score is marginally higher after watching) and **switches to v6 in all three series**, after 5, 8 and 8 fights (`11111 0000000`, `11111111 0000`, `11111111 0000`). Fights fought with v6 selected: 55.7% damage; with the stack: 34.7%. The selector finds the right model, but slowly: the likelihood gap is small (−0.0476 vs −0.0489 nats/frame at the end) because most frames are safe, so a model's fit is dominated by the easy "no contact" frames.
- Running: the same selector on the 18 procedural bosses, where the stack was the better model (B44), plus the v6 + drive arm on bosses 6–17 for the paired comparison.
- **Procedural bosses (18 × 2, drive on in all arms):** selector 39.5% damage, 3/36 wins; it chose v6 for 22 of 36 fights. v6 + drive alone 37.7%, 2/36; stack + drive 45.4%, 8/36. So on the procedural bosses the selector *also* prefers v6, although the stack controls better there (difference n.s.: Wilcoxon p = .16). **The better predictor of contacts is not the better controller.** Frame-level contact likelihood rewards calibrated "probably safe" predictions on the many easy frames; control value depends on getting the few dangerous frames right in the places the planner considers going. This is the objective-mismatch problem of model-based RL showing up in a metacognitive selector. A selector scored on *decision-relevant* frames (the agent's near-boss positions at dangerous phases), or on achieved returns (a bandit over models), is the obvious next version.
- **Decomposing B44 with the new v6 + drive arm (same 18 bosses × 2):** naive 22.7% → **+ drive 37.7%** (p = .002) → + aim/attention/back-off 45.4% (vs + drive: p = .16) → + phase clock 49.5% (p = .21). **Most of the general-learner gain on procedural bosses comes from the engagement drive**; the world-model refinements add a further ~12 points that this sample size can't resolve, and on Margit they cost points (B46). Results summary updated to say so.

## B48 · Choosing the world model by results (bandit over models), Margit
- Same two candidate models, both learning from all perception; before each fight a UCB bandit picks one by the mean damage it achieved in the fights where it was used (each tried once first).
- 3 series × 12 fights: **45.9% damage** (likelihood selector 43.5%, v6 alone 52.3%, stack alone 40.0%). Choice sequences: `011110111111`, `010000001000`, `010000000000`. Two series lock onto v6 after one fight each. One series lands on the stack after v6's first fight went badly (22% damage), because single-fight returns are very noisy (SD ≈ 20 points).
- **Reading.** Selecting by returns finds the better model faster than selecting by prediction when it isn't unlucky. Selecting by prediction is slower but never locks onto the wrong model for good. Both recover most of the gap between the worse and better fixed model. A combination (likelihood as the prior, returns as the evidence) is the natural design, and is how a player who "trusts their read" but changes approach after repeated deaths behaves.

## B49 · Combined selector: likelihood as prior, returns as evidence
- Each model gets a Gaussian posterior over its damage per fight (prior mean 0.45, plus 0.08 for the current likelihood favourite, prior weight 2 fights). Pick the higher posterior mean.
- 3 series × 12 fights on Margit: **47.2% damage** (bandit 45.9%, likelihood 43.5%, v6 alone 52.3%, stack alone 40.0%). Choices: `111110000000`, `110000000000`, `100000000000`. It starts on the likelihood favourite (the stack), and the returns move it to v6 after 1, 2 and 5 fights in every series, with no wrong lock-in. The fastest and most robust of the three selectors, ~5 points below always using the best model (the price of finding out which one it is).
- Selector series on Margit, B47–B49: prediction only (slow, safe) → returns only (fast, can lock in wrong) → both (fast and safe). This is the metacognitive layer the general learner needs, since B46 showed that no fixed bundle of cognitive biases is best on every boss.

## B50 · Selector on the procedural bosses, 6 attempts each (324 fights)
18 procedural bosses × 6 attempts, drive on in all arms, learning carried across attempts:

| learner | damage | wins / 108 | attempts 4–6 |
|---|---|---|---|
| plain v6 | 36.0% | 8 | 33.9% |
| aim + attention + back-off (stack) | **46.0%** | **18** | 47.6% |
| combined selector (keeps both) | 39.7% | 10 | 40.9% |

- **With 3× the data, the stack's advantage over v6 on procedural bosses is clear**: +10 points (Wilcoxon p = .0006), more wins, and better on 17 of 18 bosses. This revises B47's "n.s." (12 points at n = 36). So the world-model refinements do help on this family; the drive is still the larger part of the gain from the naive learner.
- The combined selector lands in between (better than v6, p = .03; worse than the stack, p = .01). It picks the stack only about half the time: the likelihood prior favours v6 on these bosses too, and per-fight returns are too noisy to overturn it within 6 attempts. Across both families: on Margit the selector gets within 5 points of the better model (B49); on procedural bosses it gives back 6 of the 10. **Model selection by "how well do I predict the hits" misleads in one family and helps in the other**; a selector needs a decision-relevant score, not frame-level likelihood (B47).

## B51 · A decision-relevant score doesn't change the verdict: the stack is a worse predictor but a better controller
- Selection score restricted to frames within 3 m of the boss (where the planner actually chooses), balanced between contact and no-contact frames (`WM_SCORE=balanced`), likelihood only.
- **Margit:** 44.9% damage; switches to v6 after 4, 2 and 4 fights (`1111 0…`, `11 0…`, `1111 0…`), faster than the frame-level score (5–8).
- **Procedural (18 × 6):** picks the stack in only 17 of 108 fights; 35.6% damage = plain v6 (36.0%) and below the stack (46.0%, p = .0006).
- **Reading.** On *both* boss families, plain v6 predicts the felt contacts better, even in the near, decision-relevant frames and with the class imbalance removed. Yet on procedural bosses the stack *fights* better. So the stack's benefit there doesn't come from more accurate danger estimates. It comes from how its biases shape behaviour: imagining the boss turning toward where I'll be makes lateral repositioning look worse and committed dodges look better, and pooled evidence makes it bolder in places it has seen little. These are useful biases for decisions, not better beliefs. No predictive score can find them; selecting them needs evidence from outcomes (returns), which is slow because single-fight returns are noisy (B48–B49). This is a clean instance of a known tension in model-based RL and in cognition: the best model for prediction and the best model for control can differ.

## B52 · A prior learned from earlier bosses of the family: works within the family, misleads across families
- Selector: each model starts with a prior equal to its mean damage on the 18 *earlier* procedural bosses (B50: v6 0.36, stack 0.46, strength 3 pseudo-fights), updated with this boss's returns; Thompson sampling with a Gaussian posterior (return sd 0.2) so the disfavoured model still gets tried (`WM_SELECT_BY=family`).
- **Held-out procedural bosses 18–35 × 6 (fresh bosses, not used for the prior):**

| learner | damage | wins / 108 | attempts 4–6 |
|---|---|---|---|
| plain v6 | 47.9% | 17 | 48.7% |
| stack | 56.5% | 25 | 57.1% |
| family-prior selector | **57.9%** | **29** | 57.9% |

  Stack > v6 again on new bosses (p = .03, better on 12/18). The family selector picks the stack in 83/108 fights and matches it (vs stack p = .60; vs v6 p = .002). So **the "which biases work here" knowledge transfers to unseen bosses of the same family**, and the selector loses nothing for keeping v6 available.
- **Margit (3 × 12), same procedural prior:** 40.7%, no wins, vs v6 alone 52.3% and the B49 combined selector 47.2%. Selection strings `110011100000`, `111110100010`, `110111000010`: it does reach v6, but only after 5–7 fights, because the prior says the stack is 10 points better and one Margit fight returns are noisy.
- **Reading.** A learned prior over inductive biases is the right object (it turns 6 noisy returns into a quick decision), but it must be conditioned on *how similar this boss is to the family* the prior came from. Margit is hand-authored, faster and more combo-heavy than the procedural bosses; an agent should notice that from the scouting phase and weaken the prior. Next (B53): make the prior strength depend on a family-similarity signal measured during scouting, e.g. how well the procedural-trained library predicts this boss (out-of-family surprise → weak prior, rely on likelihood+returns).

## B53 · Is Margit perceptually "out of family"? No. Interleaving models within fights: half a fix
- **Fingerprint probe** (`scripts/fingerprint.py`): from 1500 scouting frames, 7 perceptual statistics (fraction of time swinging, swing length, swings/s, combo fraction, swings per bout, movement speed, size). Margit's distance from the procedural family (bosses 0–17) is z = 0.66, among the most *typical* bosses (held-out procedural 0.61–1.52). So whatever makes v6 better on Margit is not visible in coarse behaviour statistics, and a "this boss looks different, trust the prior less" gate built on perception would not fire. Also: across procedural bosses the stack-minus-v6 difference has between-boss variance indistinguishable from zero (only noise around +0.10), so empirical Bayes would make the family prior even *stronger*. Margit is a genuine exception the family data gives no warning of.
- **Interleaved selector** (`WM_SELECT_BY=interleave`): every 300 frames, re-pick the world model by Thompson sampling on its exchange ratio (hits dealt / hits taken while it was in control, Gamma posteriors, family prior as pseudo-counts). Both models get evidence in every fight.
  - **Margit 3 × 12:** 43.0% (family prior 40.7%, combined selector 47.2%, v6 alone 52.3%). Final exchange ratios favour v6 in 2 of 3 series by a wide margin (0.48 vs 0.25, 0.45 vs 0.16); the third series settled on the stack (0.44 vs 0.37). The signal is strong once collected; the cost is the time spent mixing.
  - **Held-out procedural 18 × 6:** 54.5%, 20 wins: between v6 (47.9%, p = .06) and the stack (56.5%, n.s.); the stack held only 55% of frames.
- **Reading.** Exchange ratio is a much sharper outcome signal than end-of-fight damage (2–3× gaps on Margit vs 12-point damage gaps), so segment-level evidence is the right currency. But Thompson sampling keeps mixing too long on a 6–12-fight budget, and on bosses where the stack is right the mixing costs ~2–3 points. Across B47–B53 no selector beats simply committing to the family's better model on procedural bosses, and none recovers more than half of v6's advantage on Margit. **Practical rule for the final agent: use the family default (stack) and switch on a strong exchange-ratio signal** (B54 candidate: switch only when posterior P(other better) > 0.9, otherwise exploit). The line of work has given a clear scientific result (prediction ≠ control; outcome evidence needed) and is now at diminishing returns; next priority moves to sleep/replay (R6).

## B54 · Mostly-greedy segment trials: the first selector that matches the better model on both families
- Same as B53 (re-pick the world model every 300 frames, scored by hits dealt / hits taken while in control, family prior as pseudo-counts) but greedy on the posterior mean, with Thompson exploration in only 20% of segments (`WM_IL_EPS=0.2`).

| selector | Margit (3 × 12) | held-out procedural (18 × 6) |
|---|---|---|
| plain v6 | **52.3%** (B46) | 47.9% |
| stack | 34–40% (B46) | **56.5%** |
| likelihood (B47) | 43.5% | – |
| combined likelihood + return (B49/B50) | 47.2% | 39.7% (bosses 0–17) |
| family prior on returns (B52) | 40.7% | 57.9% |
| interleave, Thompson (B53) | 43.0% | 54.5% |
| **interleave, ε = 0.2 (B54)** | **52.3%** | **56.3%** (vs v6 p = .02; vs stack p = .95) |

- Margit: v6 holds 77–91% of frames by the end in two series (exchange ratio 0.54 vs 0.32, 0.57 vs 0.35); in the third the two models score the same (0.58 vs 0.57), so it doesn't matter which one plays. Procedural: the stack holds 70% of frames.
- **Reading.** The selection problem from B47–B53 is solved by two choices together: (1) a dense outcome signal (exchange ratio per 5-second segment instead of final damage per fight, ~10× more evidence per fight), and (2) exploit what that evidence says rather than probability-matching on it. This is close to how a player tests a tactic: try something for a few exchanges, keep it if the trades go your way. It is now the default selector for the full agent (`WM_SELECT=1 WM_SELECT_BY=interleave WM_IL_EPS=0.2`). B55 (Godrick-like boss) tests it on a third family.

## B55 · Godrick-like boss: v6 beats the stack again, and the B54 selector fails
- New hand-authored boss (`BOSS=godrick`, `GODRICK_MOVES`/`GODRICK_STRINGS` in `arena3d/program.py`): heavy axe, 3-hit sweep strings branching into an overhead or a stomp, axe charge, jump slam, delayed whirlwind with a 5.5 m ring; phase 2 adds a dragon-breath projectile, dragon swipe and dragon slam. Timings from general knowledge of the fight, not frame-annotated footage. Sanity: aggro dies in ~12 s (60% boss HP left), state oracle wins 1 of 3.
- 3 series × 12 fights, same scouting checkpoint for all arms:

| learner | damage | fights 7–12 |
|---|---|---|
| plain v6 | **30.1%** | **40.2%** |
| stack | 15.6% (vs v6 p = .003) | 13.8% |
| combined likelihood + return | 24.9% | 27.6% |
| interleave ε = 0.2 (B54 default) | 16.2% (vs v6 p = .005) | 19.6% |

- **Two hand-authored bosses, same verdict: v6 is the better controller, and by a lot on Godrick (2×).** The stack's advantage is specific to the procedural family. Plain v6 also keeps learning on Godrick (30% → 40% in later fights); the stack doesn't.
- **The B54 selector picked the stack in 2 of 3 series**: segment exchange ratios favoured it (0.14 vs 0.09, 0.21 vs 0.15) even though v6 alone deals twice the damage. Two likely causes: (1) Godrick deals and takes few hits, so the family prior (which favours the stack) dominates the counts for longer; (2) credit leaks across segments: a 3-hit string or a delayed whirlwind that lands in one segment was set up (or failed to be dodged) under the previous model. Under interleaving, the score measures "who is in control when the trade resolves", not "whose decisions caused it".
- Next (B56): credit each hit to the model that was in control when the decision was made (hit taken → model at the boss move's onset; hit dealt → model when the swing started), and weaken the family prior.

## B56 · Decision-time credit + a weak prior fixes Godrick
Godrick 3 × 12, interleave ε = 0.2, varying credit assignment and family-prior strength:

| variant | damage | final exchange ratios (v6, stack) | v6 share of frames |
|---|---|---|---|
| v6 alone / stack alone | 30.1% / 15.6% | – | – |
| B54 (credit at resolution, prior 3) | 16.2% | 0.09/0.14, 0.15/0.21, 0.35/0.18 | 1 of 3 series |
| decision credit, prior 3 | 25.2% | 0.33/0.37, 0.19/0.28, 0.39/0.27 | mixed |
| **decision credit, prior 1** | **26.6%** (vs v6 p = .51) | 0.40/0.09, 0.38/0.20, 0.36/0.27 | 85–95% in all 3 |
| resolution credit, prior 1, 15 s segments | 24.2% | 0.47/0.07, 0.39/0.22, 0.13/0.07 | 2 of 3 |

- Crediting a hit taken to the model in control when the boss move *started* (when the dodge decision was made), and a hit dealt to the model in control when my swing started, reverses the misleading ratios: with a weak prior all three series settle on v6 and the damage gap to v6-alone closes (26.6% vs 30.1%, n.s.). Credit assignment to the decision, not the outcome moment, is what makes short within-fight trials valid when moves are long and delayed.
- B57 (running): the same selector (`WM_IL_CREDIT=decision WM_FAMILY_N=1`) on Margit and held-out procedural bosses, to check it keeps B54's wins there.

## B57 · No selector setting wins on all three families; plain v6 is the most robust single choice
Same selector as B56's best (decision credit, prior 1, ε = 0.2) on the other two families:

| learner | Margit | held-out procedural | Godrick | mean |
|---|---|---|---|---|
| plain v6 | 52.3% | 47.9% | 30.1% | **43.4%** |
| stack | 34–40% | 56.5% | 15.6% | ~36% |
| B54 selector (resolution credit, prior 3) | 52.3% | 56.3% | 16.2% | 41.6% |
| B56 selector (decision credit, prior 1) | 43.2% | 51.0% | 26.6% | 40.3% |

- Decision credit fixes Godrick but costs ~9 points on Margit (one series locked onto the stack: ratios 0.24 vs 0.37) and ~5 on procedural (stack share 63%, vs 70% in B54; vs B54 p = .16). With 6–12 fights per boss, any selector's settings trade one family against another; a 36-fight Margit run has a series-level SE of several points, so the B54-vs-B57 Margit gap is itself only ~2 SE.
- **Conclusion of the selection line (B47–B57).** (1) Prediction quality does not identify the better controller (B47, B51). (2) Outcome evidence does, but per-boss evidence within a dozen fights is too noisy to beat a robust default on every family. (3) The stack's advantage is family-specific (procedural only); on both hand-authored bosses it hurts, on Godrick by 2×. So the robust agent uses plain v6 + drive and treats the stack's biases as a family-specific option. The scientific question that remains is *why* those biases hurt on hand-authored bosses: B58 ablates the stack's components on Godrick.

## B58 · On Godrick the attention window (persist) is what hurts
One stack component at a time, Godrick 3 × 12 (each with its own scouting checkpoint):

| model | damage | fights 7–12 | vs v6 |
|---|---|---|---|
| plain v6 (current frame) | 30.1% | 40.2% | – |
| v6 + back-off (`WM_HIER=4`) | 29.2% | 28.5% | p = .70 |
| onset frame | 29.0% | 32.7% | p = .73 |
| onset frame + aim model | 24.7% | 31.6% | p = .21 |
| **v6 + attention window (`WM_PERSIST=1`)** | **20.9%** | 19.6% | **p = .05** |
| full stack | 15.6% | 13.8% | p = .003 |

- The 90-frame post-swing attention window is the main cost on Godrick (−9 points); the aim model costs a further ~5 (n.s.); back-off and the onset frame are neutral. With persist, scouting records 37% more exposures and 25% more contacts: on a boss whose rings, delayed whirlwind and 3-hit strings land well after a move's own swing, contacts from the *next* move or the shockwave get filed under the previous one. That's the same superstition mechanism hypothesised in B31, now localised to one component.
- B59: the stack without persist (onset + aim + back-off) on all three families. If it keeps the procedural gain, it's a candidate for a model that's good everywhere.

## B59 · The attention window carries the stack's procedural gain *and* its Godrick cost
Stack without persist ("lite": onset frame + aim + back-off), all three families:

| model | Margit | held-out procedural | Godrick |
|---|---|---|---|
| plain v6 | 52.3% | 47.9% | 30.1% |
| full stack | 34–40% | 56.5% | 15.6% |
| stack without persist | 53.1% | 44.4% (vs stack p = .002) | 20.9% (vs v6 p = .03) |

- Removing the attention window takes the stack back to v6 level on Margit, but it also removes the entire procedural gain (56.5 → 44.4). So the 90-frame post-swing window is the double-edged component; aim + back-off add nothing on procedural and cost on Godrick.
- **Diagnostic** (`scripts/godrick_diag.py`, one 6-fight series each): on Godrick about a third of all hits taken are shockwaves that arrive after the boss has gone back to idle (21–22 of 61–67 hits, for v6 and v6+persist alike), and shockwaves overall are the largest single source. The world models learn rings only as statistics over (time since onset, distance, bearing); a ring is really a visible expanding front.
- B60 (running): a perceptual "ring reflex": perceive a live shockwave's centre and radius (noisy), learn its expansion speed from consecutive frames, and roll so the i-frames cover the front's arrival (`WM_RING_REFLEX=1`). Intuitive physics for a visible effect instead of a learned hazard map.

## B60 · Seeing shockwaves: small gain on Godrick, not significant
Shockwaves are visible expanding fronts, so the agent can perceive each live ring's centre and radius (noisy) and learn its expansion speed and the largest radius reached (`_ring_percept`). Two ways to use it, plain v6, 3 × 12 fights:

| | Godrick | Margit |
|---|---|---|
| v6 | 30.1% | 52.3% |
| + reflex (roll so i-frames cover the front's arrival) | 30.0% | 47.4% (n.s.) |
| + fronts in the planner's imagination (`WM_RING_PLAN=1`) | 33.1% (1 win, the first on Godrick) | 52.3% |
| + both | 34.6% (p = .51) | – |

- The reflex alone does nothing: in a diagnostic series 13 of 22 shockwave hits landed while the player was already committed (mid-attack or mid-roll), and 25 of 47 triggers had too little stamina to roll. Putting the predicted fronts into imagined futures (so the planner won't start a swing that a ring will cross) helps a little on Godrick; Margit has rings only in phase 2 and doesn't change.
- **Reading.** Reacting to a visible ring is too late; the cost is set by commitments made before the ring exists, i.e. by *anticipating* the ring from the move (the hazard maps' job). Perceiving live fronts is a small, safe addition; kept as an option.
- B61: longer series on Godrick (24 fights) to see where v6 + ring perception levels off.

## B61 · Godrick learning levels off at ~30–35% within 7 fights
4 series × 24 fights (96 per arm), plain v6 + drive, same scouting:

| | fights 1–6 | 7–12 | 13–18 | 19–24 | all | wins |
|---|---|---|---|---|---|---|
| v6 | 24.7% | 39.8% | 28.0% | 27.5% | 30.0% | 0 |
| v6 + seen shockwave fronts | 26.7% | 35.1% | 32.0% | 37.2% | 32.7% (p = .29) | 0 |

- Learning happens in the first ~7 fights and then stops; no wins in 192 fights, while the state oracle (true boss state, 14-frame lookahead) gets to 7% boss HP left and wins 1 of 3 (B55 sanity check). Ring perception adds ~3 points, still not significant with 96 fights.
- So on Godrick the world-model learner hits a ceiling that more fights don't lift. B62 measures where the ceiling comes from: the same MPC planner with Godrick's true per-move simulator (`mpc_grammar`) and with a learned string model on top of the simulator (`mpc_scratch`). If those win, the ceiling is in the learned danger maps; if not, it's in the planner.

## Audit (Nov 27) and B63 · Shockwaves as remembered objects: no gain
- **Audit.** `scripts/hit_filmstrip.py` renders the lock-on view for the 1.2 s before each hit (`results/filmstrips/godrick_hits.png`). What it shows on Godrick: (1) the agent rolls *away* from shockwaves, but a ring expands at ~0.4 m/frame vs a roll's 0.2, so it is caught when the i-frames end; (2) during the jump slam's wind-up the engagement drive walks it in; (3) it trades light attacks into a 3-hit axe string and takes a head hit for 29. Structural finding: the plain world model (v6) records exposures only while the boss is in a strike, and rings land after the strike ends, so **v6 cannot learn ring danger at all**; the action set also has no forward or diagonal-forward roll, the standard souls answer to a ring or a sweep.
- **B63 object memory** (`WM_RINGOBJ=1`): when a ring appears, remember (move, time since onset, centre in the onset frame), follow it to learn expansion speed and how far it gets (learned radii match the true 2.5/4.5/5.5 m to within 0.5 m), and imagine remembered rings of the hypothesised move during planning. Godrick 3 × 12: 29.4% (1 win) vs v6 30.1%; with live-front imagination too, 26.5%. No gain.
- So knowing where and when rings will be is not what limits Godrick. B62 (true-simulator ceiling) and B64 (forward rolls) next.

## B62 · Where the Godrick ceiling comes from: the learned physics, not the planner or the strings
Same MPC planner (9 macro-actions, 3 imagined futures, H = 24, move identity read at 12 frames), Godrick, 6 fights each:

| imagined futures from | wins | boss HP left | fight length |
|---|---|---|---|
| true per-move simulator + true strings | 4/6 | 10% | 122 s |
| true simulator + true strings + engagement drive | **6/6** | 0% | **26 s** |
| true simulator + strings learned from scratch in the fight | 1/6 | 24% | 153 s |
| learned danger maps + learned strings (v6, B61) | 0/192 | ~70% | – |

- With an accurate model of *how each move moves and hits*, the same planner beats Godrick every time, fast. With the learned danger maps it never wins. **The ceiling is the learned physics model.** Strings matter too (true → learned costs 5 of 6 wins with a true simulator, B62 row 3), but that gap is smaller than the physics gap.
- This changes the research target: not better selection among statistical hazard maps, but a learned model of the boss's motion that's good enough to simulate. → B65.

## B65 (running) · Mirror simulation: re-enact what you've seen
`arena3d/mirror.py`. The agent keeps an episodic memory of every boss move it has watched: the boss's perceived full-body pose over the move (pose estimate with noise: 3 cm root, 0.03 rad per joint), in the move's onset frame (origin = boss at onset, x-axis = boss→player line), plus when shockwaves or projectiles appeared and how far rings got. To imagine a future, it re-enacts a remembered episode of the hypothesised move (identity from 70%-accurate reading or learned strings; episode picked by similar onset distance) on its own copy of the arena, aligned to the current boss position, facing and time since onset, and checks real 3D contact against its own imagined body. No boss program, mocap library or boss simulator. This is motor-simulation / mirror-system theory as a world model: predict another body by running its remembered movements through your own body-and-physics model. Smoke test (Godrick, 1 fight each, short scouting): 24% and 48% damage on first fights (v6 first fights ≈ 25%).

## B65 · Mirror simulation, debugged: first real wins on Godrick
**First run (as described above):** Godrick 24.2%, Margit 4.9% (dies in ~15 s). The audit tool for this was a fidelity test (`scripts/mirror_fidelity.py`): from mid-move frames, imagine 15 frames ahead and compare the imagined blade tip and "am I hit?" with what really happens.
- First version: tip error 3.2 m, worse than assuming the blade stays put (0.97 m); the true simulator with hidden hold/identity gets 0.32 m. Causes found and fixed, in order:
  1. Replayed moves stopped at the end of the episode, so follow-ups in a string were never imagined → **chain** into the next move (the remembered episode says whether a move followed; which one from the learned strings).
  2. Instances of the same move differ by ~45° in facing and ~1 m in position from the onset frame, because the real boss turns to follow the player and lunges according to distance → **anchor to now**: find where in the remembered move the boss is (closest joint pose near the expected time) and place that frame exactly on the boss's currently perceived root position and facing (facing from the quaternion difference). Tip error 3.2 → 1.3 m.
  3. 30% of identity labels are wrong → **label-free recognition**: score every remembered episode of every move by how well its frames match the boss's last 4 perceived poses (and the current player distance, since lunges depend on it); sample replays by that match (sharp temperature). Tip error → 0.9 m.
  4. Counterfactual aim: the replay keeps turning to follow the *imagined* player by the difference between where it is and where the remembered player was, at a learned turn rate, until the remembered swing commits (fastest seen blade motion).
- Hit prediction for a fixed action, Margit (AUC / Brier; true simulator 1.0 / 0.0): stand 0.92 / 0.10, attack 0.94 / 0.09, roll back 0.98 / 0.05, roll sideways 0.98 / 0.04. Imagined rolls still raise some false alarms.

**Fights (3 series × 12, noisy pose memory, scouting as for the world-model learner):**

| learner | Godrick | Margit |
|---|---|---|
| learned danger maps (v6) | 30.1%, 0 wins (0/192 over B55–B61) | **52.3%** |
| mirror simulation, first version | 24.2% | 4.9% |
| **mirror simulation, fixed** | **47.2%, 3 wins** (vs v6 p = .018); fights 1–4 / 5–8 / 9–12: 38 / 44 / **60%** | 34.8% (vs v6 p = .003) |

- **On Godrick this is the first learner without a boss simulator to win, and it keeps improving with fights** (60% damage in fights 9–12, where v6 plateaued at ~30% for 24 fights). It learns the boss by remembering whole movements and re-enacting them through its own body-and-contact model.
- On Margit it is still well behind v6: Margit's staff strings are fast and chain tightly, so recognition errors and chaining errors cost more. Next: more imagined samples, recognise using the blade path as well as the joints, and mix the replay's predictions with the learned danger maps where memory is thin.

## B66 · Replays that blend into new movements: Margit wins too
Hits on Margit came at 1–3 frames into a new move: the engine blends the boss's body from its last pose into the next movement over 8 frames, and the fast blade sweep in that transition is real danger. Chained and opener replays now (a) start a new move facing the (imagined) player, as at every remembered onset, and (b) blend from the current imagined pose into it the same way bodies do. Fidelity: tip error 0.9 → 0.77 m, standing-hit AUC 0.92 → 0.93. Episodes now also record whether a move followed (cut by the next one) so chaining follows the memory.

| learner (3 × 12) | Godrick | Margit |
|---|---|---|
| learned danger maps (v6) | 30.1%, 0 wins | 52.3%, 0 wins |
| mirror simulation (B65) | 47.2%, 3 wins | 34.8%, 0 wins |
| **mirror simulation + blending (B66)** | **60.8%, 2 wins** (vs v6 p < .0001) | 43.5%, **2 wins** (vs v6 p = .07) |

- Godrick doubles relative to the world-model learner. On Margit it now wins fights (2 of 36; v6 has none) though mean damage is still a bit below v6. B67: procedural bosses; B68: more imagined futures per decision on Margit.
