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
