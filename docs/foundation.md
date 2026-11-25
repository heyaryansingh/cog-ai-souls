# Foundation v2: an embodied Bayesian agent that learns bosses the way people do

*Nov 25, 2026. Aryan's feedback after playing MiniSouls: "nothing like real life… or a real souls game. Rethink the training, modeling, foundation, world modeling, state… incorporate research from cognitive AI, robotics, world models, with Bayesian thought." This replaces the agent design in `revamp-plan.md` §3; the Arena environment plan in that file stands.*

## 0. The one-paragraph version

A person fighting a new boss is not doing reinforcement learning over pixels. They:
- **see a body move**, and parse its motion into actions;
- **treat the boss as an agent** with a repertoire and habits;
- run a rough **mental simulation** of their own body and the boss's swings;
- **act with practised reflexes** while deliberating only when it matters;
- **after each death, replay what happened** and revise their theory of the boss;
- carry **schemas across bosses** ("big wind-up, delayed swing, roll late").

The agent copies that structure. It has a hierarchical generative world model (perception → events → agents → physics), a factored belief state with exact or Rao-Blackwellized inference, model-based control with learned habits and uncertainty-based arbitration, and learning at four timescales: tick, fight, death, and boss family. Every component maps to a result in cognitive science, robotics or world-model research, and each one can be removed (lesioned) to test whether it matters.

## 1. Environment realism (Arena, built in v0)

| Real fights | Arena |
|---|---|
| Boss seen as a moving body, no labels | 17-joint 3D skeleton, noisy keypoints with dropout, no move labels |
| 2D/3D space, spacing, angles, rolling *through* swings | Continuous 2D arena, lock-on frame, directional rolls, strafing bosses |
| Delays, combos, tracking, mix-ups | Log-normal holds, branching combo strings, per-move tracking, shared wind-ups |
| Hitbox geometry and projectiles | Arcs, lines, discs, rings with safe zones, straight and homing projectiles |
| Body with commitment and resources | Animation-locked actions, i-frames, stamina, poise, flasks, parry window, jump over low attacks |
| Different builds | Light, medium and heavy bodies with different roll distance, i-frames and poise |
| Perception delay | Boss-related inputs delayed by 6 ticks (200 ms); proprioception current |

**Still to add for realism:**
- a **camera field of view** (you cannot see behind you);
- **audio events** (a whoosh, a charge sound) as a second sense;
- **momentum** (acceleration and turn limits on the player);
- **boss aggression adaptation** (bosses that punish passivity);
- *stretch:* a **MuJoCo physically simulated body** for the player (robotics-grade embodiment).

## 2. The world model, layer by layer

### 2.1 Low level: state estimation (robotics)
Noisy, delayed keypoints go into a **Kalman filter / smoother** over boss kinematics (root position, velocity, heading, joint velocities), with missing-data handling for dropouts. This is standard robot state estimation (Thrun, Burgard & Fox, *Probabilistic Robotics*). Projectiles get multi-target tracking.

### 2.2 Mid level: event segmentation (cognitive science, neuro-ethology)
People segment observed action at points where prediction fails (event segmentation theory, Zacks et al. 2007). The agent learns the boss's **motion syllables** without labels:
- an autoregressive HMM / sticky HDP-HSMM over the filtered kinematics (Fox et al. 2011; Johnson & Willsky 2013);
- this is the approach of **MoSeq**, which discovers the "syllables" of mouse behaviour from depth video (Wiltschko et al. 2015).

Syllables are the agent's own vocabulary for the boss's moves. A **self-supervised pose encoder** trained to predict future kinematics in latent space (JEPA-style: LeCun 2022; V-JEPA, Bardes et al. 2024) gives fast amortized recognition; the HSMM keeps it calibrated.

### 2.3 High level: the boss is an agent (theory of mind)
The boss's choice of next syllable depends on context: distance, angle, the player's last action, phase. The agent infers this as a **policy of another agent**, which is Bayesian inverse planning applied to an opponent (Baker, Saxe & Tenenbaum 2009). The representation is the slow mind's probabilistic program, extended to combo graphs:
- semi-Markov over syllables;
- branching continuations;
- context laws ("punishes heals");
- phase changes.

Structure (which moves exist, which chain) is learned by resample-move SMC with birth, merge and split moves, as in Entries 6–20.

### 2.4 Physics of harm: a game engine in the head (intuitive physics)
For each syllable the agent learns:
- a **kinematic template** (pose trajectory, travel, tracking rate);
- a **spatial hit map** in boss-local coordinates: a Gaussian-process or kernel-density estimate of where the hit lands, updated from every hit taken and every near miss.

This is the "intuitive physics engine" view (Battaglia, Hamrick & Tenenbaum 2013; Ullman et al. 2017, *Mind games*). People simulate where a swing will reach; they do not memorize state-action values.

### 2.5 Self-model: an internal forward model of one's own body (motor control)
Roll displacement, i-frame window, stamina costs and attack timings are learned from proprioception and outcomes. These are **internal forward models** with efference copy (Wolpert, Ghahramani & Jordan 1995), and they make adapting to a new build fast (E8/E10 showed this in 1D).

### 2.6 What the structure misses: a learned residual model (world-model RL)
A small neural **recurrent state-space model** (Dreamer's RSSM, Hafner et al. 2023) is trained on prediction errors of the structured model. Their predictions are combined by Bayesian model averaging. The structured model is sample-efficient; the residual catches what it cannot express. This is the "theory plus intuition" pattern of **EMPA**, which learns Atari games about as fast as humans by Bayesian theory learning plus planning (Tsividis et al. 2021).

## 3. The belief state

The belief factorizes the way the world does, and each factor gets the cheapest exact-enough inference. This is Rao–Blackwellization (Doucet et al. 2000):

| Factor | Inference |
|---|---|
| Boss program (moves, combos, laws, hit maps) | SMC particles, refreshed between and within fights |
| Current syllable and phase-in-syllable | Exact HSMM forward filter per particle (as in 1D) |
| Boss kinematics | Kalman filter conditioned on the syllable |
| Projectiles | Track filters |
| Own body | Observed (proprioception) + self-model posterior |

From the belief the agent computes a **threat field**: P(a hit lands at offset (Δx, Δy) within t ticks). It is the 2D version of the 1D threat clock, like risk maps and occupancy forecasts in robot motion planning.

## 4. Acting: deliberation, habit, and the arbiter

- **Deliberation:** MPPI (Williams et al. 2017) over short action sequences in the world model, from belief samples. The objective is **risk-sensitive** (CVaR over imagined outcomes): players fear death more than they value damage.
- **Habit:** a fast policy over belief features, distilled from the planner by DAgger / guided policy search (Ross et al. 2011; Levine & Koltun 2013). In 1D it matched a 4× more expensive planner and was robust when the model was wrong (E28–30).
- **Arbitration:** uncertainty-based control between the two (Daw, Niv & Dayan 2005; Lee, Shimojo & O'Doherty 2014), and a learned value-of-computation gate (Hamrick et al. 2017). Habit leads while the world model is unreliable; planning takes over as it becomes trustworthy, which is the crossover found in E29–31.
- **Information:** scouting is valued by its *decision value*, not raw information gain. E15 showed that IG-driven rolls got the agent killed.

## 5. Learning at four timescales (complementary learning systems)

| Timescale | What changes | Mechanism | Brain analogy |
|---|---|---|---|
| Tick (33 ms) | Belief over state | Filtering | Perception |
| Within a fight | Boss program | Online SMC refresh (E22: doubles first-encounter wins) | Working memory |
| Between attempts | Program structure, hit maps, self-model | **Offline replay** of the last fight (SMC rejuvenation, Dyna-style imagined fights) | Hippocampal replay during rest (McClelland, McNaughton & O'Reilly 1995; Kumaran, Hassabis & McClelland 2016) |
| Across bosses | Archetype priors, syllable library, encoder, habit | Empirical Bayes; amortized inference (intuition net) | Schema learning (neocortex) |

## 6. Training

1. **Simulated developmental curriculum** (from the Neoteny idea): thousands of procedurally generated bosses across archetypes. They train the slow-learning parts: pose encoder, syllable library prior, archetype hyperpriors, intuition net and habit. Nothing about a *test* boss is trained in; test bosses are learned online by inference only.
2. **Domain randomization** (robotics sim-to-real): keypoint noise, dropout, latency, body build and boss speed all vary in training, so the agent cannot overfit simulator specifics.
3. **Held-out generalization:** new bosses; new archetype combinations; a held-out archetype; held-out moves; a new body build.

## 7. Evaluation

- **Learning curves:** wins and deaths-to-first-win per attempt, with boss-clustered statistics and pre-registered replications (as with P1).
- **Baselines:**
  - recurrent PPO on keypoints and on pixels;
  - RL² across bosses;
  - DreamerV3-small;
  - the agent with each component lesioned.
- **People:** a playable Arena port records every input. Fit each model's per-decision action likelihood to human play (two-step-task style). Test whether people's fitted habit weight falls across attempts, as the agent's does. Pilot data: Aryan, Boss 1 of MiniSouls, first win on attempt 8 (agent: attempt 8).
- **Real-game bridge (stretch):** run pose estimation on real Elden Ring boss footage and test whether syllable discovery recovers the documented moveset.

## 8. Order of work

1. Arena v0 (done): env, archetypes, body, scripted references. Next: field of view, audio events, momentum, a cheat-planner ceiling.
2. Recurrent-PPO baseline on keypoints, started early because it runs for hours.
3. Perception stack: Kalman → syllable discovery, scored against true move segments (NMI, boundary F1).
4. Agent world model on syllables (program + hit maps + self-model) → threat field → MPPI planner.
5. Habit distillation, arbitration, replay between attempts, archetype priors and intuition net.
6. Playable Arena port with logging; human-fit analysis; real-footage bridge.
