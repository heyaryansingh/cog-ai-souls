# Enhanced Rolling Horizon Evolution Algorithm with Opponent Model Learning (FightingICE)

## 1. Citation and reading status
- Zhentao Tang, Yuanheng Zhu, Dongbin Zhao, Simon M. Lucas. "Enhanced Rolling Horizon Evolution Algorithm with Opponent Model Learning: Results for the Fighting Game AI Competition." arXiv 2003.13949 (2020); later in *IEEE Transactions on Games*. https://arxiv.org/abs/2003.13949
- **Reading status:** full PDF processed through a web-fetch extraction tool (direct PDF download blocked in this sandbox). I got the algorithm, features, training regimes and all result tables (II–V). Some hyperparameters (discount, N-step length) are not given in the paper itself per the extraction. Author list from memory of the arXiv page; verify.

## 2. Core idea and exact method
**Platform.** FightingICE (Ritsumeikan): 2-player 2D fighting game, 60 fps, **16.67 ms per frame** decision budget, **15-frame observation delay**, built-in forward model (simulator) the agent can call. Characters ZEN, GARNET, LUD (LUD's action data hidden before competition). 400 HP, 60-s rounds; higher HP wins on timeout. 56 actions.

**RHEA (planner).** Population n = 7 action sequences of length l = 4 over 56 actions; random init; elitism k = 1; uniform crossover elite × non-elite; mutation replaces one gene (applied with p_m = 0.85); fitness = λ·(normalized HP difference at end of simulated sequence, ±1 for simulated KO) + (1−λ)·diversity bonus, λ = 0.5; evolve until the 16.67 ms budget is used; execute first action. Shift buffer tried—little effect (short horizon).

**Opponent model.** Input 18 features: both players' HP, energy, x, y, one-hot state {Stand, Crouch, Air, Down}, relative dx, dy, normalized to [0,1]. Output: softmax over 56 actions. **Architecture: a single linear layer + softmax** (no hidden layer); MLP and LSTM were tried and did worse. Xavier init, Adam, lr 1e-4. Delay handling: roll the delayed observation forward 15 frames with the forward model before querying. In planning, the model's predicted opponent action is fed into the forward model at each simulated step (opponent no longer random/idle).

**Online learning — three variants**, trained on the current opponent's (state, action) pairs collected during a round, **updated only in the between-round break**, starting from random init each match:
- **SL**: cross-entropy to observed opponent action (behavior cloning).
- **Q**: treat logits as Q-values; N-step return target, MSE, minibatch.
- **PG**: REINFORCE on opponent's actions with reward = normalized HP difference (from the opponent's perspective), i.e., model learns which actions *pay off* for the opponent, not just which are frequent.
- RHEAOM-R: random (untrained) model as control.

**How the opponent model is evaluated.** *Not* by prediction accuracy (no numbers reported; authors only say SL has higher accuracy than RL variants). Evaluated by downstream win rate and HP difference, learning curves over rounds, and self-play convergence.

## 3. Experiments (200 rounds per match, 5 repeats, mirror characters)
**Table II – RHEA variants vs each other, mean win rate (%) GAR / LUD / ZEN:**
| Agent | GAR | LUD | ZEN |
|---|---|---|---|
| RHEA | 51.3 | 24.1 | 31.1 |
| RHEAOM-R | 53.5 | 41.4 | 40.7 |
| RHEAOM-SL | 50.0 | 61.1 | 54.7 |
| RHEAOM-Q | 42.4 | 59.8 | 58.7 |
| RHEAOM-PG | 52.8 | 63.7 | 64.9 |

**Table III – vs five 2018 competition bots (Thunder, KotlinTestAgent, JayBotGM, MogakuMono, UtalFighter), mean win rate:**
| Agent | GAR | LUD | ZEN |
|---|---|---|---|
| RHEA | 84.5 | 69.6 | 73.4 |
| RHEAOM-R | 85.2 | 79.9 | 50.6 |
| RHEAOM-SL | 87.6 | 91.2 | 84.0 |
| RHEAOM-Q | 86.4 | 92.3 | 82.7 |
| RHEAOM-PG | 89.6 | 93.1 | 87.3 |

**Table IV – model plugged into MCTS (Thunder → ThunderOM-SL/PG):** improves Thunder on all characters; RHEAOM-PG still wins 79.1/60.3/48.3 (GAR/LUD/ZEN) vs ThunderOM-PG and 81.1/69.6/54.2 vs ThunderOM-SL.

**Table V – 2019 FTGAIC:** RHEAOM-PG ("RHEAPI") 2nd (122 pts) behind ReiwaThunder (133); only top-5 entry without MCTS; largest HP difference and lowest frame cost in Standard league; slower than ReiwaThunder in Speedrunning except LUD.

Self-play convergence (Fig. 4): SL and PG converge faster/with less oscillation. Learning curves (Fig. 5): Q and PG improve early, PG stays ahead.

Compute: everything runs within the real-time 16.67 ms/frame budget on a CPU; learning happens in ~5-s round breaks.

## 4. Limitations
- Authors: does not beat everyone; hand-tuned heuristics (ReiwaThunder) still win against known opponents; real-time limits model capacity; depends on a perfect built-in forward model; FightingICE-specific features; no prediction-accuracy evaluation.
- Mine: (i) The "opponent model" is a **stationary reactive policy** p(a | s) with no latent state, no timing/duration, no memory of the opponent's own sequence—it cannot express combos, feints or phase changes. (ii) No uncertainty: a point estimate fed into planning; no exploration value. (iii) RHEAOM-R beating RHEA shows part of the gain is just "assume the opponent does *something*" rather than learned knowledge. (iv) Win rate vs. competition bots conflates model quality and planner quality; no ablation of prediction quality vs. win. (v) Random init each match: no transfer across opponents. (vi) PG's advantage (weighting rare deadly moves) is a hint that the *loss* should be decision-aware, not likelihood-only.

## 5. Ideas to steal
1. **Baseline for MiniSouls: "RHEA/MPC + linear softmax opponent model updated between attempts."** Cheap, published, and exactly the non-Bayesian, no-latent-structure foil to our HSMM posterior. Implement in NumPy in an afternoon: 18-ish features → softmax over boss move ids, cross-entropy, update after each death. Predicts it will learn frequent moves fast and miss rare delayed/feint moves—test in E2/E4.
2. **Random-model control (RHEAOM-R)**: add an "untrained boss model" ablation to separate "planning against any opponent" from "planning against the learned opponent."
3. **Decision-aware weighting.** Their PG result says rare high-damage moves deserve more weight than their frequency. For our intuition-net loss or the planner's hypothesis weighting, add an evaluation that reports **damage-weighted predictive log-likelihood** (log p(next move) weighted by its damage), not just accuracy.
4. **Report what they didn't:** per-attempt **next-move prediction accuracy and log-loss** alongside win rate, to separate model learning from acting (we have "induction speed" already—keep both).
5. **Observation delay + forward-model roll-forward**: if we add perception lag (human-like ~200 ms), condition the filter on delayed cues and roll the belief forward; a clean knob for human comparisons.
6. **Planner budget realism:** 7 × 4 sequences evaluated within 16 ms is enough to be competitive; our 7 actions × 64 rollouts may be overkill—use their numbers to justify a small planner.
7. **External-validity testbed**: FightingICE runs on a laptop (Java + Python bridge); a stretch result would be "our HSMM opponent model replaces their linear model inside RHEA and improves win rate vs. 2018 bots with fewer rounds."

## 6. Structures and models of thought
- **Model-based planning with an explicit opponent model** (forward model + predicted opponent action) is the minimal theory-of-mind architecture; our design enriches the opponent model from p(a|s) to a latent-state generative program, which is what's needed for timing and feints.
- **Learning between rounds (offline consolidation)** while acting fast online mirrors our "slow mind between deaths / fast filter during fight" split.
- **Utility-weighted learning** (PG) ≈ resource-rational allocation of model capacity to consequential events—supports the metareasoning gate attending to dangerous uncertainty, and suggests priors/proposals should prioritize high-damage moves.
- Shows the limits of a **reactive, memoryless opponent representation**; good evidence for why "reading the boss" needs latent segments and duration laws.
