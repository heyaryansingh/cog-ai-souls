# In-context Reinforcement Learning with Algorithm Distillation (AD)

## 1. Citation and reading status
- Laskin, Wang, Oh, Parisotto, Spencer, Steigerwald, Strouse, Hansen, Filos, Brooks, Gazeau, Sahni, Singh, Mnih. "In-context Reinforcement Learning with Algorithm Distillation." ICLR 2023 (arXiv 2022). https://arxiv.org/abs/2210.14215
- **How I read it:** I read the arXiv HTML full text (main paper plus appendices) through a web-fetch tool. Direct download was blocked in my sandbox, so I saw the content through a model-extracted pass of the whole page, not as raw PDF text. All numbers below come from that pass. Figure-only results (exact curves) are described qualitatively.

## 2. Core idea and exact method
- **Idea.** An RL algorithm is a long history-conditioned policy, so distil it by behaviour cloning on *learning histories*, not on expert trajectories. The transformer learns to output the action the source algorithm would take *at that stage of its learning*. At test time it improves its policy in context, with frozen weights.
- **Data.**
  - Run a gradient-based source RL algorithm on many tasks, one random task per run, until convergence. Save every (o, a, r) across all episodes.
  - Sources: UCB (bandit), A3C (Dark Room / Key-to-Door: 2000 histories, 100 actors), distributed Q(λ)/DQN (Watermaze: 4000 histories, 16 actors).
- **Context.** Train on subsequences of length c < T sampled *across episode boundaries*. Tokens are (obs, action, reward), with no return conditioning.
- **Loss.** NLL of the source's action given h_{t-1} and o_t, summed over tasks and time (Eq. 6).
- **Architecture (Table 1).**
  - GPT-style causal transformer: embed 64, 4 layers, 4 heads, FFN 2048, absolute positions, post-norm, dropout 0.1.
  - Attention dropout 0.5 (0 on Dark Room Hard). Label smoothing 0.2 on Dark Room Hard only.
  - **Sequence (token) mask probability 0.3** (0.5 on Dark Room Hard).
  - My estimate: about 1.1M parameters (each layer is about 16k attention + 262k FFN).
  - Watermaze image encoder: 3 conv blocks, 128 channels, kernel 5, stride 2, then 256-unit linear.
- **Optimisation.** Batch 128; Adam (0.9, 0.99); cosine LR peaking at 3e-4.
- **Evaluation.** The model rolls out and fills its own context queue with the last c transitions. Scores are averaged over 5 training seeds × 20 eval seeds.

## 3. Experiments
| Env | Details |
|---|---|
| Adversarial bandit | 10 arms, 100 trials. Train: odd arms best 95% of the time; **test: even arms** (out of distribution) |
| Dark Room | 9×9, partially observed (sees only x,y), H = 20, r = 1 per goal visit |
| Dark Room Hard | 17×17, reward exactly once |
| Dark Key-to-Door | 9×9, invisible key and door, H = 50; 6561 tasks, <2k seen in training |
| DMLab Watermaze | 72×96×3 pixels, 8 actions, H = 50, continuous goal space |

- **Baselines.**
  - ED (expert distillation): same model, expert data only.
  - The source algorithm itself.
  - RL² (LSTM, 1B env steps), used as an upper bound.
- **Key results.**
  - AD in-context reinforcement-learns in every environment; ED mostly does not.
  - AD matches asymptotic RL² on the Dark environments and comes within 13% on Watermaze.
  - On the bandit, AD generalises out of distribution better than RL² (close to UCB).
  - AD is **more data-efficient than its source**, though the source reaches a slightly higher asymptote.
  - AD improves a demonstration policy in context, where ED only holds it steady.
  - Attention spans several episodes.
- **Ablations that matter for us.**
  - **Number of training tasks:** 1, 9 or 18 tasks gave *no* in-context RL. 37, 75 and 151 gave some. The best models used 1212 and 2424 tasks (18% and 37% of the domain).
  - **Context length:** a context covering 2–4 episodes is needed for near-optimal in-context RL; about 1 episode gives only early signs.
  - **Subsampled histories:** keeping every 10th episode (2000 → 200) still works, and AD then learns *faster* than its source.
  - Mask probability 0.3–0.5 works best. In-context RL appears at every model size; bigger models help on Key-to-Door.
  - A transformer beats an LSTM, but both work.
- **Compute.** No hardware or wall-clock figures are given. The models are tiny, so cost is dominated by generating source learning histories.

## 4. Limitations
- **Reported by the authors:**
  - Needs multi-episode context, so long-episode environments need longer-context models.
  - Environments were chosen to have short episodes; Atari and Gym were excluded.
  - Bounded by the source algorithm.
  - Offline only.
  - Return conditioning and world models were not explored.
- **Noticed by me:**
  1. The in-context algorithm is opaque. You cannot read off *what* it believes about the task, so there is no calibration and no explanation of a death.
  2. Out-of-distribution evidence is thin: one bandit flip plus unseen-but-same-family grid tasks. Nothing tests structurally new dynamics, which is exactly our T3/T4.
  3. The required task diversity (≥ about 40 tasks just to see in-context RL, about 1000+ to do well) is far above our "40 training bosses."

## 5. Ideas to steal
1. **The AD baseline will fail with only 40 training bosses.** Their task-count ablation shows no in-context RL below 37 tasks, and a large gain going from 151 to 1212. Plan: train AD on **procedurally sampled bosses from the prior Θ (1000+ programs)**, not only the 40 hand-designed ones. Report AD-40 against AD-1000 as an extra point: "black-box test-time learning needs more task diversity than our structured prior."
2. **Context must span 2–4 attempts.**
   - Boss attempts are long (hundreds to thousands of ticks), so raw-token AD is infeasible. Use *subsampled* histories (their every-10th-episode trick) and/or tokenise at the **move or event level** (cue onset, player action, hit or dodge outcome) instead of per tick.
   - That cuts each attempt to roughly 50–200 tokens, which fits 4 attempts in about 1k tokens. This is the same budget as the intuition net (64 × 1024 tokens).
3. **Cost.** Use their exact architecture (d = 64, 4 layers, 4 heads, mask probability 0.3, about 1M parameters) so the AD baseline costs no more than the intuition net (CPU-hours). Reuse the PPO runs as source histories, as the plan already says. Subsample them.
4. **Demonstration-prompting test.** Seed AD's context with a human's first 2 deaths and check whether it improves on them. Run the same protocol for our agent, with the human deaths as evidence. This is a cheap human-agent "imitation and improvement" experiment.
5. **Metric.** Use their "data efficiency relative to the source" framing: deaths-to-first-win for AD against deaths-to-first-win for the PPO that generated its data.
6. **Out-of-distribution flip.** Their bandit flip (train on odd arms, test on even) is a template for a minimal T4 rule-change probe, for example swapping which side a delayed attack punishes.

## 6. Structures and models of thought
- **Learning as inference in a sequence model.** AD shows that a small transformer can implement an RL update rule implicitly in its activations. That makes it the right *contrast* for our explicit posterior: both adapt without weight changes, but only ours exposes a belief state.
- **Cross-episode memory is the key ingredient.** It is the computational analogue of "learning from deaths." Our design keeps that memory as an explicit, compressed posterior rather than a raw context window, which avoids the 2–4-episode context requirement.
- **Learning histories as training data.** The distilled policy captures the trajectory of improvement, including exploration. This suggests training the intuition net on *partial* fight logs at every stage of learning (1 death, 2 deaths, ...), so it learns to propose well early as well as late.
