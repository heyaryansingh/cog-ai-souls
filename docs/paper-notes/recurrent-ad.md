# Scalable In-Context RL with Recurrent Algorithm Distillation (RAD)

## 1. Citation and reading status
- Yuanqing Ma, Zhenrui Zheng, Chenjun Xiao (CUHK-Shenzhen). "Scalable In-Context Reinforcement Learning with Recurrent Algorithm Distillation." arXiv 2609.35333, 2026 (venue not stated on the page). https://arxiv.org/abs/2609.35333
- **How I read it:** I read the arXiv HTML full text (main paper plus appendices) through a web-fetch tool, as a model-extracted pass, not as raw PDF. The paper gives most results only as figures, so exact regret and return values are not available. FLOPs and hyperparameter tables were available in the text.

## 2. Core idea and exact method
- **Problem.** AD's raw context window grows quadratically in cost, and it forgets anything older than the window.
- **Idea.** RAD splits memory in two:
  - a **working memory** of the last K raw transitions;
  - a **long-term memory** of m latent tokens.
- **Components.**
  - A **Compression Transformer** q_ω turns a chunk of history into m latent tokens. It is pretrained with a reconstruction decoder, following Context Cascade Compression (C3).
  - A **GRU-style gate** U_φ merges new latents with old ones. The first compression is a direct write.
  - After each compression, the working memory keeps only the last p transitions.
  - The **AD policy transformer** π_θ predicts the action from [latents, working memory, current observation].
- **Tokenisation.** o, a and r are separate tokens (3 per step). Chunk length is a multiple of 3, so a timestep's tuple is never split.
- **Training.**
  - Phase 1: compressor reconstruction pretraining (20k–50k steps).
  - Phase 2: joint action cross-entropy L_RAD over policy, compressor and gate (100k steps).
  - Gradients flow through only the last **G** compression events (truncated backpropagation through time).
  - Darkroom curriculum on the maximum number of compressions: 1, then 3 at 30k steps, 6 at 50k, unlimited at 75k.
- **Theory claim.** AD approximates a posterior-sampling policy, so distal history matters for inference.
- **Sizes.**
  - Policy: GPT-2-style, d = 64, 4 layers, 4 heads (8 on Meta-World), FFN 256.
  - Compressor: d = 64, 2–4 layers.
  - No parameter counts are given. By my estimate each network is about 0.2M parameters.

| | Delayed bandit | Darkroom | Key-to-Door | Meta-World |
|---|---|---|---|---|
| AD context | 150 / 900 | 240 | 300 | 1200 |
| RAD K / p / m / G | 50/5/15/∞ | 25/5/15/5 | 52/10/60/5 | 80/20/60/6 |
| RAD policy context | 165 | 90 | 216 | 300 |
| FLOPs/action AD → RAD | 59.8M or 263M → 43M | 145M → 28M | 203M → 86M | 1.84G → 131M |

## 3. Experiments
- **Environments.**
  - Delayed bandit: 50 pulls, then N_delay zero-reward distractor steps, then 50 more pulls. Train with delay 50; test with 0, 50, 100 and 200.
  - Darkroom: 81 goals, 9:1 train/test split.
  - Dark Key-to-Door.
  - Meta-World ML1 (Reach, Push).
- **Source algorithms.** UCB and PPO. PPO used 100 streams and 100k steps on the grid tasks, 1M steps on Meta-World.
- **Results (figures only).**
  - AD_short forgets what it learned before the delay and has to re-explore.
  - RAD matches UCB at delays of 100 and 200, while AD_long degrades.
  - RAD beats AD on Darkroom. On Key-to-Door it beats AD but trails DPT and IDT, which need extra supervision.
  - RAD generally matches or beats AD on Meta-World.
  - RAD uses more than 80% less compute in most environments.
- **Ablations.**
  - GRU gate beats replace, residual and multiplicative gates.
  - A plain autoencoder compressor beats VAE and VQ-VAE late in training (VQ-VAE learns faster early but is noisier).
  - **15 latent tokens is enough** (best among the sizes tried).
  - The per-token tokenizer and a DPT-style tokenizer give similar returns.
- **Compute.** 8× RTX 4090, PyTorch, AdamW, gradient clipping at 1, cosine schedule. No wall-clock times are reported.
- **Convergence (delayed bandit).** All methods start near regret 33. AD_long converges to 6.9. RAD is slowest, reaching its best at about 60k updates.

## 4. Limitations
- **Reported by the authors:**
  - Training is unstable because of backpropagation through time; gradients are truncated to G events.
  - Slow convergence.
  - Smaller savings with sparse rewards (0.42× on Key-to-Door).
- **Noticed by me:**
  - Results are only in figures, with no confidence numbers in the text.
  - Still bounded by the source algorithm and still a black box.
  - Out-of-distribution testing only covers longer delays, not new dynamics.
  - The 4090 cluster is overkill for these model sizes. A single-GPU or laptop run seems plausible, but the paper does not show it.

## 5. Ideas to steal
1. **Use RAD-style memory for the AD baseline.**
   - Boss attempts are long, so plain AD cannot hold the 2–4 attempts it needs. A RAD-lite version keeps the current attempt as raw tokens (K of about 50–100 events) and compresses each finished attempt into **m = 15 latent tokens** with a GRU gate.
   - This gives a fair black-box "learns across deaths" baseline at roughly 30–130M FLOPs per action, which a laptop can run in real time at about 10–30 decisions per second.
2. **The delayed-bandit design maps directly onto the boss setting.**
   - Their test inserts zero-information steps between informative ones. The boss analogue is: does the agent remember phase-2 evidence gathered two attempts ago, after an attempt where it died early in phase 1?
   - Add this as an E1 sub-probe, a **"retention across uninformative deaths"** curve. Our explicit posterior should be flat (no forgetting); AD_short should lose the evidence.
3. **Compressed-attempt summaries for the intuition net.** RAD shows that about 15 latent tokens per long history suffice. Our intuition net's 1024-token fight log could hold *per-attempt summary tokens* plus the raw current attempt, which lets it scale to 20+ attempts in E1.
4. **Report FLOPs per decision** for every agent (AD, RAD-lite, amortized-only, SMC + q_φ, SMC with prior proposals). That gives E5/E6 a hardware-independent cost axis, using the same metric RAD uses.
5. **Training-length curriculum.** If the intuition net's recursion or the fight-log length becomes unstable, ramp the number of attempts in context (1, then 3, then 6, then all), as their compression curriculum does.

## 6. Structures and models of thought
- **Working memory plus a consolidated long-term store with a gated update.** This is a cognitive split that mirrors the plan's "exact filter within a fight, posterior over programs across fights." RAD learns the consolidation; we do it with Bayes. The comparison makes concrete what an explicit sufficient statistic (the posterior) buys over a learned compression.
- **The posterior-sampling interpretation of AD** supports framing all the in-context learners as approximate Bayesian agents. E6 can then measure each one's gap to an exact Bayes oracle, which our HSMM filter can compute.
