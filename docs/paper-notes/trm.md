# Less is More: Recursive Reasoning with Tiny Networks (TRM)

## 1. Citation and reading status
- Alexia Jolicoeur-Martineau (Samsung SAIL Montréal). "Less is More: Recursive Reasoning with Tiny Networks." arXiv 2510.04871, Oct 2025. https://arxiv.org/abs/2510.04871
- **How I read it:** through a web-fetch tool, as a model-extracted pass. The arXiv HTML version cut off in §4.2, so I read the remainder (§4.3 onward: architecture details, hyperparameters, results tables, failed ideas, conclusion) from the arXiv PDF. Between the two I covered the whole paper, but not as raw text. One discrepancy: the footnote on the MLP parameter count (5M vs 19M) conflicts within the paper itself.

## 2. Core idea and exact method
- **Background.** TRM simplifies HRM (Hierarchical Reasoning Model), which uses two 4-layer transformers f_L and f_H, a fixed-point / implicit-function-theorem 1-step gradient, and ACT with an extra forward pass.
- **TRM uses one tiny 2-layer network** and two carried states:
  - **y**, the embedded current answer;
  - **z**, a latent reasoning state.
- **One recursion.** Do n = 6 updates z ← f(x, y, z), then one update y ← f(y, z). The same network is used for both; whether x is included tells it which task it is doing.
- **Deep recursion.** Repeat that T = 3 times. The first T−1 passes run **without gradients**; the last pass is backpropagated **in full** (no 1-step approximation). Effective depth per supervision step is T(n+1)·layers = 42.
- **Deep supervision.** Up to N_sup = 16 steps. Each step starts from the previous (y, z), *detached*, and gets its own loss. This is the main source of the gains.
- **Halting (ACT).** A single binary Q-head predicts "is the answer correct," with a BCE loss. No extra forward pass is needed (HRM's needed one). Used during training to stop early on solved examples.
- **Details.**
  - Hidden size 512. RMSNorm, SwiGLU, rotary embeddings (inherited from HRM).
  - **EMA of weights 0.999.** Stable-max cross-entropy.
  - AdamW (0.9, 0.95), 2k warmup, batch 768.
  - LR 1e-4 (embeddings 1e-2 on ARC). Weight decay 1.0 (Sudoku, Maze) or 0.1 (ARC).
- **Attention-free variant (TRM-MLP).** Replaces self-attention with an MLP over the sequence length (MLP-Mixer style). It is better for small fixed contexts (Sudoku's 81 cells) and worse for 30×30 grids.

## 3. Experiments
- **Data** (all small-data regimes with heavy augmentation):
  - Sudoku-Extreme: about 1k training puzzles, 423k test, 1000 shuffle augmentations per puzzle.
  - Maze-Hard (30×30): 8 dihedral augmentations.
  - ARC-AGI-1/2: 1000 augmentations (colour permutation, dihedral group, translation).
- **Results:**

| Method | Params | Sudoku-Ext | Maze-Hard | ARC-1 | ARC-2 |
|---|---|---|---|---|---|
| Direct prediction | 27M | 0.0 | 0.0 | 21.0 | 0.0 |
| HRM | 27M | 55.0 | 74.5 | 40.3 | 5.0 |
| TRM-Att | 7M | 74.7 | 85.3 | 44.6 | 7.8 |
| TRM-MLP | 5M (Sudoku) / 19M | 87.4 | 0.0 | 29.6 | 2.4 |
| Gemini 2.5 Pro 32K | ? | – | – | 37.0 | 4.9 |
| o3-mini-high | ? | 0.0 | 0.0 | 34.5 | 3.0 |
| DeepSeek R1 | 671B | 0.0 | 0.0 | 15.8 | 1.3 |

- **Ablations on Sudoku-Extreme (Table 1), accuracy:**

| Variant | Accuracy |
|---|---|
| **Full TRM** | **87.4** |
| With ACT's extra forward pass | 86.1 |
| Separate f_H and f_L (10M params) | 82.4 |
| No EMA | 79.9 |
| 4 layers, n = 3 | 79.5 |
| Self-attention instead of MLP | 74.7 |
| T = 2, n = 2 (depth 12) | 73.7 |
| 1-step gradient | 56.5 |

- **Cited HRM analysis (ARC).** Deep supervision alone took accuracy from 19% to 39%; recursion added only 35.7% to 39.0%.
- **Compute.**
  - Sudoku: 1× L40S, under 36 hours (60k epochs).
  - Maze: 4× L40S, under 24 hours.
  - ARC: about 3 days on 4× H100.
- **Failed ideas.**
  - Mixture-of-experts (worse generalisation).
  - Backpropagating through only the last k recursions (no help).
  - Removing ACT altogether (worse).
  - **Tying the input embedding to the output head (large drop).**
  - TorchDEQ fixed-point solving (slower and worse).

## 4. Limitations
- **Reported by the author:**
  - Choices are not optimal across datasets (the MLP variant fails on Maze and ARC).
  - No scaling laws.
  - No theory of *why* recursion beats depth; overfitting is suspected.
  - Supervised and deterministic only: one answer, not a distribution.
- **Noticed by me:**
  1. **Not laptop-scale as published.** 5–7M parameters and 60k epochs on augmented data took 36 hours on one L40S for Sudoku.
  2. A deterministic single answer conflicts with our need for a *proposal distribution* for SMC.
  3. The gains come with extreme augmentation, so generalisation is to symmetric variants, not to structural novelty.
  4. It is fixed-size grid input-output, not sequential evidence.

## 5. Ideas to steal
1. **Deep supervision with detached carry is the thing to copy, more than the recursion.**
   - For the intuition net, carry (y = current proposal embedding, z = latent) across 2–4 refinement passes, detach between passes, and put a loss on every pass.
   - The plan's "unroll 3 passes, loss on every pass" is right. Make the carry **detached** and backpropagate fully through the last pass only. Their 1-step gradient ablation (56.5 vs 87.4) shows the approximation is costly.
2. **The SMC feedback slots in as "x" for the y-update.** Each pass, z ← f(fight-log tokens, y, z, SMC summary) n times, then y ← f(y, z). The SMC summary is ESS, per-head log-weight residuals, and which moves were unexplained.
3. **A halting head becomes the metareasoning signal.**
   - TRM's Q-head predicts "is the answer correct." Train ours to predict "will another pass raise the SMC log marginal likelihood by more than δ," and stop passes when it says no.
   - This also gives E5/E6 a learned "think more?" signal, which is a resource-rational stopping rule.
4. **Sizing.** Their result that *2 layers beat 4 at the same compute* supports keeping the intuition net shallow: try 2 layers × d = 128–256, recursed, instead of 4 layers. Use **EMA 0.999** (worth 7.5 points) and do **not** tie the embedding to the output head.
5. **Attention vs MLP.** If the fight log is fixed-length and short (a per-attempt summary grid), an MLP-mixer may be better. For variable-length logs of 1024 tokens, keep attention (TRM-MLP collapses on long contexts).
6. **Augmentation is cheap data for a small network.** Use symmetries of the boss domain:
   - permute primitive and move labels;
   - mirror left/right;
   - time-shift and rescale durations within the prior;
   - permute family IDs.
   This is their 1000× augmentation analogue and should reduce overfitting to the 40 hand bosses.
7. **Capability target to state.** A 1–2M recursive net trained by inference compilation should beat a same-size non-recursive net at proposal quality (ESS or log marginal likelihood per pass). Report that as an E6 sub-result, mirroring TRM's direct-prediction vs recursive comparison (0% vs 87%).

## 6. Structures and models of thought
- **Iterative refinement of a hypothesis (y) by a scratchpad (z).** This is close to how people revise a theory: propose, check against evidence, fix. Pairing TRM-style refinement with an external verifier (SMC weights) makes it a proper *generate-and-test* loop, which TRM lacks because it gets no feedback between passes.
- **Recursion as depth on demand.** Spending more passes on harder or novel bosses is adaptive computation. That is the same metareasoning principle as the plan's gate, but inside perception and inference.
- **Small networks plus heavy priors (augmentation) generalise better than large ones in the low-data regime.** This is consistent with the project's bet on strong structure over scale.
