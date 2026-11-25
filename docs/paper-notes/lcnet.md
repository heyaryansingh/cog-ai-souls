# LCNet: Lu et al. 2024

## 1. Citation and access
- Qihong Lu, Tan T. Nguyen, Qiong Zhang, Uri Hasson, Thomas L. Griffiths, Jeffrey M. Zacks, Samuel J. Gershman, Kenneth A. Norman. "Reconciling shared versus context-specific information in a neural network model of latent causes." *Scientific Reports* 14:16782, 2024. doi:10.1038/s41598-024-64272-5. PDF: https://cocosci.princeton.edu/papers/Lu2024a.pdf
- **What I read.** The main text of the PDF, through a text-extraction tool. I did not read the supplements (S3 on stickiness, S4 on the episodic-memory shortcut, S6–S7 on the SEM 2.0 comparison and RSA). The main text does not give α, λ, the context-vector dimension, hidden sizes or learning rates.

## 2. Core idea and exact method
**The problem.** SEM gives each latent cause (LC) its own network. That protects against interference but shares nothing. A single shared network shares everything but forgets catastrophically.

**LCNet's solution.** One shared network. Each LC is just a **random context vector** c_k ~ N(0, I) that modulates the hidden layer.
- The **input→hidden** weights learn structure shared across contexts.
- The **context→hidden** weights learn the context-specific deviations.
- A new LC samples a new random vector.

**Latent cause inference (LCI), every time step:**
- **Likelihood.** For each existing LC k, compute the prediction loss MSE(f(x_{t−1}, c_k), x_t). A new LC is scored with a fixed preallocated random vector. The likelihood is a softmax over negative losses.
- **Prior (sticky CRP):** P(z_t = k | z_{1:t−1}) ∝ count_k + λ·I[z_{t−1} = k] for k ≤ K, and α for k = K+1.
- **Posterior.** Prior × likelihood. Use the local MAP: pick the argmax LC, then train the network on the current observation under that LC's context vector.
- **Boundaries.** An event boundary is an LC switch, not a threshold on prediction error.

**Recurrent version (Sim 3).** A GRU with the standard equations: z_t = σ(W_z[h_{t−1}, x_t]), r_t = σ(W_r[h_{t−1}, x_t]), h̃_t = tanh(W[r_t ⊙ h_{t−1}, x_t]), h_t = (1−z_t) ⊙ h_{t−1} + z_t ⊙ h̃_t.

## 3. Experiments
**Sim 1: function learning.**
- Four polynomials, each = a shared term + an idiosyncratic term. Training is fully blocked.
- Input: x plus a 128-d noisy binary context cue, with 10% of bits flipped.
- N = 100 runs.
- Results:
  - The standard network forgets severely.
  - SEM does not forget, but learns each new function from scratch.
  - **LCNet forgets only slightly *and* learns later functions faster**, because it reuses the shared term.
  - If the LCI module is lesioned, LCNet outputs the shared term.

**Sim 2: blocked vs interleaved schema learning** (cafe/bar transition graphs, from Beukers et al.).
- LCNet does better with blocked training than interleaved training, as humans do. A standard RNN shows the opposite.
- The blocked advantage depends on the sticky prior matching the environment's autocorrelation (Supp. 3).
- Metric: cluster purity = (1/n) Σ_k max_j |Ĉ_k ∩ C_j|.
- N = 20.

**Sim 3: naturalistic video (the META dataset).**
- About 25 hours of video, compressed to 30-d frame vectors (76% of variance kept). The model is trained only on next-scene prediction.
- Metrics:
  - scaled point-biserial correlation between model and human boundaries, at coarse and fine grain;
  - AMI between inferred LCs and 45 ground-truth sub-event labels.
- Both are significant (p < 1e−5, N = 12). The fit is slightly worse than SEM 2.0.

## 4. Limitations
**Reported by the authors:**
- Full LCI runs at every step, which is costly and biologically implausible. Supp. 4 tries an episodic-memory shortcut.
- LCs form a flat list: cost grows linearly with their number, and redundant LCs cannot be merged.
- Random context vectors carry no similarity structure between contexts.
- Performance is sensitive to how well the stickiness matches the environment.
- Only one LC is active at a time.
- There is slight forgetting.

**My own observations:**
- The local MAP is greedy, with no retrospective re-segmentation.
- The likelihood is a softmax over MSE, not a calibrated probability, so the posterior's scale is arbitrary.
- There is no duration model. Stickiness is a geometric-duration assumption, the HMM equivalent, which is poor for fixed-length wind-ups.

## 5. Ideas to steal
1. **Shared network + per-cause embedding = the right parameterization for the intuition net's handling of boss families and moves.**
   - Give each inferred move (and each family) a context embedding that conditions a *shared* timing/cue-prediction network.
   - Shared structure, like "wind-ups end in a strike," lives in shared weights. Boss-specific deviations live in the embedding.
   - This is exactly the plan's "shared in weights, context-specific in latent causes" split (§1.1). Unlike LCNet, **the embeddings should not be random**: either infer them (amortized), or derive them from the primitive id, so that similar moves get similar codes. This fixes LCNet's stated limitation.
2. **Lesion test.** LCNet with LCI lesioned outputs the shared term.
   - Our equivalent: the agent with family/move identity removed should fall back to the Θ prior means.
   - This is a clean ablation that checks shared structure was actually learned. Add it to E2 as "no-CRP" with the prediction "behaves like Θ-mean agent."
3. **Blocked vs interleaved curriculum as an experiment.**
   - Train or evaluate on bosses presented blocked (all attempts on boss A, then boss B) or interleaved (A, B, A, B).
   - A sticky-CRP agent should do better when blocked. Humans should too, per Beukers et al.
   - This is a cheap, human-comparable E3/E4 manipulation. The interleaved condition also tests whether family recognition re-identifies a boss on re-encounter.
4. **Match stickiness to environment autocorrelation.** LCNet's blocked benefit depends on it. In our HSMM, the analog is the duration prior. Explicit-duration HSMMs beat a sticky HMM for wind-ups, so cite this as motivation for HSMM over sticky-HMM.
5. **Metrics:**
   - **AMI between inferred move labels and ground-truth moves** (an induction-quality metric that does not care about label permutation);
   - **cluster purity**;
   - **point-biserial correlation between agent-inferred boundaries and true move onsets** (or human-annotated ones).
   - Add them to the stage-2 gate check (segmentation accuracy).
6. **Shortcut instead of full LCI at every tick.** Run full inference over family/move identity only when surprise or ESS triggers it (our surprise-gated rejuvenation), and otherwise keep the current cause. This is their Supp. 4 idea and our §3.2 design. Cite it as convergent.

## 6. Structures and models of thought
- **Latent causes indexed by embeddings into a shared function approximator.** This is a neural-compatible way to have a CRP over contexts without one network per context. It maps onto the family CRP plus a shared intuition net.
- **The sticky CRP prior** captures three things: an unknown number of causes, a simplicity bias (α) and temporal persistence (λ). Our HSMM generalizes persistence to explicit durations.
- **Inference via predictive loss.** "Which cause best predicts the next observation" is the same computation as the HSMM per-hypothesis marginal likelihood. Our exact likelihood is a principled version of their softmax-over-MSE.
- **Known gaps worth fixing:** hierarchy (families → moves), merging (our split/merge moves), structured context codes. Our design addresses all three, which is a citable contrast for the report.
