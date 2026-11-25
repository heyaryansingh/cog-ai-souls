# Distribution Transformers: Fast Approximate Bayesian Inference With On-The-Fly Prior Adaptation

## 1. Citation and how much I read
- George Whittle, Juliusz Ziomek, Jacob Rawling, and a fourth author the fetch extracted as "Maike A. Osborne". This is most likely Michael A. Osborne (Oxford); **verify before citing**.
- arXiv 2502.02463 v3 (18 May 2026). The project plan lists it as ICML 2026; the extracted text did not state a venue. https://arxiv.org/abs/2502.02463
- **Read:** essentially the full text (about 60k characters: main text plus Appendices A–J) via the arXiv HTML with a summarising fetch tool, plus a second pass quoting Appendices C.2, D, and E verbatim.
- **Caveats on extraction:**
  - Table 8 (training statistics) came through flattened. I give its value ranges, not row-by-row values.
  - Its "model size" column is ambiguous. The trailing numbers (0.29–23.9) are plausibly millions of parameters.
  - Some cells of Tables 1 and 2 were garbled.

## 2. Core idea and exact method
- **Map (prior, data) → posterior**, with both prior and posterior represented as **Gaussian mixture models**. Each GMM component (weight, mean, Cholesky covariance) is **one token**.
- **Architecture.**
  - A transformer *decoder*: 6 units, latent size 64, MLP hidden size 2048, 8 heads. It cross-attends from prior-component tokens to data tokens and uses **no positional encodings**, so it is permutation-equivariant over components and data.
  - Prior embedding: an MLP on log-weights and Cholesky-log parameters.
  - Unembedding: inverts the embedding, with a softmax over components for the weights.
- **Training.** Sample from a hierarchical joint p(ϕ) p(x|ϕ) p(z|x), where ϕ indexes the prior (the meta-prior only needs to be samplable).
  - **Posterior loss:** E[−log q(x | z, prior_ϕ)], the expected NLL under the output GMM. Proposition 3.1 shows this equals the expected forward KL(true posterior ‖ q) up to a constant. This is the same objective as inference compilation.
  - **Prior loss:** E[−log q(x|ϕ)] (no data), added as a regulariser and to encourage "latent-space conjugacy."
- **Training regime.** Adam, no dropout or weight decay ("overfitting here is impossible as the model sees each sample only once"), cosine schedule with 5 warm-up epochs.
- **Sequential use.** Because input and output are both GMMs, the posterior after step t is fed back in as the prior for step t+1: an approximately conjugate, constant-cost filter.

## 3. Experiments and numbers
**§4.1 Inverse-gamma prior, normal-variance likelihood (analytic posterior).** Expected KL over 1000 unseen problems:
| Method | Narrow KL | Wide KL | Time per 1000 problems |
|---|---|---|---|
| SVI | 0.0425 | 0.0558 | 148 s |
| PFN-5000 | 0.0038 (±0.079) | 0.2935 | 0.003 s |
| TabPFNv2 | 0.0112 | 0.1513 | 1.52 s |
| ACE-5 | 0.0094 | 0.0048 | 0.037 s |
| DT-2 | 0.0044 | 0.0058 | 0.014 s |
| **DT-5** | **0.0004** | **0.0003** | 0.016 s |
DT-5 is about 9000× faster than SVI with about 100× lower KL.

**§4.2.1 GP with hyperpriors** (5-D input, 5 observations):
- DT has the best PPD NLL (0.81) and hyperposterior NLL (0.31), against PFN (0.90 / 1.53) and ACE (1.08 / 0.35).
- DT's time is 9.5 s against PFN's 9.2 s.
- Calibration MMD: DT 0.0036 against ACE 0.0035 on the PPD.

**§4.2.2 Quantum two-level system:** DT matches TabPFNv2 and ACE on NLL and is faster. MMD 0.0027, the best listed.

**§4.3.1 Sensor fusion filter** (4-D state, nonlinear non-Gaussian sensors). Expected NLL and time per 100-series step:
| Method | Expected NLL | Time per step |
|---|---|---|
| EKF | 95.9 | 0.010 s |
| Particle filter | **−0.244** | 0.818 s |
| DT | −0.197 | 0.017 s |
DT is about 48× faster than the PF and slightly worse in NLL.

**§4.3.2 Factor stochastic volatility** (10 factors, 30 observations per step): DT matches the NLL of a PF with very many particles at about 1000× lower cost per step (figure only).

**Appendix results:**
- **C.2 Meta-prior shift (OOD):** train on the wide meta-prior InverseGamma(4,6); test on InverseGamma(24,6) for rate and scale ("largely non-overlapping mass").
  - **DT-5 KL rises 18×, from 0.0003 to 0.0054.**
  - ACE-5 goes from 0.0058 to 0.0132.
  - DT is still better than ACE, but it degrades by an order of magnitude.
- **C.3 Prior-loss ablation:** without the prior loss, posterior KL is unchanged (0.0004), but the prior reconstruction KL rises from 0.0004 to 0.30. My mapping of table rows to conditions is inferred from garbled extraction.
- **C.4 Component-count generalisation:** trained with 10 components, tested with 2/5/10/20 components, KL is 0.149 / 0.099 / 0.087 / 0.095. Mild degradation off the training value.
- **C.5 Error accumulation over recursion:** excess NLL is indistinguishable from zero up to depth ≈ 70, then rises slowly. At depth 100 the DT retains about 95% of the PF's information.
- **C.1 Scaling with latent dimension (1–24):** DT stays at least 10× better than SVI.

**Compute (Appendix E, Table 8):**
- **Hardware:** an **RTX 2080 Super laptop GPU (8 GB)** for §4.1, §4.2.2, and §4.3.1; an RTX 3090 for §4.2.1 and §4.3.2.
- **DT runs** (as far as the flattened table can be read): batch sizes 1000–5000; 20–200 epochs of 100 batches; **10–80M total training samples**.
- **Training times about 170 s to about 7,500 s** (≈3 min – 2 h), with model sizes plausibly about 0.3–24M parameters.
- Learning rates 1e-4 to 5e-3.

## 4. Failure modes and limitations
- **Reported by the authors:**
  - higher offline training cost, since training spans a space of priors;
  - needs a reasonably well-specified meta-prior;
  - cost is quadratic in the number of components and in latent dimension (full covariance);
  - hyperparameters were tuned by hand;
  - error accumulates over long recursions.
- **Noticed by me:**
  - The GMM representation suits low-dimensional continuous latents. It does not cover discrete or structured latents (segmentation, program structure, family identity), which are the core of our boss posterior.
  - Again there is no weighting or correction, so the 18× OOD KL increase cannot be repaired at test time.

## 5. Ideas to steal
1. **The compute budget is laptop-feasible.** Small transformers (latent size 64, 6 layers) trained on 10–80M fresh synthetic samples for minutes to about 2 hours on an 8 GB laptop GPU reached near-exact posteriors. This is an external sanity check for our "1.5M params, 4–8 h, 300–500M tokens" estimate. Note that their samples are tiny (few tokens) whereas ours are 1024-token logs, so the token throughput is the number to verify.
2. **The OOD protocol from C.2.** Train under meta-prior A; test under a shifted meta-prior B with "largely non-overlapping mass"; report expected KL (or NLL) on A and on B, and the **ratio** (18× for DT). For E6, define our **amortisation-gap ratio** = (posterior error of q_φ alone on T_k) / (error on T1), alongside the same ratio for SMC + q_φ. A large ratio for amortised-only next to a ratio near 1 for SMC + q_φ is the headline.
3. **The prior-adaptation idea.** Condition the intuition net on the *current prior* (the hierarchical timing prior Θ and the family library as tokens), not just the log. Then when Θ is updated across bosses, the net adapts without retraining. Their meta-prior sampling during training (randomise the prior per example) is how to get there. In our design, sample Θ from a hyperprior per training boss, and feed a summary of Θ as tokens.
4. **The prior-reconstruction auxiliary loss.** Add a cheap auxiliary head trained to reproduce the prior with no data. It regularises the net and gives a clean check that it represents "what I believe before evidence."
5. **Recursive (filtering) use and error-accumulation test.** Their depth-to-failure analysis (about 70 recursions before measurable degradation) suggests a parallel experiment: feed the net's posterior summary back in across attempts (attempt 1 → 2 → … → 10) and measure the drift against SMC.
6. **Expected NLL as the main proposal metric.** E_{p(B,log)}[−log q_φ(B|log)] on held-out prior samples equals forward KL up to a constant. Report it per head (segmentation, label, timing, family) as the stage-2 gate metric, together with ESS.

## 6. Structures and models of thought
- **Beliefs as sets of hypothesis tokens.** Representing a belief state as an unordered set of weighted components, updated by attention to evidence, is a neural analogue of a particle or mixture belief. It is close in spirit to our weighted set of boss-program hypotheses, and suggests the intuition net could *read the current particle set* as tokens (recursive refinement with SMC feedback, §3.3 of our plan).
- **Approximate conjugacy enabling recursion.** An inference module whose output format equals its input format can be iterated, which gives cheap online belief updating. It mirrors how a fixed belief format (for example, a sufficient-statistics prior over timings) enables learning that accumulates across attempts.
- **Prior-conditioned intuition.** The fast system takes the current prior as input, so meta-learning changes intuition without retraining. This is a mechanism for transfer across bosses.
