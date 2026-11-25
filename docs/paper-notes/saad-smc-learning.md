# Sequential Monte Carlo Learning for Time Series Structure Discovery

## 1. Citation and reading status
- Feras A. Saad, Brian J. Patton, Matthew D. Hoffman, Rif A. Saurous, Vikash K. Mansinghka. "Sequential Monte Carlo Learning for Time Series Structure Discovery." *ICML 2023*, PMLR 202. Code: https://github.com/fsaad/AutoGP.jl (Gen/Julia). (arXiv id not checked.)
- **Reading status: full text read** directly from the course copy /mnt/project-files/cog-ai/text/SMCLearning.txt (PDF-to-text, incl. appendices A–D). Some equations were mangled by two-column extraction; I reconstructed them from context.

## 2. Core idea and exact method
**Model (a probabilistic program over symbolic structures).** Kernel structure k ~ PCFG over grammar B ::= LINEAR | PERIODIC | GAMMAEXP | …; ⊕ ::= + | * | CP; k ::= B | (k1 ⊕ k2). Params θ_i ~ LogNormal(0,1) i.i.d., noise η ~ InvGamma(1,1), f ~ GP(0, k_θ), y_i ~ N(f(t_i), η). Posterior over (k, θ, η) is transdimensional.

**Algorithm 1: SMC via data annealing (resample-move, IBIS-style).**
- Particles (k, φ) from prior, w = 1. Data split into T batches D_j (streaming = online; nested subsets = offline annealing).
- *Reweight*: copy kernel, incremental weight = P(y_j | k, φ, y_{1:j−1}) (exact GP predictive).
- *Resample* if ESS < threshold (M/2); weights reset to mean.
- *Rejuvenate*: N_rejuv involutive-MCMC structure moves, then HMC on continuous params φ.

**Involutive MCMC structure moves.**
- **SUBTREE-REPLACE**: pick a node path b uniformly, replace that subtree with a fresh one from the conditional prior; noise η' drawn by a Gibbs-like conjugate step η' ~ InvGamma(1 + n/2, 1 + Σ(y_i − μ'_i)²/2) using GP posterior mean μ'. Acceptance r_R = [P(η')P(y|k',θ',η') (1/|k'|) P(η|μ,y)] / [P(η)P(y|k,θ,η)(1/|k|) P(η'|μ',y)] — prior terms cancel because the proposal is the conditional prior; |k| = number of nodes (uniform path choice).
- **DETACH–ATTACH** (pair of mutually inverse moves, direction chosen with prob ξ): DETACH replaces subtree at a with its own sub-subtree at (a,b) (dropping a "scaffold"); ATTACH wraps a subtree in a fresh scaffold sampled from the conditional prior. Lets the chain go LINEAR → LINEAR + PERIODIC **while keeping the inferred LINEAR parameters**, which subtree-replace cannot do without resimulating from the root.
- **Pseudo-marginal / multiple-try parameter proposal (Sec. 3.2.3, Prop. 3).** For a structure change, split params into shared S, discarded D, fresh F. Draw U auxiliary candidate parameter sets: shared via log-normal random walk around current values, fresh from the prior; choose one ũ with prob ∝ ω = importance weight P(k', θ'^u, η, y)/q(θ'^u) (the variance-optimal choice); also draw U−1 reverse auxiliaries so the extended move is an involution. Acceptance ratio Eq. (58) includes Σ_u ω terms on both sides. Effect: proposals of a new structure no longer die because of bad fresh parameters. Fig. 4: with U ≥ 5 acceptance ≈ 1 for a strongly favored structure; accepts/sec has diminishing returns — suggest U ∝ dim(θ').
- This is a "multi-particle, single-step AIS" proposal; contrast with single-particle multi-step AIS RJMCMC (Karagiannis & Andrieu).

**Complexity argument for SMC vs MCMC.** With log annealing schedule n/2^i and r rejuvenations per step, per-particle cost Σ r (n/2^i)³ = O(r n³) vs. MCMC with same budget O(r log n · n³). SMC wins when structure is identifiable from n* ≪ n early points.

## 3. Experiments
- **Runtime vs accuracy (8 TSDL datasets; M = 48 particles/chains/threads):** SMC reaches the same SMAPE as MCMC (Saad 2019) with 10×–100× less wall-clock; greedy search (Duvenaud 2013, BIC, depth 10) worse on 5/8 datasets, high variance.
- **M3 monthly (1,428 series), 18-step horizon, best of 10 seeds per method.** AutoGP lowest overall: SMAPE 10.69 (next AutoTheta 12.46, AutoETS 12.47, AutoARIMA 13.23, LSTM 14.65, TFT 14.74), MASE 0.57 (next 0.70), MSIS 4.77 (next AutoETS 5.09). Significant (Mann-Whitney, Bonferroni over 54 tests) for SMAPE horizons 5–18, MASE 1–18, MSIS 5–18.
- AutoGP settings: linear annealing schedule adding 5% data per step, M = 48, resample at ESS = 24, N_rejuv = 100. Runtime 34.68 ± 8.59 s per series (between AutoETS 21 s and LSTM 37 s; TFT 415 s), n2d-standard-48.
- ESS traces (App. B): several rejuvenations between resampling events; degeneracy not severe.

## 4. Limitations
- Authors: dense GP O(n³); runtime varies with structure complexity; best-of-10-seeds evaluation favors stochastic methods; optimal U not theoretically characterized.
- Mine: uses only 48 particles — fine because rejuvenation (100 MCMC steps) does the heavy lifting; the "learning" is pure inference (no amortization), so per-dataset cost is fixed; moves are generic tree edits with uniform node choice — no data-driven proposals; "best of 10 seeds" isn't a calibration claim.

## 5. Ideas to steal
1. **Algorithm 1 is our slow mind almost verbatim.** Replace GP marginal likelihood with our exact HSMM forward-filter log-likelihood (per hypothesis), "data batches" = attempts (deaths) or chunks of ticks, rejuvenation = (FFBS + conjugate Gibbs) for parameters + structure moves for the boss program. Use their defaults as a starting point: M ≈ 48–64 particles, resample at ESS < M/2, many rejuvenation steps per batch rather than many particles.
2. **Structure moves for boss programs as involutive pairs:**
   - SUBTREE-REPLACE ↔ "resample one move's primitive + timing from the conditional prior."
   - DETACH–ATTACH ↔ **split/merge of moves** and **add/remove a law** while *keeping* existing parameters (e.g., move k → moves k1, k2 sharing k's primitive). Write each as (proposal, involution, acceptance) so correctness is checkable.
3. **Multiple-try parameter proposals for new structure (U auxiliary candidates, choose ∝ importance weight).** For a split move, draw U = 5–10 candidate timing parameter sets for the new move (shared ones by log-normal random walk around current, new ones from prior/Θ or from the intuition net), select ∝ weight, include the reverse auxiliaries. This is the single most useful trick for making split/merge and "new move" proposals accepted. Vectorizes trivially (U forward filters in parallel).
4. **Data annealing offline:** after the fight, re-run SMC over nested subsets of the log (attempt 1, 1–2, …) to rebuild the posterior from scratch when particles have degenerated; a principled "sleep on it" between bosses.
5. **Complexity argument for the report:** SMC over attempts with rejuvenation costs O(r·Σ_j T_j) per particle vs. MCMC re-run from scratch each death at O(r·j·T) — cite their accounting.
6. **Metrics:** ESS traces with resampling markers (App. B style figure) and runtime-vs-accuracy curves (Fig. 5 style: x = wall-clock, y = hit-time prediction error) for prior-proposal SMC vs. q_φ-SMC vs. MCMC — a natural E6 figure.
7. **Conjugate-within-proposal trick:** like η' ~ InvGamma given μ', sample conjugate params (cue confusion Dirichlet, transition rows) exactly *inside* a structure proposal and account for it in the ratio — reduces rejection.

## 6. Structures and models of thought
- **Program-like hypotheses with a grammar prior (PCFG)**: compositional, interpretable models whose size adapts to data — the same stance as boss programs over a primitive library with laws.
- **Weighted ensemble of hypotheses** (Fig. 1 multimodal posterior: additive vs multiplicative seasonality) — the agent should keep competing boss explanations (feint vs. delayed attack) rather than commit.
- **Online structure revision as evidence arrives** (data annealing): the cognitive picture of theory change — structure grows when evidence demands, and rejuvenation can also *shrink* models (unlike greedy search).
- **Reversible, local edits that preserve learned parts** (detach/attach): "revise the theory, keep what you know" — the mechanism behind fast transfer when a known boss gains one new move (T2/T3).
