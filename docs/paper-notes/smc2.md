# SMC²: an efficient algorithm for sequential analysis of state-space models

## 1. Citation and reading status
- Nicolas Chopin, Pierre E. Jacob, Omiros Papaspiliopoulos. "SMC²: an efficient algorithm for sequential analysis of state space models." *Journal of the Royal Statistical Society: Series B* 75(3): 397–426, 2013. arXiv 1101.1528. https://arxiv.org/abs/1101.1528
- **Reading status:** full arXiv PDF processed through a web-fetch extraction tool (direct download blocked). I got the algorithm, theory statements, N_x adaptation, complexity, comparisons and both experiments. I did not see figures or the discussion contributions printed with the JRSS-B paper.

## 2. Core idea and exact method
**Problem.** Sequential inference of p(θ, x_{1:t} | y_{1:t}) and evidence p(y_{1:t}) in state-space models when p(y_t | y_{1:t−1}, θ) is intractable. IBIS (Chopin 2002: SMC over θ with resample-move) needs those increments exactly.

**Algorithm.** N_θ θ-particles from the prior; each carries its own particle filter with N_x x-particles.
- At each t: advance every PF one step; θ-weight ×= PF likelihood-increment estimate p̂(y_t | y_{1:t−1}, θ) = mean of PF incremental weights.
- If ESS(θ-weights) < threshold (e.g., N_θ/2): resample θ-particles (with their PFs) and apply a **PMMH move**: propose θ̃ (independent Gaussian fitted to current θ-particle mean/cov in experiments; RW scaled by particle covariance also possible), run a fresh PF for θ̃ over y_{1:t}, accept with ratio p(θ̃) p̂(y_{1:t}|θ̃) q(θ|θ̃) / [p(θ) p̂(y_{1:t}|θ) q(θ̃|θ)].
- **Exactness**: on the extended space (θ, all PF random variables), the θ-marginal is exactly p(θ | y_{1:t}) for **any** N_x (unbiased likelihood estimates, pseudo-marginal/PMCMC argument, Andrieu et al. 2010). Prop. 1: equivalent mixture form where one PF lineage is distributed as the exact smoother — enables conditional SMC (particle Gibbs) and trajectory recovery.
- Outputs: filtering/predictive averages over θ-particles; **evidence p̂(y_{1:t}) = Π_s Σ_m w̄_m p̂(y_s | ·, θ_m)** at near-zero cost → sequential model choice; Rao-Blackwellized estimates averaging all N_x lineages.

**Adapting N_x.** PMMH acceptance degrades as t grows unless N_x ∝ t. When acceptance < threshold (e.g., 20%), double N_x via either (1) **exchange step**: generate a new PF with larger N_x for each θ-particle, reweight by p̂_new / p̂_old (generalized IS with near-optimal backward kernel; valid for any N_x), or (2) **conditional SMC** regenerating PFs conditioned on a retained trajectory (weights unchanged; needs stored histories). N_x may differ per θ-particle.

**Complexity and stability.** Per-step cost ∝ N_θ N_x; memory O(N_θ N_x) if histories discarded (O(t N_θ N_x) naive). Under strong assumptions: with N_x growing linearly in time since last resample, ESS stays bounded below; resampling times grow geometrically; cumulative cost O(N_θ t²) vs IBIS O(N_θ t). Not truly online (cost per step grows with t via rejuvenation).

**Relation to alternatives.** N_x → ∞ recovers IBIS; N_x = 1 a poor importance sampler. Kitagawa's self-organizing PF (θ in the state, never moved) degenerates; Liu–West (shrink + jitter) is biased; Storvik / Fearnhead / particle learning use sufficient statistics but still degenerate and need special structure. Fulop & Li (2011) independently similar.

## 3. Experiments
- **Lévy-driven stochastic volatility, synthetic** (T = 1000; N_θ = 1000, N_x from 100, grows when acceptance < 20%; N_x reaches low thousands near a volatility jump at t ≈ 400). Five runs consistent. SOPF with 10⁵ particles collapses to one distinct θ by t ≈ 700. Liu–West with 2×10⁵ particles (h = 0.1) at comparable CPU: biased posterior for ξ and biased log-evidence, discontinuity at the jump. Reference: PMCMC N_x = 500, 10⁵ iterations. Runtime 1.5–7 h per run, single core, plain Python (2008 desktop).
- **S&P 500** 753 daily returns (2005–07): N_θ = 2000; compares 1-factor, 2-factor, 2-factor+leverage via sequential log-evidence; evidence shifts after the jump ≈ t = 550.
- **Athletics records** (women's 3000 m, two best annual times 1976–2010): GEV observations, location as 2nd-order random walk; N_θ = 1000, N_x = 250, independent Gaussian proposal, 30–70 s per run; P(1993 record unusual | beat previous record) ≈ 9.4×10⁻⁴ (sd 3.3×10⁻⁴ over 10 runs).

## 4. Limitations
- Authors: not online (rejuvenation cost grows with t); stability results under restrictive assumptions; memory for smoothing/CSMC; adaptive N_x fiddly; limited model set; tempering variant loses guarantees; low-dimensional (Gibbs-type) move kernels not explored.
- Mine: the independent-Gaussian θ proposal works for 3–5 continuous parameters but not for discrete/transdimensional θ (our boss programs); PMMH acceptance collapses in high dimension; for our model the inner filter is **exact**, so SMC² reduces to IBIS — the PF part is only needed if inner state space becomes too large.

## 5. Ideas to steal
1. **Our design is already "IBIS with exact inner filters" (RBPF over static params).** SMC² is the precise fallback if the inner (move, stage, tick) state blows up (e.g., laws conditioned on distance/HP/last-2 moves make the hidden state large): replace the exact forward pass with an N_x-particle filter per hypothesis and keep correctness via the pseudo-marginal argument. Cite this as the theoretical license for a "sloppy inner filter."
2. **Resample-move schedule driven by ESS < N/2** and **move acceptance-rate monitoring**: when rejuvenation acceptance drops below ~20%, increase inner-filter accuracy (N_x) or switch proposal. A concrete adaptive rule for the metareasoning gate's "spend more compute" decision.
3. **Independent proposal fitted to the current particle cloud** for continuous boss parameters (log timing means/variances, speed factor): fit a Gaussian (or mixture) to the weighted particles, propose independently, MH-correct. Cheap in NumPy, often mixes far better than random walk. The intuition net's q_φ is a learned generalization of exactly this.
4. **Sequential model evidence for free**: track log p̂(log_{1:t} | family f) across attempts to do family recognition (E3: "variant of family 3 within one attempt") and to detect rule change (T4) as an evidence shift — just like the S&P model comparison after the jump.
5. **Exchange step**: when moving from a cheap approximate likelihood (e.g., coarse duration bins) to the exact one, reweight particles by p̂_new/p̂_old instead of restarting — useful for a coarse-to-fine filter.
6. **Report a degeneracy comparison**: our resample-move vs. a "Liu–West style jitter" baseline and a "no-move" SOPF baseline on tiny bosses vs. exact enumeration — shows why rejuvenation matters (their Fig.: SOPF collapses, L&W biased).

## 6. Structures and models of thought
- **Nested inference (hypotheses over models, each running its own filter over states)** = two timescales of belief: slow beliefs about *what the boss is* and fast beliefs about *what it is doing now*. This is the cognitive split our architecture claims.
- **Exact-approximate principle**: noisy, cheap internal simulation can still support unbiased learning if used inside a correct weighting scheme — a computational-level justification for bounded, sample-based reasoning (resource-rational).
- **Evidence as a running scalar** for model comparison is a natural "surprise at the level of theories," feeding the family CRP and rule-change detection.
- **Adaptive compute allocation** (grow N_x when acceptance drops) is a primitive form of metareasoning.
