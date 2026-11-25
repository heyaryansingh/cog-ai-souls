# Bayesian Nonparametric Hidden Semi-Markov Models (HDP-HSMM)

## 1. Citation and reading status
- Matthew J. Johnson, Alan S. Willsky. "Bayesian Nonparametric Hidden Semi-Markov Models." *JMLR* 14 (2013) 673–701. arXiv 1203.1365. https://arxiv.org/abs/1203.1365 · https://www.jmlr.org/papers/v14/johnson13a.html · code: `pyhsmm` (github.com/mattjj/pyhsmm).
- **Reading status:** full arXiv PDF and the full JMLR PDF were both processed through a web-fetch extraction tool (I could not download the PDF directly; arxiv.org is blocked to curl in this sandbox). I got the equations for messages, forward sampling, auxiliary variables, the DA sampler, the REDD table and the main synthetic findings. I did not see figures, so mixing claims below are the authors' text, not curves I inspected. Treat equation indices as approximate.

## 2. Core idea and exact method
**Problem.** The (sticky) HDP-HMM learns the number of states but forces geometric dwell times; self-transition stickiness is a crude fix. The HDP-HSMM keeps the HDP prior over transitions, **removes self-transitions**, and gives each state an explicit duration distribution (Poisson, negative binomial, anything with a sampler).

**Generative model (finite / weak-limit form).**
- β | γ ~ Dir(γ/L, …, γ/L) (global state weights; L = truncation level).
- π_i | α, β ~ Dir(αβ_1, …, αβ_L); used rows are π̄_ij = π_ij (1−δ_ij)/(1−π_ii) (self-transitions removed).
- θ_i ~ H (emission params), ϑ_i ~ G (duration params).
- Segment s: z_s ~ π̄_{z_{s−1}}, D_s ~ p(d | ϑ_{z_s}), y over the segment i.i.d. f(θ_{z_s}).
- Factorial extension for disaggregation: ȳ_t = Σ_k y_t^(k) + w_t, each y^(k) an independent HDP-HSMM.

**HSMM backward messages (right-censored).**
- B_T(i) = 1
- B_t(i) = Σ_j B*_t(j) · π̄_ij
- B*_t(i) = Σ_{d=1}^{T−t} B_{t+d}(i) p(D=d | i) p(y_{t+1:t+d} | i) + p(D > T−t | i) p(y_{t+1:T} | i) (censoring term, following Guédon).
- Cost O(T²N + TN²); with durations truncated at d_max, O(T·d_max·N + TN²). (In their experiments durations were *not* truncated; they used changepoint candidates instead.)

**Forward sampling (the HSMM FFBS block move).**
1. x_1 ∝ p(x_1 = k) B*_0(k).
2. D_1 = d ∝ p(D = d | x_1) p(y_{1:d} | x_1) B_d(x_1) / B*_0(x_1).
3. Next label ∝ π̄_{x_1, j} B*_{D_1}(j); repeat until T.
This draws the *whole* label+duration sequence jointly from its exact conditional: the key mixing advantage over single-site samplers.

**Weak-limit Gibbs sampler (the recommended one).** Cycle: (a) block-resample the state sequence by messages + forward sampling; (b) resample emission and duration params given segments (conjugate where possible); (c) resample auxiliary counts; (d) resample π, β.
- **Auxiliary-variable trick for self-transition removal.** Removing π_ii breaks Dirichlet conjugacy (likelihood has (1−π_ii)^{−n_i}). For every segment that leaves state i, draw ρ ~ Geometric(1 − π_ii) on {0,1,…} (the number of "virtual" self-transitions). Then π_i | counts, ρ ~ Dir(αβ_i + Σρ, αβ_j + n_ij for j≠i): conjugate again. π_ii itself ~ Beta(αβ_i, α(1−β_i)) when needed. Authors report this mixes faster than an MH update (faster ACF decay, multivariate PSRF → 1 sooner).
- β updated from table counts as in Teh et al. HDP (Chinese restaurant franchise auxiliary counts), using ρ-augmented counts.

**Direct-assignment (DA) sampler.** π and θ integrated out; resample each x_t one at a time. Each candidate label k is scored by left-transition × right-transition CRF predictive ratios, ((αβ_k + n_{prev,k})/(α(1−β_prev) + n_{prev,·})) etc., × duration predictive of the (possibly merged) segment × block observation predictive. Changing one label can merge/split up to three segments. Ergodic if duration support covers 1…d_max. **Mixes much worse** than weak-limit (sequential single-site updates on strongly correlated sequences); ~50× slower per iteration in their Python.

**Changepoint side information.** Restrict boundaries to candidate changepoints (from first differences); duration prior renormalized over allowed lengths; message cost becomes O(T_c² N + N² T_c). REDD: ~5,000 samples → 100–200 candidates; one component chain resampled in ~0.1 s.

## 3. Experiments
- **Synthetic.** (i) Geometric HDP-HSMM DA ≈ sticky HDP-HMM DA on HMM data; weak-limit mixes much faster than DA. (ii) Data from a 4-state Poisson-HSMM: HDP-HMM keeps high normalized Hamming error and often wrong K; Poisson HDP-HSMM concentrates on true K, low error. (iii) NegBin HDP-HSMM on HMM data with r ~ Unif{1..6}: posterior concentrates on r = 1 (it "discovers" geometricity) with only slight mixing slowdown. 25 chains per experiment.
- **REDD power disaggregation** (5 devices; 18 day-long sequences from 4 houses; 20-s median filter; L = 2× expected modes, e.g. 6 fridge, 20 lighting; 20 samples, every 50 iterations; median accuracy):

| House | EM FHMM | F-HDP-HMM | F-HDP-HSMM |
|---|---|---|---|
| 1 | 46.6 | 69.0 | 82.1 |
| 2 | 50.8 | 70.7 | 84.8 |
| 3 | 33.3 | 67.3 | 81.5 |
| 6 | 55.7 | 61.8 | 77.7 |
| Mean | 47.7 | 67.2 | 81.5 |

Accuracy = 1 − (summed absolute device-estimate error) / (total observed power), per the extraction (the exact normalizing constant may include a factor of 2; check the PDF before quoting). Priors were hand-set (Appendix A).

## 4. Limitations
- Authors: weak-limit needs a fixed L (beam sampling suggested, not tested); self-transition removal costs conjugacy (fixed by ρ); DA mixes poorly; only right-censoring handled; priors hand-set; EM baseline trained on different data.
- Mine: O(T²) messages without truncation are too slow for long logs unless durations are capped; the Gibbs sampler is still a *local* chain over (labels, params) jointly, so label switching and stuck partitions (two true moves merged into one state) are not addressed—no split-merge moves. Emissions are i.i.d. within a segment; our cue model has within-move stage structure (wind-up → active → recovery), which needs an expanded "(move, stage, tick)" state space, i.e., an HSMM-as-HMM embedding, not the plain segment-likelihood formulation. Durations are independent across segments (no "delayed attack depends on previous move" dependence).

## 5. Ideas to steal
1. **Use the weak-limit sampler, not DA, for the slow mind's MCMC kernel.** Concretely, our boss program with K ≤ L moves: β ~ Dir(γ/L), rows Dir(αβ), ρ-augmentation for no self-transitions. Every move except duration/emission params is conjugate → fast NumPy Gibbs.
2. **HSMM FFBS as the rejuvenation move over segmentations.** Our fast filter already computes forward messages per hypothesis; add the backward pass B/B* with `d_max` = max move length (ticks) and censoring for the death/fight-end. Cost per attempt O(T·d_max·K + T·K²) — with T ≈ 1,800 ticks, d_max ≈ 120, K ≈ 8, that is ~2M ops: trivially NumPy-vectorizable (cumulative log-likelihood sums over windows via `np.cumsum`).
3. **Censoring term matters for us:** the player dies mid-move, so the last segment is right-censored. Without the censoring term the posterior over durations is biased short.
4. **Duration families:** NegBin(r, p) with discrete prior on r ∈ {1..6} lets the model infer whether a move timing is geometric (r=1, unpredictable) vs. peaked (large r, "memorize the timing"). This is a direct, interpretable readout for the "delayed attack" experiment (E4): delayed attacks = high-variance or bimodal duration.
5. **Changepoint restriction for speed:** use cue-change ticks (animation id changes) as candidate boundaries → messages over T_c ≪ T. Our cue stream gives these for free.
6. **Diagnostics to report:** normalized Hamming distance of inferred vs. true segmentation over iterations, posterior on K concentrating at truth, PSRF across chains. Same figures style as their Figs. 11–13, on tiny bosses.
7. **Validation:** test our NumPy sampler against `pyhsmm` (the plan already says so) on a 3-state Poisson HSMM; HDP-HMM-vs-HSMM gap on non-geometric durations is a known-sized effect to reproduce.

## 6. Structures and models of thought
- **Explicit-duration latent segments = "move" representation.** A boss move is a segment with a typed duration law; this is exactly what humans memorize ("the third swing is delayed"). Geometric HMMs cannot represent "it's about to happen," so they cannot support anticipation; HSMMs carry a hazard function, which is what Bayesian interval timing (Jazayeri & Shadlen) requires.
- **Hierarchical sharing (HDP)**: global β = a shared repertoire of moves; per-state rows = context-specific transition habits. Maps to "family library + boss-specific transitions" (our CRP over families can be an extra HDP level).
- **Nonparametric growth**: L-truncated Dirichlet lets new moves appear when evidence demands—the computational form of "that's a new attack."
- **Block (whole-sequence) inference vs. local edits**: FFBS is the "re-parse the whole fight" operation; DA is "relabel one moment." The paper's result that block moves mix ~orders better is an argument that the slow mind should re-parse fights wholesale after each death, not patch.
