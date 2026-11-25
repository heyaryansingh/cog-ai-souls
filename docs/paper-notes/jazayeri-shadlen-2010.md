# Jazayeri & Shadlen 2010: Temporal context calibrates interval timing

## 1. Citation and access
- Mehrdad Jazayeri, Michael N. Shadlen. "Temporal context calibrates interval timing." *Nature Neuroscience* 13(8):1020–1026, 2010. doi:10.1038/nn.2590. PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC2916084 (it sat behind a reCAPTCHA wall for my fetcher). Read instead: the author PDF at https://shadlenlab.columbia.edu/publications/publications/mike/JazayeriNatNeurosci2010.pdf.
- **What I read.** The main text, methods and figure legends of the author PDF, through a text-extraction tool. Equations came out garbled in extraction. I rebuilt them from the prose and checked them against the paper's stated numbers: the MLE constant is 0.99 at w_m = 0.1 and 0.92 at w_m = 0.3. I did not read the supplementary material. Fitted Weber fractions appear only in a figure (Fig. 6 inset), so I give no exact values.

## 2. Core idea and exact method
**Task: Ready-Set-Go interval reproduction.**
- Two 100 ms flashes ("ready", "set") are separated by a sample interval t_s. The subject presses a key t_p after "set" to reproduce t_s.
- Spatial positions vary from trial to trial, so subjects cannot use a spatial strategy.
- **Priors.** Three discrete uniform priors, each with 11 values, that partially overlap:
  - Short: 494–847 ms
  - Intermediate: 671–1023 ms
  - Long: 847–1200 ms
- **Feedback.** A response counts as correct if it falls within a window of ±k·t_s. k is set by a 1-up/1-down staircase (step 0.015), so about 52% of trials are rewarded. Early or late responses get no feedback.
- **Subjects.** Six people. About 500 trials per session. Learning took about 500 trials, judged complete when bias and variance changed by less than 10% between sessions.

**Observer model.** Three stages, with scalar (Weber) noise at both ends.
- Measurement: t_m ~ N(t_s, (w_m t_s)²), i.e. p(t_m | t_s) = 1/(√(2π) w_m t_s) · exp[−(t_m − t_s)² / (2 w_m² t_s²)].
- Estimator: t_e = f(t_m).
- Production: t_p ~ N(t_e, (w_p t_e)²).
- Prior: uniform on [t_s,min, t_s,max]. Posterior: π(t_s | t_m) ∝ p(t_m | t_s) p(t_s).

**Estimators compared:**
- MLE: f(t_m) = t_m(√(1 + 4w_m²) − 1)/(2w_m²). This is proportional to t_m and ignores the prior.
- MAP: the MLE clipped to the prior's support.
- BLS (posterior mean): f(t_m) = ∫ t_s π(t_s | t_m) dt_s. This gives a sigmoid mapping from t_m to t_e.
- General form: f = argmin_{t_e} ∫ l(t_e, t_s) π(t_s | t_m) dt_s.

**Fitting.**
- The full model is p(t_p | t_s) = ∫ p(t_p | f(t_m)) p(t_m | t_s) dt_m.
- Two free parameters, w_m and w_p, are fit by maximum likelihood across all three priors jointly (Nelder–Mead, trapezoidal integration).

## 3. Experiments and key results
**Behavior:**
1. Production s.d. grows with the mean (the scalar property).
2. t_p regresses toward the prior mean. The bias is stronger for the long prior and for longer intervals.
3. t_p still increases monotonically with t_s.

**Bias–variance analysis.**
- BIAS is the RMS of the per-interval biases. VAR is the mean variance across intervals.
- RMSE² = VAR + BIAS², so contours of equal RMSE are quarter circles in (BIAS, √VAR) space.

**Model comparison:**
- **MLE** misses the prior-dependent bias for every subject.
- **MAP** underestimates bias and overestimates variance for every subject.
- **BLS** fits bias, variance and RMSE for every subject, and beats both MLE and MAP statistically.
- **Noise variants:**
  - Scalar measurement noise beats constant-s.d. measurement noise for all subjects.
  - Scalar production noise beats constant-s.d. production noise for 5 of 6 subjects (AIC).
  - Setting w_m = w_p is rejected (p < 0.03 for one subject, p < 1e−7 for the rest).
- **Generalization check.** Fitting on the short and long priors predicts the intermediate prior equally well.

**Learning speed.**
- The first prior took about one session (~500 trials). One subject needed 4 sessions for the first condition.
- Later priors typically stabilized after one practice session, so re-calibrating to a new prior is faster than learning the first one.

**Compute.** Trivial.

## 4. Limitations
**Reported by the authors:**
- Only six subjects.
- Discrete priors are treated as continuous.
- This is an observer model, not an observer-actor model. The full loss-minimizing actor was not derived.
- The internal prior may not equal the prior the experiment imposed.
- Attention and the reward schedule could shift how prior and likelihood are weighted.
- Neural noise in the estimator is not modeled separately.

**My own observations:**
- The data are asymptotic behavior after ~500+ trials. The paper says nothing about the trial-by-trial dynamics of acquiring the prior, which is the regime our agent lives in (a few deaths).
- The feedback window rewards being near t_s, which is not the same as the asymmetric costs in a souls fight (dodging early is as fatal as dodging late, but the deadline is hard).
- The paper models a single interval, not sequences of intervals or hazard-based anticipation.

## 5. Ideas to steal
1. **Scalar-noise timing likelihood as a selectable agent "perception model".**
   - For a boss wind-up of duration d, the agent's internal measurement of elapsed wind-up time gets noise σ = w_m · d, with w_m ≈ 0.1–0.2 (typical human Weber fraction for timing; the paper's fitted values are in Fig. 6 and roughly in this range).
   - Action execution gets noise σ = w_p · t_e.
   - With both off, you have the "ideal" agent. With both on, you have the "human-like" agent for E4.
2. **Use BLS, not MAP, to pick the dodge time.** This matches the plan's "posterior predictive hit time" rule. Better still, use the general form argmin_t ∫ l(t, t_hit) p(t_hit | data) dt with the game's actual asymmetric i-frame loss: success iff t_hit ∈ [t_dodge + a, t_dodge + a + iframe]. That is the "observer-actor" extension the authors left out. Report BLS, MAP and loss-optimal variants as an E2 ablation.
3. **Bias–variance decomposition as an E4 metric.**
   - For each boss move with variable delay, compute the agent's dodge-time BIAS (RMS across delay values) and VAR.
   - Plot them in the (BIAS, √VAR) plane with RMSE quarter circles, for humans and agent.
   - Prediction: BLS-type agents (and humans) trade bias for lower variance, while an MLE/reactive agent has zero bias and high variance.
4. **The overlapping-priors design gives a clean test of the delayed-attack prediction.**
   - Build boss variants whose delayed-swing durations come from Short / Intermediate / Long uniform sets that share some durations.
   - The same physical delay should produce *different* dodge timings depending on the boss's context distribution: early under the short context, late under the long one.
   - This is the cleanest test of whether the agent (and players) calibrate to temporal context. It turns "delayed attacks as adversarial examples" (§7.4 of the plan) into a quantitative prediction: error ∝ (d − prior mean), with slope growing with d because of scalar noise.
5. **The predicted asymmetry.**
   - Under scalar noise, a *longer-than-usual* delay is pulled toward the mean more strongly, so the agent dodges too early.
   - A shorter-than-usual one also gets pulled toward the mean, but less strongly, so the agent dodges a little late.
   - Design souls-style "delayed" attacks at the long end and measure the death-rate asymmetry.
6. **Capability target.** After enough attempts on a boss with variable delays, the agent's dodge-time bias curve should look sigmoidal against the true delay, like the BLS curve. If it stays flat (it learned only the mean) or stays the identity line (it ignores the prior), something is wrong.
7. **Re-calibration speed for E3 (transfer).** Humans re-calibrate to a new temporal prior faster than they learn the first one. Our hierarchical Θ predicts the same. Measure trials-to-stable bias for boss 1 versus boss 5 of a family.

## 6. Structures and models of thought
- **A three-stage observer (measurement → Bayesian estimator → noisy production)** with *separate* noise sources. The agent's self-model should likewise keep perceptual uncertainty (cue noise) apart from motor uncertainty (execution jitter, i-frame variability). The plan's self-model over rules R is the place for w_p.
- **The loss function is part of cognition.** The choice of estimator (MAP vs BLS vs loss-optimal) is a testable cognitive commitment. The planner's utility already defines a loss, so make the dodge-timing decision explicitly loss-optimal under the posterior predictive.
- **Temporal context as a learned prior over durations.** This maps directly onto the hierarchical LogNormal timing prior Θ per primitive. Scalar noise is consistent with log-space timing models, so the LogNormal choice is cognitively motivated: equal noise in log-time is the scalar property.
