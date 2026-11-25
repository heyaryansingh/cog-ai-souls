# Körding & Wolpert 2004: Bayesian integration in sensorimotor learning

## 1. Citation and access
- Konrad P. Körding, Daniel M. Wolpert. "Bayesian integration in sensorimotor learning." *Nature* 427:244–247, 2004. doi:10.1038/nature02169.
- The Nature version is paywalled. I read an open copy of the published PDF at https://jhanley.biostat.mcgill.ca/bios602/Bayes/nature02169.pdf, through a text-extraction tool. That covered the main text and methods. Equations were garbled in extraction and are restated in standard notation from the prose. I did not read the online supplement.
- Note: a search result labelled as this paper on the Wolpert lab site (KorKuWol04.pdf) is a different paper, Körding, Ku & Wolpert 2004, *J Neurophysiol*, "Bayesian integration in force estimation." I summarize it briefly at the end because it is relevant.

## 2. Core idea and exact method
**Task.**
- Subjects reach to a target with a hidden finger. A cursor shows the finger position, laterally shifted by x_true.
- Each trial's shift is drawn from a prior N(1 cm, 0.5² cm²).
- Midway (10 cm), feedback flashes for 100 ms, in one of four blur conditions with relative frequency 3:1:1:1:
  - σ0: a clear sphere;
  - σM: a cloud of 25 spheres with SD 1 cm;
  - σL: a cloud with SD 2 cm;
  - σ∞: no feedback.
- Endpoint feedback is shown only on σ0 trials.
- 10 subjects, 2000 trials each. The last 1000 were analyzed. Subjects were unaware of the shift.

**Three models:**
1. **Full compensation.** Use the sensed shift directly. Mean deviation is 0 at every blur level; blur changes only variance. MSE = σ²_sensed.
2. **Bayesian.** x̂ = [σ²_s/(σ²_s + σ²_p)]·μ_p + [σ²_p/(σ²_s + σ²_p)]·x_sensed. MSE = σ²_s σ²_p/(σ²_s + σ²_p), which is always lower than model 1. The slope of deviation against x_true rises with blur.
3. **Mapping.** A fixed learned correction, learnable only on σ0 trials. The slope is the same at every blur level.

**Recovering the prior from behavior.**
- d log p(x)/dx at x̂ ≈ (x̂ − x_sensed)/σ²_s. Integrate this numerically to get the subject's implied prior.
- This uses the approximation x_sensed ≈ x_true.

## 3. Results
- **No feedback (σ∞).** Subjects compensate by 0.97 ± 0.06 cm, matching the 1 cm prior mean.
- **Slope against true shift.** The slope increases with blur: repeated-measures ANOVA F(3,27) = 82.7, p < 0.001, and every adjacent pair of blur levels differs. This rules out models 1 and 3.
- **Offset.** Slope plus offset ≈ 0, as model 2 predicts (no deviation at the prior mean).
- **Inferred sensory SDs:** σ0 = 0.36 ± 0.04 cm, σM = 0.67 ± 0.1 cm, σL = 0.8 ± 0.1 cm.
- **Inferred priors.** The priors recovered from behavior resemble the true Gaussian for each subject.
- **Bimodal prior experiment.**
  - 6 new subjects, 4000 trials over 2 days.
  - The prior was a mixture of two Gaussians 4 cm apart, each with SD 0.5 cm. Feedback was always blurred (SD 4 cm), with endpoint feedback on every trial.
  - Behavior was nonlinear, as a Bayesian observer with a bimodal prior predicts. The fitted mode separation was 4.8 ± 0.8 cm.
- **Learning time.** The prior is learned within ~1000 trials. Only the post-training half was analyzed.

**Companion paper (Körding, Ku & Wolpert 2004, *J Neurophysiol*, force estimation).**
- 11 subjects; the prior over force was N(2 N, 0.5 N) or N(2 N, 1 N).
- Slopes were steeper with the narrow prior: day-2 slopes ranged 0.16–0.89, and the groups differed at p < 0.04.
- C_feedback was 0.81 N, which predicted a slope ratio of 0.55; the observed ratio was 0.60.
- When the prior switched from wide to narrow, slopes adapted within a day.
- Subjects also showed an unexplained systematic underestimation of force; the authors suggest an effort cost.

## 4. Limitations
**Reported by the authors:**
- Sensory uncertainty is not measured independently. It is inferred under the Bayesian assumption, so the argument is partly circular.
- The mapping model is excluded only under this specific feedback design.
- Prior inference assumes x_sensed ≈ x_true and zero-mean motor noise.
- Only one task was tested.

**My own observations:**
- The paper studies asymptotic behavior after ~1000 trials, not few-shot learning.
- It is a single static Gaussian variable, not sequences.
- The loss is quadratic. The real loss in a dodge task is a window (hit or miss), which changes the optimal action when the posterior is skewed or multimodal. The bimodal experiment is the key one for us: with a multimodal posterior, the mean is a bad action.

## 5. Ideas to steal
1. **The "slope against uncertainty" signature as a human-likeness test (E4/E5).**
   - Manipulate *cue informativeness* (already in the plan, §2) as the analog of blur.
   - Plot the slope of the agent's dodge-time error against (true hit time − prior mean hit time), at each informativeness level.
   - Bayesian prediction: the slope rises as the cue becomes less informative. A reactive/mapping agent has a constant slope.
   - This is a simple, one-figure test that the agent integrates its prior with the cue in the right proportion. Run it on humans in the pygame viewer too.
2. **Recovering the prior from behavior (the "inferred prior" method).**
   - From the agent's dodge choices under blurred cues, recover its implied prior over hit times and compare it to the true boss distribution and to the posterior the agent reports.
   - This is a behavioral calibration check that does not trust the agent's internals.
   - The same method applied to humans gives each player's implied prior after N attempts. Compare the two.
3. **The bimodal prior experiment, adapted as a boss design.**
   - Build a boss whose move has a bimodal delay (fast swing OR delayed swing, like a souls "mix-up").
   - With a weak cue, a BLS/point-estimate agent dodges in the middle and dies to both. A loss-aware agent picks one mode, or picks an action that covers both (step back, block).
   - This is a sharp ablation for "dodge from the posterior predictive, not a point estimate" (plan §1.1 row 2). Predict that the MAP (N = 1) ablation fails exactly here.
4. **Use MSE reduction as a metric.** Bayesian MSE = σ²_s σ²_p/(σ²_s + σ²_p), against full-compensation MSE = σ²_s. This gives a closed-form expected benefit of having a prior at each cue-noise level. Plot the agent's actual timing MSE against this bound as a function of attempts; it doubles as an "induction speed" readout.
5. **Prior switching (force paper).** After the boss changes its delay distribution (phase 2 or a T2 variant), track how many attempts it takes the agent's slope to adapt. Humans need on the order of one session. The agent should need far fewer attempts because it does explicit inference; that is a selling point.

## 6. Structures and models of thought
- **Precision-weighted cue combination** is the atomic operation of the fast filter. The HSMM forward pass already does this exactly. This paper is the behavioral evidence that people do it too, so it licenses calling the filter "human-like" at the computational level.
- **The prior is learned from the statistics of the task** and is task-specific (switching experiments). This supports per-boss posteriors plus a hierarchical prior Θ across bosses.
- **Nonlinear estimation under multimodal priors** argues for keeping full posterior predictive distributions (SMC particles, mixture over hypotheses) rather than Gaussian summaries anywhere in the action pathway. The reflex's belief features should include multimodality indicators (for example, hit-time quantiles, or the probability mass in each of two windows), not just a mean and variance.
