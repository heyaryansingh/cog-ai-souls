# Bayesian Surprise Predicts Human Event Segmentation in Story Listening

## 1. Citation and reading status
- Manoj Kumar, Ariel Goldstein, Sebastian Michelmann, Jeffrey M. Zacks, Uri Hasson, Kenneth A. Norman. "Bayesian Surprise Predicts Human Event Segmentation in Story Listening." *Cognitive Science* 47(10): e13343, 2023. doi:10.1111/cogs.13343; PMID 37867379; PMCID PMC11654724 (open access); preprint PsyArXiv qd2ra; code github.com/manojneuro/bayesiansurprise; data zenodo 10.5281/zenodo.8404102. Hasson lab page: https://hassonlab.princeton.edu/publications/bayesian-surprise-predicts-human-event-segmentation-story-listening
- Note: arXiv 2210.05710 (suggested in the task) is a **different paper** (Basgol, Ayhan & Ugur, predictive event segmentation with point-light displays).
- **Reading status: PARTIAL — I could not read the journal full text.** Wiley, PMC (captcha), Europe PMC (server error / rate limit), OSF (robots) and the signed OSF PDF link all failed. What I did read: the journal abstract (Europe PMC metadata), the Princeton compmem PDF of the earlier CogSci 2022 version ("Event Segmentation in Story Listening Using Deep Language Models", same authors), extracted in fragmentary slide-like form, and an auto-generated description of the code repo. Numbers below come from the 2022 version and may differ from the journal paper. **Verify before citing.**

## 2. Core idea and method (as far as I could verify)
- **Hypothesis (event segmentation theory, EST):** event boundaries occur at transient increases in prediction error. Question: *which* operationalization of prediction error?
- **Stimuli:** three spoken stories — *Monkey in the Middle* (~30 min), *Pieman* (~7.5 min), *The Tunnel Under the World* (~25 min).
- **Human data:** listeners pressed a button at perceived event boundaries; converted to per-word button-press proportions (also a kernel-density-smoothed version in the data release). The code description mentions 205 participants for *Monkey*; N for others not verified.
- **Model:** GPT-2-XL (48 layers, 1.5B params, 1,024-token context, 1,600-d embeddings); GPT-2 subword tokens aligned to transcript words by longest common subsequence.
- **Measures per word t:**
  - *Surprisal*: −log P_{t−1}(w_t) (how unexpected the actual word was).
  - *Entropy*: H(P_t) of next-word distribution (uncertainty).
  - *Bayesian surprise*: KL(P_t ‖ P_{t−1}) between successive next-word predictive distributions — how much the model's *expectations changed* after hearing w_t.
  - Embedding cosine distance between successive final-layer states.
  - *Transient* versions: each measure minus a running average up to t−1 (a "drift"/decay parameter 0.03–0.10 appears in robustness tables).
- **Analysis:** LASSO regression of boundary proportions on the measures, leave-one-story-out cross-validation; null distribution by circularly shifting the predictors (±4,000 words); results reported as correlation and percentile vs. null; window-size (50–200) and drift robustness checks; AIC model selection in code.

## 3. Results (2022 version; approximate)
- **Transient Bayesian surprise** best predicted boundaries: held-out r ≈ 0.06–0.12, ~95th–99.8th percentile of the null across window/drift settings.
- Non-transient Bayesian surprise weaker (r ≈ 0.05–0.11, ~86th–97.5th pct).
- Surprisal, entropy and embedding distance "performed poorly." Journal abstract: boundaries track transient increases in Bayesian surprise **but not surprisal**.

## 4. Limitations
- Authors (abstract): operational definitions of prediction error differ meaningfully; results from one LM family.
- Mine: effect sizes are small (r ~ 0.1); word-level LM "beliefs" are a proxy for listeners' situation models; KL between successive next-word distributions conflates lexical and event-level updates; three stories only; button press lag must be handled; correlational.

## 5. Ideas to steal
1. **Segment on belief change, not on surprisal.** In our fast filter, compute per tick both surprisal s_t = −log p(cue_t | past) and **Bayesian surprise** BS_t = KL(posterior_t ‖ posterior_{t−1}) over (a) the hypothesis weights (boss-program level) and (b) the predictive next-move distribution. Use *transient* BS (BS_t minus an exponential running mean, decay ≈ 0.05) as the trigger for "new move / rejuvenate now," instead of — or alongside — ESS collapse. Cheap: weights are a length-N vector; KL is O(N).
2. **Distinguish noise from model failure.** Surprisal fires on any unlikely cue (e.g., cue noise in confusable pairs); Bayesian surprise fires only when the cue *changes what the agent believes*. That is exactly the distinction between "unlucky observation" and "my boss model is wrong." Make this an explicit ablation in E5: gate triggered by surprisal vs. by Bayesian surprise vs. by ESS. Predict BS gives fewer false rejuvenations at the same detection rate.
3. **Human comparison design (E4/E5):** have human players press a key when they think "a new attack started" or "that was new" while watching boss replays; regress their presses on agent surprisal vs. BS vs. entropy with a circular-shift null and leave-one-boss-out CV — a direct port of this paper's analysis.
4. **Report the measure as a curve, with the circular-shift null**, which is cheap and robust for time series.

## 6. Structures and models of thought
- **Event segmentation by prediction-error-driven updating**: supports surprise-gated rejuvenation and treats "move boundary" detection as a byproduct of inference rather than a separate detector.
- **Bayesian surprise = information gain about latent causes**: links segmentation to the planner's information-gain term; one quantity serves both perception (segment) and metacognition (when to think / rejuvenate).
- **Transient vs. absolute signals**: humans respond to *changes relative to recent baseline*; our gate features should include deviations from running averages, not just raw entropy.
