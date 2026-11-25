# Inference Compilation and Universal Probabilistic Programming

## 1. Citation and how much I read
- Tuan Anh Le, Atılım Güneş Baydin, Frank Wood (Oxford), "Inference Compilation and Universal Probabilistic Programming." AISTATS 2017 (PMLR 54). https://arxiv.org/abs/1610.09900
- **Read:** the full text (v2, about 9k words, including figure and table captions and the references) via the arXiv PDF using a summarising fetch tool. The extraction was noisy in places: one competitor row of Table 1 was garbled, and the figures are not visible.

## 2. Core idea and exact method
- **"Compilation."** Turn a universal probabilistic program into a neural network that, at test time, gives proposals for **sequential importance sampling** (not resampling SMC) over execution traces. Traces are indexed by (address, instance, value).
- **Objective (Eq. 7).** The expected **forward KL**, E_{p(y)} KL(p(x|y) ‖ q(x|y;φ)), which is
  `E_{p(x,y)}[ −log q(x | y; φ) ] + const`.
  This is maximum likelihood of the proposal on traces sampled from the model.
- **Training data.** Replace each `observe` with `sample` and run the program. Minibatches are **generated on the fly and discarded after each Adam step**, so there is no finite training set and no overfitting. A fixed validation set is used only for monitoring.
- **Architecture.**
  - An **LSTM core**. At each sample statement its input concatenates:
    - the observation embedding f_obs(y);
    - an embedding of the previous sampled value;
    - one-hot encodings of the **address**, the **instance**, and the **proposal type**.
  - A per-address sample-embedding layer f_smp and proposal layer f_prop are created **on first encounter** and grown dynamically, so the network adapts to traces of varying structure.
  - Each proposal's parameters come from f_prop(h_t).
- **Captcha configuration:**
  - 2 stacked LSTMs × 512 units;
  - observation CNN: 6 conv layers (3×3; filters 64, 64, 64, 128, 128, 128) + max-pool + 2 linear layers → a 1024-dimensional embedding;
  - **about 20M parameters**;
  - Adam, learning rate 1e-4, batch 128, **about 16M training traces**.
- **Design finding.** Feeding the observation embedding at *every* step versus only the first step gave equal accuracy, but the first-step-only variant took about **3× longer to train**.

## 3. Experiments
- **GMM with unknown K.**
  - The input is a 2-D histogram of the data, embedded by a CNN.
  - Label symmetry is broken by **sorting clusters by distance from the origin** and by predicting cluster parameters rather than per-point assignments.
  - Against SMC with the prior proposal over 1–10,000 particles (50 runs per setting), the compiled proposal gives low-noise estimates at small compute (Fig. 4, qualitative).
  - Also applied to counting and localising objects from PASCAL VOC detections, with a uniform prior on K over {1..5}.
- **Captcha solving.**
  - Generative models for Baidu 2011/2013, eBay, Yahoo, reCaptcha, Wikipedia, and Facebook.
  - **Synthetic test recognition rates:** Baidu 2011 99.8%, Baidu 2013 99.9%, eBay 99.2%, Yahoo 98.4%, reCaptcha 96.4%, Wikipedia 93.6%, Facebook 91.0%.
  - **Speed:** under 100 ms per Captcha, against 500 ms and 7.95 s for segment-and-classify solvers, and "minutes" for Mansinghka et al. 2013 (MCMC in a probabilistic program).
  - **Real data (500 labelled images each): 81% on Wikipedia, 42% on Facebook.** Initial real-data rates were **below 10%**. They rose only after the authors **tuned and broadened the priors** (font size, renderer noise) to close the "synthetic gap."
- **Metrics:** recognition rate and qualitative posterior plots. **No ESS, KL, or log-likelihood numbers are reported.** Hardware and training time are not reported (Torch with a ZeroMQ link to Anglican).

## 4. Failure modes and limitations
- **Reported by the authors:**
  - **Model misspecification, the synthetic-to-real gap.** Real accuracy started below 10% and was fixed by prior engineering. The authors advise that the generative model should err toward being **too broad**.
  - Risk of overfitting to the model.
  - Symmetry breaking is only approximate (clusters at similar distances can swap).
  - The LSTM core is generic and not tied to the program's graph.
  - Observation embeddings are hand-chosen and trained from scratch.
- **Noticed by me:**
  - SIS without resampling or MCMC rejuvenation is fragile for long traces.
  - The Captcha accuracy comes from the network's MAP-like output, which is an "amortised-only" style evaluation. The gap from 93.6% on synthetic to 81% on real Wikipedia data, after tuning, is itself an amortisation gap under distribution shift.
  - No quantitative proposal-quality metric (ESS) is given, so the paper cannot say how close q is to the posterior.

## 5. Ideas to steal
1. **Fresh-sample training.** Generate boss/log pairs on the fly and discard them, so there is no overfitting and the effective dataset is the prior itself. Use a fixed validation set only for monitoring (the expected-NLL gate). If simulation is the bottleneck, use Paige & Wood's regenerated pool.
2. **Address-keyed heads for stochastic structure.** Our boss programs have variable numbers of moves, conditional laws, and family choices. Key proposal heads on a structural "address" (which slot or decision: segmentation boundary, primitive label, timing parameter, family) plus an instance index. Feed the address embedding, the previous choice's embedding, and the log embedding at each step of an **autoregressive** proposal over the program. This is the same structure with a transformer core instead of an LSTM.
3. **Feed the observation summary at every decoding step.** It cost nothing in accuracy and gave a 3× faster training run in their hands, which matters for the laptop budget.
4. **Symmetry breaking for exchangeable latents.** Sort move-primitives or families by a canonical key (for example, first appearance time in the log) before computing the NLL. Otherwise label switching corrupts the cross-entropy targets.
5. **Broad priors for robustness.** Their prior-broadening fix suggests training q_φ on a *wider* boss prior than the evaluation prior (inflate timing variances and transition entropy), as cheap insurance against T3/T4 novelty. This trades a little in-distribution sharpness for OOD robustness, and it is an ablation worth running in E6: q_φ trained on the matched prior versus a 2× broadened prior.
6. **The "particles × runs" figure.** Their 1–10k particles with 50 runs per setting is a good template for our E6 compute axis.
7. **Target size.** Their 20M-parameter net trained on 16M traces is an upper reference. Our 1.5M net on short token logs is about 10× smaller, which is plausible because our observations are low-dimensional event streams, not images.

## 6. Structures and models of thought
- **A compiled recognition model for program-like hypotheses.** A network that proposes *entire program traces*, choice by choice, conditioned on observations, is a model of fast "intuitive theory recognition" for structured hypotheses. It is exactly what our intuition net is.
- **Neural proposals inside exact inference.** Importance weighting means a wrong intuition costs time, not correctness. That is the formal basis of the plan's claim, and it gives the testable prediction for E6: under novelty, the amortised-only answer degrades while the weighted answer only gets slower.
- **Model-based robustness through prior breadth.** Generalisation from a simulator depends on the simulator's prior covering reality. This links to the agent's self-model and to the CRP's "new family" option as an explicit escape hatch.
