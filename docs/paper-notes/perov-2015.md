# Data-driven Sequential Monte Carlo in Probabilistic Programming

## 1. Citation and how much I read
- Yura N. Perov, Tuan Anh Le, Frank Wood, "Data-driven Sequential Monte Carlo in Probabilistic Programming." NIPS 2015 Workshop on Black Box Learning and Inference. https://arxiv.org/abs/1512.04387
- **Read:** the full text, including the appendix with model and code listings, via the arXiv PDF using a summarising fetch tool, plus a second pass for verbatim quotes. The result figures (1–3) were images I could not see, and the text gives **no numeric tables**.
- **Note on names:** author names are from my memory of the arXiv listing; the fetch did not extract them.

## 2. Core idea and exact method
- **Setting.** SMC over execution traces in Anglican. The incremental weight is W_n = p(x_n|x̄_{n−1}) p(y_n|x_n) / q(x_n|x̄_{n−1}). By default probabilistic programming uses the prior as the proposal, which gives high-variance weights when observations are informative.
- **Learned proposal.** q̂(x_n | N_θ(φ(ρ_n))). Here ρ_n = (x_{n−1}, y_{1:n}) is the "environment", φ is a **hand-crafted feature vector**, and N_θ is a discriminative neural network.
- **Objective (Eq. 1).** Expected KL, approximated by a **weighted cross-entropy on SMC samples**:
  `L_n(θ) ≈ −Σ_i w_n^(i) log q̂(x_n^(i) | N_θ(φ(ρ_n)))`.
  The training pairs come from running SMC with 5000 particles, prior proposal, on *training data* (real data, not samples from the model). In practice they use samples from the **smoothing** distribution p(x_n | x_{n−1}, y_{1:N}), even though the target is filtering. This differs from Paige & Wood and Le et al., which train on prior joint samples.
- **Model.** The dependent Dirichlet process mixture of objects (DDPMO) for detection-free tracking in video: about 120 lines of Anglican, a generalised Pólya-urn procedure, and conjugate XRPs (α = 0.1, ρ = 0.32).
- **Features.** Distances to the 3 nearest clusters; 10-bin HSV histograms of 7×7 patches around them; the histogram of the new pixel's patch.
- **Outputs.** 5 softmax probabilities: the 3 nearest clusters, "any other cluster" (p_4 split evenly over the K−3 others), and a new cluster. If K < 3 they fall back to the prior; if K = 3, p_4 = 0.
- **Network.** One hidden layer of 100 tansig units, softmax output, cross-entropy loss. That is tiny, around 10^3–10^4 parameters.
- **Mixing with the prior.** The text mentions decreasing p* "thus increasing the probability of using the prior proposal". This implies a **defensive mixture** with probability p* of the learned proposal and 1−p* of the prior, but p* is never formally defined.

## 3. Experiments
- **Data:** a soccer video with ground truth. The train and test sets are two subsequences of the same video.
- **Compared proposals:** prior, learned NN, and a hand-tuned fixed p_{1:5}.
- **Metrics:** particle log-weight, sequence frame detection accuracy (SFDA), and average tracking accuracy (ATA), each against the number of particles.
- **Findings (verbatim where quoted):**
  - "SMC inference with 10 particles with the data-driven proposal produces results similar to … running SMC with thousands of particles under the prior proposal" (in log-weight).
  - The gains on task metrics are smaller, "especially in respect to the SFDA metric."
  - "**with many particles, SMC with the prior proposal outperforms SMC with the data-driven proposal**."
  - The NN roughly matches the hand-tuned proposal; the hand-tuned one is better on test SFDA. Decreasing p* did not improve SFDA at 100+ particles.
  - The authors conclude that fast discovery of high-probability regions does not guarantee good task metrics.
- **Compute:** the number of training pairs, training time, and hardware are not reported. ESS is not reported.

## 4. Failure modes and limitations
- **Reported by the authors:** the hand-engineered features carry most of the benefit (the spatial factor dominates); the learned proposal loses to the prior at large particle counts; log-weight gains do not translate fully into task metrics.
- **Noticed by me:**
  - Training on SMC output from real data makes the proposal inherit that SMC run's biases. With 5000 prior-proposal particles the "teacher" may itself be poor.
  - Using smoothing samples to train a filtering proposal mismatches the target. The proposal sees "future-informed" labels it cannot reproduce online.
  - The appendix code adds the log-likelihood of the sampled cluster as a weight, and it is unclear whether the q correction is applied correctly. An incorrect weight would bias results, which is a real risk in any hand implementation.
  - The test set is in-distribution (same video), so this is no OOD test.

## 5. Ideas to steal
1. **Always plot accuracy against particle count, over the full range.** Their key finding was a *crossover*: learned proposals win at few particles and lose at many. E6 must show q_φ, prior, and mixture proposals across 1…10^3 particles, and report where the crossover happens under novelty (T3/T4). A crossover moving left under novelty is a crisp, quantitative signature of the amortisation gap.
2. **Defensive mixture proposal.** Use q_mix = p*·q_φ + (1−p*)·p_prior, with p* possibly set adaptively from the net's own confidence or from the metareasoning gate. This bounds the importance weights (w ≤ p/((1−p*)p_prior)) and fixes the many-particle regime where pure q_φ underperforms. This is cheap insurance for the T3/T4 novelty conditions.
3. **Report both inference-quality and task metrics.** Their log-weight versus SFDA gap maps to ours: log Ẑ / ESS (inference) against dodge success or hit-time prediction error (task). Plan E6 to show both, since they may disagree.
4. **A "wake-phase" training source.** Training on weighted SMC samples from *real* logs (Eq. 1) is the natural form of our planned `train_wake.py`: the importance-weighted cross-entropy −Σ w_i log q_φ(B_i | log). Use it as a fine-tuning objective on logs from actual fights, but on filtering rather than smoothing targets to avoid the mismatch above.
5. **Unit-test the weight computation** against a brute-force posterior on the tiny boss (already in the plan). This paper shows how easy it is to get the q correction wrong.

## 6. Structures and models of thought
- **Learned proposals over discrete structural choices** (which cluster or object a pixel belongs to, or a new one) under a CRP/DP prior. This matches our **family CRP** and segmentation choices: the intuition net should output "existing family k / new family" probabilities, with a "new" option always kept.
- **Feature-based bottom-up cues.** A small discriminative net over well-chosen features was as good as a hand-tuned proposal. For a laptop-scale agent, good tokenisation of the fight log (cue onset, time since last move, hit or no-hit) may matter more than network size.
