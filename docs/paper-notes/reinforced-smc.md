# Reinforced Sequential Monte Carlo for Amortised Sampling

## 1. Citation and how much I read
- Sanghyeok Choi, Sarthak Mittal, Víctor Elvira, Jinkyoo Park, and a fifth author whose name the fetch tool extracted as "Esmeralda S. Whitammer" (Edinburgh, CIFAR Fellow). **Verify this name before citing**: I could not confirm it independently, and it may be a garbled extraction.
- Proceedings of ICML 2026 (PMLR 306). arXiv v3, 31 Jul 2026. https://arxiv.org/abs/2510.11711 (code: "ReinforcedSMC" on GitHub).
- **Read: partial.** I read the main text (§1–§5), the references, Appendix A, and the start of Appendix B via the arXiv HTML and PDF with a summarising fetch tool. The text cut off in Appendix B.1 both times. **I did not read Appendices C–I**: implementation details, network sizes, K, thresholds, compute, the ablations in G, alanine dipeptide in H, and the discrete-space results in I.

## 2. Core idea and exact method
- **Problem.** Sample from an unnormalised density π(x) ∝ R(x). The sampler is a latent Markov chain x_0→…→x_N with learned forward kernels p→_θ and a fixed backward kernel p←. The main instance is a diffusion sampler: Euler–Maruyama steps N(y + u_θ Δt, σ²Δt I).
- **Unifying "dictionary."** Hierarchical VI = MaxEnt RL = SMC/AIS:
  - the policy is the SMC proposal;
  - the learned flow functions F_n^φ are the SMC intermediate targets.
- **Why off-policy.** Reverse KL, KL(p→_θ ‖ π p←), is the "typical choice". It is **mode-seeking**, and off-policy training under it needs importance-sampling corrections that have high variance. Trajectory-balance losses avoid both problems:
  - **TB** (Eq. 8): (log Z_θ + Σ log p→_θ − log R(x) − Σ log p←)²;
  - **SubTB** (Eq. 7): the same squared log-ratio over sub-trajectories, using F_m and F_n.
  Both can be trained on trajectories from *any* behaviour distribution.
- **Assignment.** TB trains the policy θ and SubTB trains the flows φ. The authors say "other combinations are suboptimal or unstable" (§G.2, not read).
- **SMC as behaviour policy (§3.2).** Each batch mixes two kinds of trajectories:
  - on-policy rollouts;
  - off-policy rollouts, made by running SMC that targets the current flows with the current policy as proposal, then generating full trajectories from the SMC output x_N with the backward kernel.
  SMC's resampling finds modes the policy alone misses, and those trajectories teach the policy.
- **Importance-weighted replay buffer (IW-Buf, §3.3).** Each batch m gets a normalising-constant estimate Ẑ^m:
  - on-policy: the mean of the final weights;
  - SMC: a product of per-segment means between resampling steps.
  Each stored sample carries weight Ẑ^m · W_N^{m,k}, and training samples are drawn from the buffer in proportion to that weight. This makes the buffer approximately target-distributed across batches of different quality.
- **Adaptive weight tempering (§3.4, Eq. 15).** Replace w ↦ w^λ, with λ* = the largest λ ∈ [0,1] such that ESS(w^λ) ≥ γK, found by binary search. λ* should rise towards 1 as the sampler improves. This stops early, bad proposals from producing degenerate SMC teachers.
- **ESS** (Eq. 14): (Σw)²/Σw², which ranges from 1 to K. Resampling happens when ESS < κ. An AIS-only variant (no resampling) is also tested.
- **Evaluation:** metrics are reported **on the amortised sampler alone, without SMC at test time**, averaged over 5 runs.

## 3. Experiments (main text)
- **Gradient-free targets (Table 1):** GMM40 (d=2, 5), Funnel (d=10), ManyWell (d=32). Baselines: SMC-RWM, DDS, LV, on-policy TB. Metrics: ELBO (higher is better), EUBO (lower is better), Sinkhorn.
  - On-policy methods show **severe mode collapse**.
  - GMM40 d=2 EUBO: TB 273.10 → TB+IW-Buf 0.88.
  - ManyWell: EUBO 252.37 → 166.30; ELBO 160.69 → 162.84.
  - GMM40 d=5 ELBO: −5.19 → −4.48.
- **Gradient-based targets (Table 2):** Funnel-10, Robot4-10, GMM40-50, MoS-50, ManyWell-64. Baselines: SMC-HMC(-ESS), PIS, DDS, LV, CMCD-KL/LV, SCLD.
  - Robot4 Sinkhorn: on-policy TB 1.72; SMC+IW-Buf **0.39**; SMC *without* the buffer 64.48 ± 103.61, which is very unstable.
  - On GMM40-50, plain SMC-HMC-ESS has the best Sinkhorn (1477.04); the method does not win everywhere.
  - Some configurations are marked × (training diverged).
- **Ablations (§4.3):**
  - Learned interpolation plus learned correction of the flows is best on ManyWell (ELBO 162.81, EUBO 166.30); fixed linear or cosine annealing schedules are unstable on GMM40-50.
  - Buffer prioritisation (Table 4): IW-Buf beats uniform, reward-prioritised, and loss-prioritised buffers. GMM40-5 EUBO: on-policy 3156.7 → IW-Buf 1183.3.
- **Compute and architecture:** not in the parts I read (they are in the appendices).

## 4. Failure modes and limitations
- **Reported by the authors:** the benchmarks are mostly low-dimensional; some loss combinations and fixed schedules diverge; SMC without the IW buffer is highly unstable; there is no dedicated limitations section. Future work: Bayesian posteriors, discrete problems, and high-dimensional targets.
- **Noticed by me:**
  - The setting is an *unnormalised density with no data conditioning*: a single target, not amortised over observations. The "amortised" refers to the sampler, not to inference over datasets. Transferring it to our setting (q_φ(B | log)) is plausible but untested.
  - EUBO needs ground-truth samples, which are available on benchmarks but not on real posteriors.
  - The variance of the Ẑ^m estimates can dominate the buffer weights. One lucky batch could monopolise the buffer.

## 5. Ideas to steal
1. **A wake phase where SMC teaches the intuition net.** After sleep training on prior samples (forward KL), fine-tune q_φ on posterior samples produced by **SMC with q_φ itself as proposal**, run on real fight logs. The loss is the weighted NLL −Σ_k W_k log q_φ(B_k | log). The SMC samples come from rejuvenated particles (FFBS, split/merge), so they contain structure the net misses. This is exactly the "SMC as behaviour policy" loop, specialised to forward-KL / maximum-likelihood training. Forward KL on SMC samples is off-policy-friendly in the same way TB is.
2. **Adaptive tempering of SMC teacher weights.** When building wake-phase targets, temper the weights to keep ESS ≥ γK (γ ≈ 0.3–0.5), so a degenerate SMC run does not hand the net a one-particle "posterior." Log λ* over training: it is a free progress signal.
3. **Importance-weighted replay across bosses and attempts.** Store (log, B_k, Ẑ·W_k) from every slow-mind run, and sample replay in proportion to those weights. That gives the net a cross-boss memory of solved inferences. It is the analogue of hippocampal replay and fits the "learns from few deaths" story.
4. **Report the amortised sampler alone *and* with SMC.** They evaluate the sampler without SMC, while we have the amortised-only baseline in E6. Use their **EUBO-style mass-coverage metric** where ground truth exists (the tiny boss with an enumerable posterior): EUBO = E_{x∼p}[log p̃(x) − log q(x)] penalises a proposal that misses modes, while ELBO penalises one that puts mass in the wrong places. Report both on the tiny-boss correctness test.
5. **Mode-collapse diagnostic.** In multimodal boss posteriors (for example, two families that explain a log equally well), check whether q_φ covers both modes. Report the fraction of posterior modes that receive at least 1% proposal mass.

## 6. Structures and models of thought
- **A dictionary between RL and inference.** Proposal = policy, intermediate target = value/flow, resampling = selection. It suggests the metareasoning gate and the intuition net could share machinery: both estimate "how much will more computation help here."
- **Self-improving amortisation (an expert-iteration loop).** Slow, deliberate inference produces better answers; fast intuition is trained to imitate them; the improved intuition makes the next deliberation cheaper. This is the computational form of "practice turns System-2 into System-1" and fits the agent's death-to-death learning.
- **Experience replay weighted by evidence (Ẑ).** A principled memory consolidation rule: replay more often the episodes where your model explained the data well.
