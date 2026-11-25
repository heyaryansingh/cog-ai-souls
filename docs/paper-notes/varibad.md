# VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning

## 1. Citation and reading status
- Zintgraf, Shiarlis, Igl, Schulze, Gal, Hofmann, Whiteson. "VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning." ICLR 2020. https://arxiv.org/abs/1910.08348
- **How I read it:** I read the arXiv HTML full text (main paper plus appendices with hyperparameters) through a web-fetch tool, as a model-extracted pass, not as raw text. MuJoCo results are figure-only, so I describe them qualitatively. The paper has a minor internal inconsistency on the gridworld horizon (H⁺ given as both 45 and 60).

## 2. Core idea and exact method
- **Bayes-adaptive MDP framing.** The task (reward and transition functions) is a latent m. The Bayes-optimal policy acts on (state, belief b_t) to maximise return over a horizon H⁺ that spans *several episodes*, so exploration that pays off in later episodes is rewarded.
- **Problem.** Exact planning in a BAMDP is intractable, and posterior sampling explores badly.
- **Method:**
  - **Encoder.** A GRU over (s, a, r, s') produces q_φ(m | τ_{:t}), a Gaussian with latent dimension 5, updated every step. The prior at t = 0 is N(0, I).
  - **Decoder.** p_θ(r | s, m), and in the gridworld also p(s' | s, a, m). It reconstructs the **whole trajectory, past and future**, from the current belief, which forces the latent to predict unseen parts of the task.
  - **ELBO.** Summed over all t, reconstruction minus KL(q_t ‖ q_{t-1}), with the previous posterior as prior.
  - **Policy.** π(a | s, μ_t, σ_t). It conditions on the full belief parameters (mean *and* variance), so uncertainty drives exploration.
  - **RL gradients do not flow into the encoder.** The VAE and the policy have separate optimisers and buffers. This is faster and avoids conflicting gradients.
  - **Test time.** Encoder plus policy only. No gradient steps, no planning, decoder unused.
- **Sizes.**
  - Gridworld (A2C): policy 2×32; encoder FC40 → GRU64; reward decoder 2×32 with 25 output heads (BCE); 16 processes; 60-step rollouts.
  - MuJoCo (PPO): policy 2×128; encoder FC 32/16/16 → GRU128; reward decoder 64/32 (MSE); KL weight 0.1; batch 3200.
  - Total parameter counts are about 10⁴–10⁵ (my estimate).

## 3. Experiments
- **Gridworld.** 5×5 grid, random goal, H = 15, r = −0.1 per step and +1 at the goal, horizon of 4 episodes, 20 seeds.
  - Compared against a hand-coded **Bayes-optimal** policy and a **posterior-sampling** policy.
  - VariBAD is close to Bayes-optimal. Bayes-optimal is near-optimal from episode 2; posterior sampling needs about 6 episodes.
  - VariBAD learned the exact Bayes-optimal solution in 4 of 20 seeds; RL² in 0 of 20.
- **MuJoCo.** AntDir, HalfCheetahDir, HalfCheetahVel, Walker (random physics); H = 200; meta-trained with H⁺ = 2 episodes (400 steps).
  - Baselines: RL², PEARL, E-MAML, ProMP, and a PPO oracle with the true task.
  - Only VariBAD and RL² adapt *within the first episode*. VariBAD's first rollout is close to the oracle.
  - RL² is slower, less stable and worse on HalfCheetahDir.
  - PEARL (posterior sampling) is good only from episode 3.
  - The gradient-based methods (E-MAML, ProMP) are far behind in the first episodes.
- **Compute (HalfCheetahDir, rough).** ProMP and E-MAML 5–8 h; PEARL 24 h; **VariBAD 48 h**; RL² 60 h. On-policy, so frame-hungry (PEARL is far more sample-efficient).
- **Analysis.** The latent variance collapses once the goal is found. In MuJoCo, latent dimensions flip sign between left and right tasks within a few steps.

## 4. Limitations
- **Reported by the authors:**
  - No formal guarantees.
  - **Out-of-distribution tasks would likely break both inference and the policy.**
  - Decoder unused at test time; using it for planning is suggested as future work.
  - Fixed N(0, I) prior at episode start is likely suboptimal.
  - On-policy sample cost.
  - Belief is interpretable only in the gridworld.
  - Requires a meta-training task distribution.
- **Noticed by me:**
  - The latent has no structure (5-dimensional Gaussian), so it cannot express "new move added" or "compositional combo." This is a ready-made amortization gap.
  - No ablation removing the decoder is reported, even though it is the method's key idea.

## 5. Ideas to steal
1. **Add VariBAD as the "learned-belief" baseline**, alongside or instead of RL².
   - It sits exactly between RL² (no explicit belief) and our agent (structured explicit posterior). The baseline ladder becomes: RL² → VariBAD (learned unstructured belief) → amortized-only intuition net (learned structured proposal) → full SMC agent.
   - Each step adds one ingredient, which makes E6 a clean ladder.
2. **Laptop feasible at gridworld scale.** GRU64, latent 5, MLP 2×32–128 is tiny, and the cost is PPO frames. Budget it like the PPO baselines (1–3 CPU-hours, 2–5M steps). Use a cross-attempt horizon H⁺ = 4–6 attempts, so it learns to *scout* on attempt 1.
3. **Use their Bayes-optimal comparison.**
   - We have an exact HSMM filter per hypothesis, so for small boss families we can compute (or approximate with an oracle-model planner) the **Bayes-optimal deaths-to-win curve**.
   - Report every agent's gap to it, and their "number of episodes until near-optimal" metric (Bayes-optimal: 2, posterior sampling: 6) as a summary statistic for E1.
4. **"Decode the future" auxiliary loss for the intuition net.** Besides proposing a program, predict the *next* cue and timing (a posterior-predictive head) from the fight log so far. This is cheap, regularises the representation, and gives a calibration check for E5.
5. **Condition on variance, not only the mean.** The reflex policy and metareasoning gate should take posterior uncertainty features (entropy over the next move, predictive hit-time variance), not just the MAP. VariBAD shows this is what produces Bayes-adaptive exploration in an amortized policy.
6. **Stop the RL gradient into the belief module.** If the reflex policy is fine-tuned with RL, do not backpropagate into the intuition net. Their result shows it is faster and avoids conflict, and it keeps q_φ a pure inference network.
7. **Out-of-distribution prediction from the authors themselves.** They predict that out-of-distribution tasks break VariBAD. Cite this as prior support for the E6 hypothesis, and test VariBAD on T3/T4 directly.

## 6. Structures and models of thought
- **Belief-state agent (BAMDP).** The normative frame for "learning a boss from a few deaths": the right policy depends on the belief, and exploring on attempt 1 is optimal when the payoff comes in attempts 2–5. Our scouting attempts are a structured version of this.
- **Amortized inference network plus a predictive decoder** is the same structure as the intuition net. The difference is that ours outputs a *symbolic program* that SMC can verify and edit, while VariBAD's is an unverifiable vector.
- **Separating inference from control** (no shared gradients) mirrors the plan's split between a slow mind (inference) and reflex (control).
