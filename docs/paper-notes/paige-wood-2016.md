# Inference Networks for Sequential Monte Carlo in Graphical Models

## 1. Citation and how much I read
- Brooks Paige and Frank Wood, "Inference Networks for Sequential Monte Carlo in Graphical Models." ICML 2016 (PMLR 48). https://arxiv.org/abs/1602.06701
- **Read:** the full text, from title to references, via the arXiv PDF using a fetch tool that summarises the text (not a verbatim page-by-page read). Figures came through only as captions, so I could not read values off the plots. The paper has **no numeric tables**: every result is a figure.

## 2. Core idea and exact method
- **Goal.** Learn offline, before seeing any dataset, a neural "inference network" that approximates the *inverse* factorisation of a directed graphical model. The network is then used as an SMC proposal: "compile once, then each new dataset costs one forward pass." The paper frames this as amortised inference (Gershman & Goodman 2014) and as the sleep phase of wake–sleep.
- **Inverse model (§3.1).** Pick a topological order and reverse it, so observations become inputs. Each latent x_i gets as parents the later variables in its Markov blanket. Proposition 1 says conditional independences in the inverse graph also hold in the forward graph (a one-way guarantee, proof only sketched).
- **Objective (§3.2, Eqs. 10–12).** Expected **forward KL**, E_{p(y)} KL(p(x|y) ‖ q(x|φ(η,y))), which reduces to
  `J(η) = E_{p(x,y)}[ −log q(x | φ(η, y)) ] + const`.
  The gradient is an expectation over the *joint*, so you train on ancestral samples from the model, with no model gradients and no posterior samples. This is the template objective for our intuition net. (The paper links the forward-KL choice to EP and Minka 2005. My extraction called it "zero-forcing", but forward KL is mass-covering, i.e. zero-*avoiding*. That is the property you want in a proposal, since a proposal must not miss posterior mass.)
- **Proposal family (§3.3).** A MADE-style masked autoregressive network models jointly the groups of dependent latents (head-to-head nodes). Each continuous output is a **mixture of D Gaussians**: 3D outputs for means, softplus std devs, and softmax weights. With a single variable this reduces to a mixture density network.
- **Training (§3.4).** Adam with defaults and ReLU. They use a *hybrid* data scheme: a finite synthetic training set plus a synthetic validation set, **both regenerated whenever validation loss rises or after a maximum number of steps**. This sits between a fixed dataset and fresh samples at every step.
- **SMC use.** Standard SMC with incremental weights γ_n/(γ_{n−1} q_n). For the pump model they use divide-and-conquer SMC (Lindsten et al. 2014) with network proposals at each node.

## 3. Experiments
| Experiment | Model | Network | Result (all qualitative, from figures) |
|---|---|---|---|
| 4.1 Polynomial regression | Laplace priors on w_d ~ Laplace(0, 10^{1−d}); Student-t noise ν=4 | 2×300 hidden, 3-component MoG output | Proposals are "slightly more diffuse" than the MCMC posterior (MCMC: 10k samples after 10k burn-in, thinned by 100) |
| 4.2 Hierarchical Poisson (pumps) | α~Exp(1), β~Gamma(0.1,1), θ_n~Gamma(α,β), y_n~Poisson(θ_n t_n) | 2×500 hidden, 10-component MoG per output; **one shared network for all pumps** | D&C-SMC with network proposals gives "reasonable" log Z estimates **with as few as 5 particles**, versus likelihood-weighted IS and IS with the network (10 runs, ±2 sd) |
| 4.3 Factorial HMM | 20 binary devices (~2^20 states per step), μ∈[30,500], σ=20, switch prob 0.05 | 4×300 MLP; input = previous latent + current obs; output = 20 Bernoullis | With 100 particles, the prior (transition) proposal collapses to **one surviving ancestry within a few steps**, while the learned proposal keeps many more unique ancestries over 30 steps (10 runs) |

- **Metrics used:** log-normaliser estimates against particle count, and the **number of unique surviving ancestral paths** (a degeneracy metric).
- **Compute:** training time, hardware, and number of training samples are **not reported**.

## 4. Failure modes and limitations
- **Reported by the authors:** training "may be quite involved"; the method relies on a correct generative model (it must plausibly generate the data later observed); the learned posteriors are sometimes over-diffuse.
- **Not tested:** no out-of-distribution evaluation and no quantitative accuracy tables.
- **Noticed by me:**
  - The inverse graph structure is hand-derived per model. A program with stochastic structure, such as our boss HSMM with a variable number of moves, has no fixed graph to invert. Le et al. 2017 fixes this with an address-keyed LSTM.
  - Over-diffuse proposals are *safe* for SMC but waste particles. They are a symptom of the forward-KL objective, not a bug.

## 5. Ideas to steal
1. **Objective.** Use E_{p(B,log)}[−log q_φ(B|log)] exactly as the plan already does. Cite this paper together with Le et al. as the source. Say explicitly that it is the forward KL, which is mass-covering and therefore safe as a proposal, and contrast it with the reverse KL used in VI.
2. **Degeneracy metric for E6.** Besides ESS, report the **number of unique surviving ancestors** (or unique boss programs) after T resampling steps, for prior versus q_φ proposals. It is cheap and makes a vivid figure, a direct analogue of their Fig. 6.
3. **"Particles needed for reasonable log Z" curve.** Plot log Ẑ (mean ± 2 sd over 10 seeds) against the number of particles (1…1000) for prior, q_φ, and q_φ mixed with the prior. The x-intercept where the curve plateaus measures the amortisation gain. Our gate check ("≥5× fewer particles") should be read off this plot.
4. **Mixture-of-Gaussians heads** for timing parameters (duration means and dispersions). Use a 3–10 component MoG with softplus std, not a single Gaussian NLL, because duration posteriors after one or two deaths are often multimodal (for example, a delayed attack versus a fast one).
5. **Hybrid data regeneration.** Keep a synthetic train/validation pool and regenerate it when validation NLL rises. On a laptop, simulating fights is costly relative to the 1.5M-parameter net, so a reusable pool refreshed periodically is a good compromise.
6. **Share proposal factors across exchangeable parts** (they used one network for all pumps). Use one head per move-slot, shared across the slots of a boss program, instead of per-index parameters.

## 6. Structures and models of thought
- **Inverse (recognition) models as compiled intuition.** The network directly learns "observations → latent causes" for a fixed generative model. That is the bottom-up half of analysis-by-synthesis, and the sleep phase of wake–sleep, as cognitive models of perception describe it. In our agent, the intuition net is this compiled inverse of the boss program.
- **Exactness preserved by importance weighting.** Because the network only proposes and SMC re-weights, the agent's beliefs stay asymptotically correct even when its intuition is wrong. This is the formal basis of "neural proposes, Bayes disposes."
- **Factorisation following model structure** (Markov-blanket parents, shared factors). It suggests the net's heads should follow the HSMM's structure (segmentation → primitive label → timing law), not output one flat vector.
