# SEM: Franklin et al. 2020

## 1. Citation and access
- Nicholas T. Franklin, Kenneth A. Norman, Charan Ranganath, Jeffrey M. Zacks, Samuel J. Gershman. "Structured Event Memory: A neuro-symbolic model of event cognition." *Psychological Review* 127(3):327–361, 2020. PDF: https://www.psychologie.uzh.ch/dam/jcr:1f7b969a-3b15-418b-bd27-1a7e785e7570/Franklin_PsychRev_127_2020.pdf
- **What I read.** Most of the PDF (about 120k characters), through a text-extraction tool, in two passes. Greek symbols were lost in extraction, so some equations are rebuilt from the prose. I covered the model, inference, segmentation simulations, generalization, and the memory simulations up to sequential recall. The final section of the extraction (the end of the sequential-recall section, the general discussion, and Appendix C on Gibbs sampling) came back mostly empty. The limitations below come from statements scattered through the body.

## 2. Core idea and exact method
**Generative model.** Scenes x_n are embedded in a structured vector space (holographic reduced representations: circular-convolution binding of roles and fillers).
- An **event** e is a latent cause with its own dynamics function f(x_{1:n}; θ_e), a GRU network.
- Events follow a **sticky CRP**: P(e_n = k) ∝ C_k + κ·I[e_{n−1} = k] for an existing event k, and α for a new event.
  - α is the simplicity bias (smaller means fewer events).
  - κ is stickiness (larger means longer events).
- Transition likelihood: P(x_{n+1} | x_{1:n}, e) = N(x_{n+1}; f(x_{1:n}; θ_e), diag(β)), with β estimated by MAP under an inverse-χ² prior.
- After a boundary, the first scene is scored by integrating over a prior on the initial state f0. For a new event this is a constant.

**Dynamics network.**
- A GRU layer with leaky ReLU (slope 0.3) and 50% dropout, then a non-recurrent hidden layer and a linear output.
- Adam with learning rate 0.01, 10 epochs per update, trained on cached observations from that event.

**Inference.** Local MAP, in one forward sweep: ê_n = argmax_e P(e | s_{1:n}, ê_{1:n−1}).
- The log-likelihood is ≈ −‖x_{n+1} − f(x_{1:n}; θ_e)‖²/(2β), so high prediction error favors a boundary or a new event.
- SEM can tell a **restart** of the same event (boundary, but the event is reused) from a **switch** to a known event or a **new** event.

**Memory model.**
- Encoding: a corrupted trace ỹ of y = (x, e, n), with independent corruptions:
  - x̃ ~ N(x, σ²I);
  - the event label is kept with probability ε_e, otherwise replaced by a null label (a "Z-channel");
  - the time index is ñ ~ U[n−b, n+b];
  - there is also a null-memory probability (forgetting).
- **Reconstruction:** Gibbs sampling of the original scenes given the traces *and* the learned event dynamics. Event schemas regularize noisy memories toward typical scenes, which produces false memories.

## 3. Experiments and key numbers
**Video segmentation (Zacks et al. 2006).**
- 3 videos encoded by a VAE into 100 dimensions, 25 restarts each.
- Point-biserial correlation with human boundaries: model r_pb = 0.168 (SEM 0.013); humans r_pb = 0.29. The model falls inside the middle 75% of humans, [0.139, 0.400], and above the permutation null [0.067, 0.080].
- The model's log boundary probability against human boundary frequency gives r = 0.22.
- It misses knowledge-based boundaries. Example: "turning on the sink" was marked by 27% of humans and 0% of model runs.

**Community structure (Schapiro et al. 2013).**
- 1400-step random walk, then 600 Hamiltonian-path steps.
- Boundary probability is higher at community transitions, and so is surprise. This reconciles prediction-error accounts with community-structure accounts of segmentation.

**Generalization (Ask/Answer in role–filler space).**
- With new fillers: boundary probability 18.0%, new-event probability 12.2%, correct event assignment 82.0%.
- When the structure is violated, both boundary and new-event probabilities are high.
- A reduced stationary (HMM-like) model fails to tell these cases apart.
- Interference protection: normalized prediction error on a familiar event is 0.037 for SEM against 0.97 for the reduced model.

**Memory:**
- **False memory (Bower et al. 1979).** Recognition of a script-consistent lure rises with the number of same-script stories. The proportion of original traces in reconstructions also rises: 0.82, 0.88, 0.92.
- **Doorway effect (Radvansky).** Working-memory error is higher after an event shift.
- **Pettijohn et al. 2016.** Overall recall is higher when the list contains an event boundary.

**Parameters.** Per-simulation tables (embedding dimensions 25–266). The authors note that parameter values only have meaning relative to the representational space.

## 4. Limitations
**Reported by the authors:**
- A single forward sweep: no retrospective revision of boundaries.
- The VAE video features have no objects or relations.
- It misses boundaries that depend on prior knowledge.
- The structured representations are assumed, not learned.
- HRR binding is lossy.
- The model was not fit to memory data.
- It is agnostic about neural implementation.

**My own observations:**
- One separate network per event means no sharing between events. LCNet (Lu et al. 2024) addresses this.
- Duration enters only through stickiness (geometric).
- Local MAP gives no uncertainty over segmentations. That matters when deciding whether a new move really appeared or a known one was just slow.

## 5. Ideas to steal
1. **Restart vs switch vs new as a three-way decision at each surprise.** When the fast filter's ESS collapses, the slow mind should explicitly compare:
   - (a) the same move, re-timed (a delayed or restarted wind-up: update timing posteriors);
   - (b) a switch to a known move the transition model did not predict (update transitions or laws);
   - (c) a new move (CRP new table).
   - Report how the agent splits surprises across these three, against ground truth. SEM shows that people make the restart/switch distinction.
2. **Fix SEM's no-retrospection limitation and show it.**
   - Our FFBS/rejuvenation re-segments past attempts after death.
   - An experiment: on a boss whose delayed swing initially looks like a new move, a forward-only local-MAP agent (the SEM-style ablation) spawns spurious moves, while the full agent merges them after rejuvenation.
   - Metrics: number of inferred moves against the truth over attempts, and AMI.
3. **Boundary-quality metric.** Point-biserial correlation between agent boundary probability and true move onsets, plus correlation of log boundary probability with the human boundary frequency (if we collect human "mark the new move" annotations in the viewer).
4. **Memory reconstruction as a source of human-like errors.**
   - If the agent stores *compressed or corrupted* fight logs (as humans do) and reconstructs them through the boss posterior, it will regularize memories toward the schema.
   - Prediction: after many attempts on boss A, a variant A′ is mis-remembered as A, a false-memory analog. That produces slower adaptation to subtle changes (T2) in a human-like way.
   - Optional "bounded memory" agent variant for E4: keep only the last k attempts verbatim, plus the posterior.
5. **Community structure ↔ boss combo structure.** Combos are communities in the move-transition graph. SEM predicts boundaries (and human hesitation) at combo-to-combo transitions. Test whether the metareasoning gate plans more at inter-combo transitions than within combos (E5).
6. **Interference protection test.** After learning boss B, re-test on boss A. A per-cause or CRP agent should keep A (low prediction error), while a single shared model degrades. That is SEM's 0.037 against 0.97 contrast, and our family-library retention test.

## 6. Structures and models of thought
- **Events as latent causes with their own dynamics, under a sticky CRP.** These are the boss moves (and families). Our HSMM moves with explicit durations are a structured upgrade.
- **Structured role–filler representations** (HRR) enable generalizing an event to new fillers. The analog is a move primitive applied at a new distance or phase: laws with preconditions are our "roles."
- **Prediction error as the segmentation signal.** This matches our surprise-gated rejuvenation.
- **Memory as generative reconstruction.** A posterior over boss programs *is* a compressed memory of past fights. Reconstructing logs from it is FFBS sampling. This gives a principled link between the slow mind and human memory distortions.
- **Neuro-symbolic split.** Neural dynamics within events, symbolic CRP over events. This is the same "neural proposes, Bayes disposes" philosophy, at a different place in the hierarchy.
