# Reading the Boss, v4: capabilities, intermediate results, end results, and the structures of thought that get us there

*25 Nov 2026. Aryan Singh. Builds on [architecture-and-training-plan.md](architecture-and-training-plan.md) (v3). It comes from reading 35 papers in full or close to it. Per-paper notes are in [paper-notes/](paper-notes/), and each note says how completely its paper was read. Where this file cites a number, it comes from those notes.*

---

## 0. In one page

**What the finished agent should be able to do**, stated as testable capabilities:

| | Capability | One-line test |
|---|---|---|
| C1 | **Track** a boss's hidden state exactly | Filter posterior matches brute-force enumeration on tiny bosses |
| C2 | **Predict** when the hit lands, with honest uncertainty | ±2-tick hit-time prediction with calibrated 80% intervals; shows the human Bayesian timing signature |
| C3 | **Act well** given a correct model | Oracle agent beats ≥ 90% of training bosses |
| C4 | **Induce** a boss's program from a few deaths | Probe accuracy and distractor ranking reach ceiling within ~5 attempts |
| C5 | **Know what it doesn't know** | Detects when a boss breaks its own rules; calibrated surprise |
| C6 | **Explore to learn** | Scouting covers every (boss move × response) pair early, and the scouting weight causally speeds learning |
| C7 | **Think only when it matters** | Gated agent keeps ≥ 95% of planner wins while planning on < 30% of ticks |
| C8 | **Transfer** to new combos, new moves, new rules and new families | Flat "novelty slope" across T1–T5 |
| C9 | **Stay correct out of distribution** (headline) | Amortization-gap ratio stays near 1 for SMC + net, and grows for the net alone |
| C10 | **Learn like a person** | Agent and humans die to the same moves, show the same timing bias, and fail at the same dependency order |

**The single sentence the project should be able to defend at the end:** *"An agent that adapts by Bayesian inference over boss programs, with a learned network only proposing hypotheses, learns new souls-like bosses in a human-like number of deaths and stays correct under novelty where amortized and in-context learners break. Its mistakes look like human mistakes."*

---

## 1. Lessons from reading the papers in full

The full reads changed the plan in nine places, beyond what the abstracts gave:

1. **The in-context RL baselines need far more tasks than 40.** Algorithm Distillation shows *no* in-context learning with 1–18 training tasks, and only works with ~1,200–2,400 ([algorithm-distillation.md](paper-notes/algorithm-distillation.md)).
   - **Fix:** split the data into a **broad meta-prior generator**, available to every learning method for pretraining (thousands of bosses), and the **40 training bosses from a narrower true distribution Θ\***, used to learn Θ. Without this split, "learning to learn" would be trivial, because the intuition net would see Θ\* during pretraining. Every method gets the same generator, which keeps the comparison fair.
2. **Learned proposals can lose at high particle counts.** Perov et al. found that 10 learned-proposal particles matched thousands under the prior, but with many particles the prior proposal won ([perov-2015.md](paper-notes/perov-2015.md)).
   - **Fix:** use a mixture proposal, q = ε · prior + (1 − ε) · q_φ, and plot E6 across the whole particle-count range.
3. **No amortized method in the literature can repair its out-of-distribution errors at test time.**
   - **Distribution Transformers:** KL error rose 18× under prior shift.
   - **Inference compilation:** started below 10% on real Captchas until the priors were broadened.

   That is exactly the gap E6 targets ([distribution-transformers.md](paper-notes/distribution-transformers.md), [inference-compilation.md](paper-notes/inference-compilation.md)).
4. **The weak-limit HDP-HSMM sampler is the right engine.** It mixes much better than direct assignment and is about 50× faster per iteration. It needs two additions:
   - a **censoring term**, because the player often dies in the middle of a boss move;
   - an auxiliary geometric variable that restores Dirichlet conjugacy once self-transitions are removed ([hdp-hsmm.md](paper-notes/hdp-hsmm.md)).

   The paper has no split/merge moves; Saad et al.'s **detach/attach moves with multiple-try proposals** (U = 5–10) fill that gap and push acceptance near 1 ([saad-smc-learning.md](paper-notes/saad-smc-learning.md)).
5. **Rejuvenate where prediction first failed, not everywhere.**
   - **SIPS** resamples from the first point where predicted and observed behavior diverge, and only recomputes particles whose plan failed ([sips.md](paper-notes/sips.md)).
   - **OPINE-World** aborts a plan on the first prediction mismatch and logs it as a counterexample ([opine-world.md](paper-notes/opine-world.md)).
   - **WorldLLM** feeds its proposer the worst-predicted data ([worldllm.md](paper-notes/worldllm.md)).

   Together: give the intuition net the attempt's worst-predicted segments, and rejuvenate from the divergence tick.
6. **Human event boundaries track Bayesian surprise (KL between successive beliefs), not surprisal.** Correlations are small (r ≈ 0.06–0.12; the Kumar et al. note is marked "verify before citing") ([bayesian-surprise-segmentation.md](paper-notes/bayesian-surprise-segmentation.md)). SEM's model fit human boundaries at 0.168, against 0.29 between humans ([sem-franklin-2020.md](paper-notes/sem-franklin-2020.md)).
   - **Fix:** the gate compares three triggers: ESS collapse, surprisal, and Bayesian surprise.
7. **Hypothesis generation, not evaluation, is the bottleneck in human opponent learning.** In Cross et al.'s rock-paper-scissors study, handing the model the true hypothesis fixed almost everything ([cross2025-rps.md](paper-notes/cross2025-rps.md)). Humans beat first-order bots (~60%), struggle with outcome-conditioned bots (~40%), and are at chance against combined bots. Brockbank & Vul find humans exploit only base rates and last-move dependencies ([brockbank-vul-rps.md](paper-notes/brockbank-vul-rps.md)).
   - **Fix:** adopt their **three-condition decomposition** (§3), and use transition-dependency order as a controlled difficulty axis.
8. **Copy TRM's deep supervision with a detached carry; recursion alone isn't the point.** A 1-step gradient approximation dropped Sudoku accuracy from 87.4% to 56.5%. Removing EMA cost 7.5 points. 2 layers beat 4 ([trm.md](paper-notes/trm.md)).
   - TRM's halt head becomes a learned "will another pass improve the SMC score?" signal for the gate.
   - The published TRM took 36 h on an L40S with 1000× augmentation. We use **boss symmetries** (move-label permutation, time shifts) as augmentation and keep d small.
9. **A cheap model-based baseline is strong.** In FightingICE, a planner with a *linear* softmax opponent model, retrained between rounds, reached 87–93% win rates against competition bots; bigger nets did worse ([fightingice-rhea.md](paper-notes/fightingice-rhea.md)).
   - **Add as baseline B6.** It tells us whether the *Bayesian, compositional* parts matter, or just "model-based plus planning."

---

## 2. Capability ladder, with metrics, pass criteria and the structures that get there

Each capability lists:
- **Metric:** exact definitions.
- **Pass:** the go/no-go threshold. These are targets I set; adjust after the first runs.
- **Gets there:** which structures and models of thought produce it, with their source.
- **Intermediate result:** the figure or table it produces.

### C1. Track (exact belief state)

- **Metric.** On tiny bosses (2–3 moves, durations ≤ 8 ticks): total variation distance between the filter posterior and brute-force enumeration. For SMC over programs: C2ST, MMD and W₂ against the exact posterior, plus an EUBO/ELBO sandwich (from [reuter2025-icl-bayes.md](paper-notes/reuter2025-icl-bayes.md)).
- **Pass.** TV < 0.01 for the exact filter. SMC converges to the enumeration as N grows. The log-evidence estimate's variance falls as roughly 1/N.
- **Gets there.**
  - **Analysis-by-synthesis:** one generative program used both to generate and to score.
  - **Exact HSMM messages with censoring:** Johnson & Willsky.
  - **Rao-Blackwellization and SMC²:** an exact inner filter per hypothesis, with formal permission to swap in an inner particle filter if the state grows (Chopin et al.).
- **Intermediate result.** *Figure 1: correctness.* Posterior error against N for prior proposal, q_φ proposal and mixture proposal. A clean course-relevant figure (Lec 8–12).

### C2. Predict (calibrated timing)

- **Metric.** Hit-time error (ticks); 80% interval coverage; expected calibration error; held-out log-likelihood of continuations.
- **Pass.** On a known boss: |error| ≤ 2 ticks median, coverage 75–85%, ECE < 0.05.
- **Human signature** (Jazayeri & Shadlen, [jazayeri-shadlen-2010.md](paper-notes/jazayeri-shadlen-2010.md)):
  - With scalar timing noise switched on, dodge times for long delayed attacks are pulled toward the prior mean.
  - The *same* wind-up length produces different dodge times on bosses whose delay sets overlap differently.
  - Plot bias against variance, as the paper does.
- **Gets there.**
  - **Bayesian least-squares timing.** The posterior-mean estimate beat MAP and maximum likelihood for all six subjects in the paper. We generalize it to minimizing the *actual i-frame loss* over the posterior predictive.
  - **Negative-binomial durations** with a prior on shape, which tells peaked timings from geometric ones (Johnson & Willsky).
  - **Bimodal "fast-or-delayed" moves**, from Körding & Wolpert's two-mode prior ([kording-wolpert-2004.md](paper-notes/kording-wolpert-2004.md)). A MAP agent should die to both versions.
- **Intermediate result.** *Figure 2: the timing signature,* for the agent with and without scalar noise, later with humans.

### C3. Act (given a correct model)

- **Metric.** Win rate of the oracle agent (true boss program given), fraction of boss HP removed, and survival time, plus the **dense score** f = 100 · (H_player + 1 − H_boss) for runs where wins are rare (from [dsle.md](paper-notes/dsle.md)).
- **Pass.** Oracle wins ≥ 90% of training bosses within 3 attempts, with no fast-death or stalemate failures.
- **Gets there.**
  - **SSUP-style "sample, simulate, update" planning:** about 4 rollouts per candidate, act when expected value passes a threshold, otherwise the best after a timeout ([ssup.md](paper-notes/ssup.md)).
  - **EMPA's planner controller:**
    - long, short or stall modes;
    - a "lost twice the same way" check for being stuck;
    - doubling the search budget on failure;
    - watching before acting.

    Source: [empa.md](paper-notes/empa.md).
  - **Checked planning:** abort on the first mismatch (OPINE-World).
- **Intermediate result.** *Table 1: acting ceiling,* with every death coded into the failure taxonomy (§3).

### C4. Induce (world-model induction from deaths)

- **Metrics:**
  - **Probe accuracy** (AgentOdyssey-style, [agentodyssey.md](paper-notes/agentodyssey.md)). Templated multiple-choice questions about the boss program are asked at attempts 0, 1, 2, 5, 10 and 20, with ground truth from the generator. Examples: "Which move follows the triple combo when you're far away?" and "How long is the delayed slam's wind-up?" Distractors are real but wrong moves.
  - **Distractor ranking** (OneLife, [onelife.md](paper-notes/onelife.md)). A true held-out continuation is ranked against 6–10 mutated ones (confusable move swapped in, wind-up shifted, law violated). Reported as Rank@1 and MRR.
  - **Normalized predictive area** (WorldLLM). 1 − mean_t [log p(test | H_t) / log p(test | prior)]. Below 0 means inference made things worse.
  - **Deaths until the true program first enters the particle set** (from [infinite-ends.md](paper-notes/infinite-ends.md)). This directly measures proposal quality.
  - **Induction speed.** Attempts until C2's pass criterion holds on a new boss.
- **Pass.** Rank@1 ≥ 0.9 and probe accuracy ≥ 85% by attempt 5 on T1/T2 bosses. True program in the particle set by attempt 3 on T1.
- **Gets there.**
  - **Probabilistic language of thought:** the boss as a program over a primitive library (BPL; course motif M3).
  - **Conditionally activated laws:** OneLife updates a law's weight only from transitions where it fired, which lifted Rank@1 from 10.8% to 18.7% over PoE-World-style inference.
  - **Bottom-up proposals with top-down filtering:** each step proposes N new hypotheses, weights them by p/q, merges them with survivors, resamples, and coalesces duplicates by summing weights (Infinite Ends).
  - **Resample-move SMC** with detach/attach and multiple-try (Saad); **weak-limit HDP-HSMM Gibbs** (Johnson & Willsky).
  - **Error-driven rejuvenation** from the divergence tick (SIPS).
- **Intermediate result.** *Figure 3: induction curves.* Probe accuracy, Rank@1 and the predictive-area metric against attempt number, for each agent.

### C5. Know what it doesn't know

- **Metrics.**
  - **Anomaly detection** (from [ying2025-adaptive-world-models.md](paper-notes/ying2025-adaptive-world-models.md)): in some attempts the boss breaks one of its own laws, and we measure AUROC of the agent's surprise signal for detecting it.
  - Calibration (C2).
  - **Excess entropy:** observed outcome entropy minus the entropy the posterior predicts. This is the stochastic-world version of OPINE-World's "missing variable" test, whose original assumes deterministic dynamics.
- **Pass.** Anomaly AUROC ≥ 0.9. Excess entropy rises within one attempt of a new move appearing.
- **Gets there.**
  - **Metacognition as a posterior over one's own model:** ESS, Bayesian surprise, model evidence.
  - **Running model evidence from SMC²**, for free.
  - **Latent-cause inference:** on each surprise, an explicit choice between "same move re-timed," "known move" and "new move" (SEM/LCNet, [lcnet.md](paper-notes/lcnet.md)).
- **Intermediate result.** *Figure 4: surprise traces* aligned to anomalies and new-move onsets, comparing the three trigger signals.

### C6. Explore to learn (scouting)

- **Metrics.**
  - **Coverage of (boss move × player response) pairs**, adapted from EMPA's "make contact with every unseen pair of object types."
  - **Action diversity** as normalized entropy, −Σ p log p / log N (AgentOdyssey). Falling diversity coincided with reward plateaus there.
  - The **causal scouting test:** sweep the scouting weight β and measure later learning speed. This turns Stafford & Dewar's correlation (early variance predicted later score, r = 0.59 over 854k players; [stafford-dewar-2014.md](paper-notes/stafford-dewar-2014.md)) into an experiment.
- **Pass.** Coverage ≥ 90% by attempt 3. There is a β > 0 that beats β = 0 on deaths-to-first-win.
- **Gets there.**
  - **Directed exploration by particle disagreement:** cheap, and used by Colas et al. ([colas2025-language-experience.md](paper-notes/colas2025-language-experience.md)).
  - **Thompson sampling over an exploration table:** removing it cost OPINE-World 20 RHAE points.
  - **Resetting exploration goals on phase or HP change** (EMPA).
  - **Surprise-seeking over learning progress:** WorldLLM found learning-progress rewards mostly zero, so we default to surprise.
- **Intermediate result.** *Figure 5: β sweep,* with coverage curves.

### C7. Think only when it matters (metareasoning)

- **Metrics.**
  - Fraction of ticks planned.
  - Win rate relative to the always-plan agent.
  - The **wall-clock ÷ game-time ratio** (AI GameStore: frontier models run about 12–18× slower than real time).
  - The speed–accuracy curve as compute cost c varies.
- **Pass.** At most 30% of ticks planned with at least 95% of planner wins. Ratio below 1 (faster than real time) on a laptop.
- **Gets there.**
  - **Dual process by amortization:** a reflex distilled from the planner by DAgger.
  - **Resource-rational value of computation:** the gate regresses Q_planner − Q_reflex.
  - **SSUP's threshold-plus-timeout** as the simple baseline gate.
  - **TRM's halting head** reused as "will thinking more help?"
- **Intermediate result.** *Figure 6: speed–accuracy frontier,* with where the gate chooses to think, overlaid on wind-up onsets.

### C8. Transfer

- **Splits.**
  - **T1:** new timings.
  - **T2:** new combinations.
  - **T3:** unseen primitive.
  - **T4:** changed player rules.
  - **T5 (new, from Ying et al.):** an *unannounced mid-attempt rule change*, where the boss enters a hidden phase that rewires one law.
  - **Re-skinned control (from WorldLLM):** cue tokens permuted, dynamics identical. This tests whether the intuition net learned structure or memorized tokens.
  - **Family test:** a variant of a training family (like PoE-World's "Alt" games).
- **Metrics.**
  - **Novelty slope:** change in area under the win-rate curve from T1 to each T_k.
  - Attempts to recognize the family.
  - Attempts for the self-model to recover rules R.
- **Pass.**
  - Family recognized in 1 attempt.
  - T4 rules recovered within 3 attempts.
  - T5 change detected within the same attempt.
  - Re-skinned performance within 10% of normal after one attempt.
- **Gets there.**
  - **Hierarchical Bayes** (Θ learned from 40 bosses).
  - **CRP over families**, with the evidence from SMC² used for recognition.
  - **Context embeddings per family**, which are structured, not random (LCNet).
  - **Self-model** as inference over one's own body rules.
- **Intermediate result.** *Figure 7: novelty slopes per method.*

### C9. Stay correct out of distribution (the headline)

- **Metrics.**
  - **Amortization-gap ratio:** error on T_k divided by error on T1, for (a) amortized-only, (b) SMC + prior proposal, (c) SMC + q_φ, (d) SMC + mixture proposal.
  - ESS and unique ancestors per attempt.
  - Accuracy against particle count over the whole range (Perov's lesson).
  - A **broadened-prior** ablation for the net (inference compilation's fix).
- **Pass (the claim).**
  - Amortized-only ≈ SMC + q_φ on T1, but its ratio is ≥ 3 on T3 and T5.
  - SMC + q_φ keeps its ratio ≤ 1.5, at ≤ 1/5 the particles of prior-only SMC.
- **Gets there.**
  - **"Neural proposes, Bayes disposes":** importance weights make proposal errors cost time, not correctness.
  - **The mixture proposal.**
  - **Conditioning the net on Θ**, with a prior-reconstruction loss.
- **Intermediate result.** *Figure 8: the amortization-gap plot.* This is the paper's main figure.

### C10. Learn like a person

- **Human study.** At least 8 people, each playing 6 bosses for up to 15 attempts in the pygame viewer, in a single session (Stafford & Dewar found spacing has small effects, so one session avoids that confound).
  - A colour-only rendering, so prior game knowledge helps less (EMPA's protocol).
  - Hyperparameters frozen before running people (SSUP's practice).
- **Metrics.**
  1. Correlation of per-move death rates between agent and humans.
  2. Learning efficiency κ = (bosses beaten / total) × (bosses beaten / attempts), compared in ratio bands as EMPA does (EMPA was within 0.1–10× of humans on 79/90 games).
  3. **Human-normalized score** clip(100 · agent / human median, 1, 10000), aggregated by geometric mean with 95% CI (AI GameStore).
  4. **Exploitability captured per dependency order**, the fraction of available advantage won against first-order, outcome-conditioned and combined bosses (Brockbank & Vul's expected win-count differential).
  5. The timing signature from C2.
  6. **Blocked versus interleaved boss curricula.** LCNet predicts blocked training helps both the CRP agent and humans.
  7. Per-run learning curves, fitted with power and exponential forms on individual runs rather than averages.
- **Pass.** Per-move death correlation r ≥ 0.5. Same rank order of difficulty across dependency orders. Agent κ within 0.5–2× of humans.
- **Gets there.**
  - **The human-like agent variant:** N = 2 particles (Infinite Ends found 2 particles fit humans better than exact inference), scalar timing noise, and a truncated history window (testing Brockbank & Vul's memory-limit explanation).
- **Intermediate result.** *Figure 9: agent versus humans,* per-move scatter, curves and dependency-order bars.

---

## 3. Why did it die? The failure decomposition

### Three-condition decomposition (adapted from Cross et al.)

Run every agent variant three ways:

| Condition | What changes | A failure here means |
|---|---|---|
| **Normal** | nothing | any of the below |
| **Truth-in-set** | the true program is guaranteed to be among the particles | *proposal* failure is ruled out; what remains is evaluation or acting |
| **Posterior = truth** (oracle) | all weight on the true program | only *execution* failure remains |

The gaps between rows separate **proposal**, **evaluation** and **execution** errors. This is the single most diagnostic table in the project.

### Death taxonomy

This is coded automatically from the agent's internals at the moment of death. It combines DSLE's failure modes with AgentOdyssey's ability-based failures.

1. **Unseen move:** the move was not in the posterior support.
2. **Mispredicted timing:** the move was right, but the hit-time error was more than the i-frame width.
3. **Right prediction, wrong action:** an acting error.
4. **Resource mismanagement:** stamina or heal.
5. **Attrition:** died slowly while trading.
6. **Stalemate:** long survival, no damage dealt.
7. **Repeated identical death:** the same move and cause as the last attempt, which signals a learning failure.

An explicit posterior makes this coding possible. Black-box baselines can only be coded by categories 4–7, and that contrast is itself a result.

### Boss difficulty profile

Each boss is rated on timing tightness, cue ambiguity, dependency order, novelty and memory load, computed from its program rather than by hand. We regress performance on these, as AI GameStore does with its capability rubric.

---

## 4. Milestones and intermediate results (go/no-go)

| Milestone | Target date | Deliverable | Go criterion | If it fails |
|---|---|---|---|---|
| **M0: env and generator** | Nov 28 | MiniSouls, broad meta-prior plus narrow Θ\*, splits T1–T5 and the re-skinned control, pygame viewer, replay test of how random each boss is | > 20k ticks/s batched; replay test shows bosses are stochastic but learnable | Simplify cues; fix the RNG |
| **M1: exact filter and oracle** | Nov 30 | HSMM filter with censoring, oracle planner, **Fig 1 (tiny-boss correctness)**, **Table 1** | TV < 0.01; oracle ≥ 90% on training bosses | Tune utility and horizon; add EMPA's stall mode |
| **M2: slow mind without the net** | Dec 3 | Weak-limit Gibbs, FFBS, detach/attach with multiple-try, CRP; induction curves with the prior proposal | True program in the set by attempt 5 on T1 with N = 64 | Fix K; drop laws |
| **M3: intuition net** | Dec 5 | Net trained on the broad prior, **ESS gain**, held-out per-head NLL | ≥ 5× particle savings at matched accuracy | Shrink the model; GRU; prior-heavy mixture |
| **M4: full agent and lifelong** | Dec 7 | Θ from 40 bosses, families, self-model, reflex and gate; **Figs 3, 6** | Deaths-to-first-win ≤ 8 median on T1/T2 | Cut families, then laws |
| **M5: baselines and novelty** | Dec 10 | Baselines below, run on T1–T5 and the re-skinned control; **Figs 7, 8**; failure decomposition table | E6 claim holds or is cleanly refuted | Report honestly. A refutation is still a result |
| **M6: humans and write-up** | Dec 12+ | **Figs 2, 9**; report | ≥ 8 participants | Report as a pilot |

**Baselines in M5, all pretrained on the same broad meta-prior generator:**
- B1: PPO.
- B2: RL².
- B3: per-boss PPO fine-tuning.
- **B4: VariBAD.** Small: GRU 64–128, latent size 5, about the cost of PPO ([varibad.md](paper-notes/varibad.md)).
- B5: amortized-only.
- **B6: planner plus a linear opponent model retrained between attempts** (FightingICE).
- B7 (optional): Algorithm Distillation on more than 1,000 prior bosses with event-level tokens, or a RAD-lite that summarizes each finished attempt.
- B8 (optional): LLM agent with memory.

---

## 5. End results: the report's claims and what would falsify them

| Claim | Evidence | Falsified if |
|---|---|---|
| **R1.** Bayesian program induction learns bosses in a human-like number of deaths | Fig 3, Fig 9, κ ratio | Median deaths-to-first-win more than 2× the human median on T1/T2 |
| **R2.** Explicit inference beats amortized and in-context learners under novelty | Fig 8, Fig 7 | Amortized-only or VariBAD matches the full agent on T3/T5 |
| **R3.** The Bayesian and compositional parts matter, not just "model-based plus planning" | B6 against ours; ablations MAP, no Θ, no CRP, no laws | B6 ≈ ours on all splits |
| **R4.** Keeping a posterior (averaging) beats committing to one hypothesis | MAP ablation on bimodal and delayed moves | MAP ≈ full agent on bimodal bosses |
| **R5.** Scouting causally speeds learning | β sweep | No β > 0 beats β = 0 |
| **R6.** Agent errors resemble human errors | Per-move correlation, timing signature, dependency-order bars | r < 0.3 and different difficulty ordering |
| **R7.** It is cheap | Wall-clock ratio, laptop CPU-hours | Slower than real time with the gate on |

Even if R2 or R6 fail, the report still has a clear contribution: an environment, a correct inference engine, and a decomposition of why agents die.

---

## 6. Structures and models of thought, and what each buys

| Model of thought (cognitive) | Structure in the agent | Source | Capabilities it enables |
|---|---|---|---|
| **Probabilistic language of thought / program induction** | Boss = program over primitives plus conditional laws | Lake et al. (BPL, "Building Machines"); OneLife; PoE-World | C4, C8 |
| **Analysis-by-synthesis** | One generative program used to sample and to score | Course motif M5; 3DP3, PClean | C1, C4 |
| **Bayesian observer–actor** | Posterior-mean timing that minimizes i-frame loss; scalar noise | Jazayeri & Shadlen; Körding & Wolpert | C2, C10 |
| **Intuitive psychology / inverse planning** | Opponent as a structured agent; error-driven SMC | SIPS; Baker et al. | C4, C5 |
| **Bottom-up proposals, top-down filtering** | Intuition net + importance-weighted SMC; merge and coalesce | Infinite Ends; inference compilation | C4, C9 |
| **Latent causes and event segmentation** | CRP over moves and families; restart/switch/new decision; Bayesian-surprise boundaries | SEM, LCNet, Kumar et al. | C5, C8 |
| **Simulation-based trial and error** | Sample, simulate, update; learn from each death | SSUP | C3, C4 |
| **Theory-based exploration** | Cover every (move × response) pair; reset on phase change; disagreement value | EMPA; Colas et al.; OPINE-World | C6 |
| **Dual process / habit vs deliberation** | Reflex distilled from the planner | DAgger; amortization | C7 |
| **Resource rationality / metacognition** | Gate on the value of computation; halting head; three trigger signals | Lieder & Griffiths; TRM | C5, C7 |
| **Hierarchical Bayes / learning to learn** | Θ from 40 bosses; family prior | Course motif M11; Lake et al. | C8 |
| **Bounded memory** (human variant) | Truncated history, N = 2 particles | Brockbank & Vul; Infinite Ends | C10 |
| **Iterative refinement** | Recursive proposer with deep supervision, detached carry, EMA | TRM | C9 |

### Intuition-net specifics updated from the reading

- **Training.** Forward-KL training on fresh samples generated on the fly, as in inference compilation. If simulation becomes the bottleneck, keep a pool regenerated on validation-loss rise (Paige & Wood).
- **Output heads.**
  - Keyed by structural "address": segmentation, label, timing, family.
  - **Canonical move ordering** breaks label symmetry.
  - **Mixture-of-Gaussians timing heads**, so bimodal moves are representable.
- **Inputs.** The observation embedding is fed at every step, which trained 3× faster in inference compilation. The net is also **conditioned on Θ**.
- **Architecture.** 2 layers with recursion, tied weights across passes, untied embedding and output head, and EMA 0.999 (TRM).
- **Wake phase.** Fine-tune on weighted SMC posteriors from the agent's own fights, with weight tempering to keep ESS ≥ γK (reinforced SMC).

---

## 7. What changed from v3

- Added C5, C10, the T5 split and the re-skinned control.
- **Baselines.** Pretrained on a broad meta-prior. Added VariBAD and the FightingICE-style planner. AD is now optional with a >1,000-boss requirement.
- **Inference engine.** Weak-limit HDP-HSMM Gibbs with censoring; detach/attach with multiple-try; error-driven rejuvenation from the divergence tick; mixture proposal; three gate triggers.
- **Evaluation.**
  - The three-condition failure decomposition and the automatic death taxonomy.
  - Probes and distractor ranking for model quality.
  - Human-normalized score and κ for human comparison.

## Sources

All in [paper-notes/](paper-notes/), one file per paper with citation, URL and reading completeness. Two notes flag issues to check before citing: author names on reinforced SMC and Distribution Transformers, and the Kumar et al. full text, which did not load.
