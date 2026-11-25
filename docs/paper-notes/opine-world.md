# OPINE-World: Programmatic World Modeling with Ontology-error-Prioritized Interactive Exploration for ARC-AGI-3

## 1. Citation and reading status
- David Courtis, Wenhao Li, Scott Sanner (University of Toronto; Vector Institute). arXiv:2607.01531 (v1 submitted 1 Jul 2026; v3 26 Sep 2026). Preprint, no venue. https://arxiv.org/abs/2607.01531
- **Reading status (honest):** Direct arXiv download was blocked in my sandbox, so I read through a web-fetch tool that returns extracted/condensed text. I covered the v3 HTML (sections 1 to 3.2.1) and the full v3 PDF (methods 3.2.2–3.2.4, experiments, Tables 1, 5–7, limitations, Appendix E). **Caveat:** fetching the unversioned PDF returned an earlier-looking text with a different section layout (Bayesian material in §3.4, "two cooperating agents", no ablations, Table 2 per-game actions). Numbers below are from v3 unless marked "(earlier version)". RHAE's formal definition (Appendix G) I saw only in the earlier text.

## 2. Core idea and exact method
**Problem.** ARC-AGI-3: 25 unknown interactive pixel games. State space, transitions, reward and goal are all hidden; the agent sees frames and a sparse level-complete reward, plays online with no level resets. WorldCoder-style program synthesis fails because it assumes a given symbolic state.

**Model W_t (Eq. 2), three cooperating levels:**
1. **NL proto-model** (Ŝ_NL, T̂_NL, R̂_NL): qualitative, partial, non-executable claims ("colour may be a confounder", candidate goals).
2. **Bayesian belief Ŝ_B**: empirical effect distributions per (object type, action, context), from a deterministic replay-based extractor.
3. **Programmatic model**: object extractor f̂⁻¹ (frames -> typed objects; Eq. 3 ŝ_t = f̂⁻¹(ŝ_{t-1}, ω_t)); deterministic transition program with per-fluent update functions ŝ⁺[X(o)] = F_X(o; ŝ, a) (Eq. 4, RDDL-like); reward program R̂ = r⁰ + (r⁺ − r⁰)·Ĝ(ŝ, a, ŝ') with a Boolean goal predicate Ĝ (Eq. 6); planner P̂.

**Learning loop (CEGIS).** Observe -> abstract objects, associate across frames, update Ŝ_B -> every mismatch between prediction and observation becomes a counterexample; synthesize revised (T̂, R̂, P̂, f̂⁻¹) with an LLM (Claude Opus 4.8) -> exploit if a goal plan exists, else explore -> execute.
- **Predictive verification Φ₁ (Eq. 7):** a hypothesis is accepted only if it is *replay-consistent*: for every stored transition, predicted successor (on verifier-selected fields V) and predicted reward match the observation exactly. Hard filter, not a score.

**Bayesian effect uncertainty (§3.2.3, Eqs. 10, 26–30).**
- Effect signature: for each paired object, Δ = set of attributes that changed; signature e = ρ(Δ) (which attributes changed, not to what; "gone" for disappearance).
- Rows j = (type τ, action a, context feature u); counts C(j, e).
- Symmetric Dirichlet: q_j ~ Dir(α₀·1 + c_j), posterior mean q̂_j(e) = (α₀ + C(j,e)) / (mα₀ + n_j), **α₀ = 1**, m_min = 2.
- Row entropy η_j = H(q̂_j)/log m in [0,1]; type entropy U_type = H[P(τ_i | D)]/log K (type posterior ∝ prior × ∏ plug-in likelihoods).
- **Ontology error:** per object, noisy-OR η_i = 1 − (1 − U_type)(1 − U_row); aggregate η_t = Σ n_j η_j / Σ n_j. Logic: the true dynamics are deterministic, so a *mixed* effect distribution means a missing state distinction (a hidden variable or a wrong type), not noise.
- **Uses of η:** (i) groups ranked by n_j·η_j go to the synthesizer, which keeps an entropy-minimizing split if found; (ii) **context-split gate (Eq. 30):** accept a new context feature if it lowers uncertainty by ≥ δ = 0.05, has ≥ 3 samples in one sub-row, and modal share ≥ 0.95; (iii) **exploration matrix** over (type, action): entries ordered by 1 − sample from Beta(1 + s, 1 + n − s) (s = count of modal effect) — i.e., **Thompson-sampling-style probing of the least-determined (type, action) cells**; refined bins use 1 − s/(n+1).
- Epistemic experiments are run "when no Φ₂-satisfying plan exists" (Φ₂: a checked goal plan).

**Goal grounding and checked planning (§3.2.4).** Before the first level clears, goal relevance comes from reward. After, a Boolean goal predicate is synthesized and verified by exact replay (a static check rejects predicates that peek at cached future states). The planner (bounded forward search) runs only after one level is cleared. It is first verified by replanning from entry states of cleared levels. Each planned step is executed live and compared to prediction. On mismatch the plan aborts, the transition becomes a counterexample, and planning is blocked until the next synthesis round.

## 3. Experiments
- ARC-AGI-3 public set, 25 games, 183 levels; single run per game; base LLM frozen (Claude Opus 4.8 high).
- **Metric RHAE** (relative human action efficiency). Earlier-version text: per level score = min(1.15, (human actions / agent actions)²), levels index-weighted, game score capped at 100, averaged over games.
- **Table 1 (v3):** Full 78.4 ± 6.3 RHAE, 160/183 levels, 20/25 games. No-Ŝ_B 58.2 ± 8.3, 126/183, 12/25. Best other system (baseline1, GPT-5.5 high coding agent) 63.7 ± 7.4, 146/183, 14/25. Base Claude Opus 4.8: 1.5%. WorldCoder and latent world models (Dreamer/MuZero family): 0.
- **Table 7 (paired RQ1):** Ŝ_B ablation costs 20.24 RHAE (d_z = 0.69, exact p = 4.3e-4), 34 more levels (+18.6 pts completion, 95% CI [8.0, 31.0]).
- **Table 6 (400-action budget):** Full 65.97 / 141 levels / 13 games; No-Ŝ_B 52.96 / 115 / 8; ε-greedy (ε = 0.1) 39.49 / 106 / 7.
- Selected per-game (Full vs No-Ŝ_B): wa30 100 vs 5.0; q50t 61.7 vs 0; dc22 82.4 vs 14.3; cn04 100 vs 47.6.
- (Earlier version) on 6 games baseline1 fails, OPINE used 2,578 actions vs humans 3,994 vs baseline1 10,874; beat human action count on 16/20 wins; mean human/agent ratio 1.7 on wins.
- RQ3 (NL layer ablation): full system needs fewer actions, synthesis rounds and tokens early in learning (Fig. 5, App. F.8); no RHAE number given.
- Compute: up to **2 billion cumulative tokens** per environment; no dollar or wall-clock numbers.

## 4. Failure modes and limitations
- Reported: deterministic, discrete, observable-Markov games only; under hidden state the ontology error has a floor (mixed outcomes are irreducible) though replay still rejects wrong models; replay consistency guarantees nothing for unvisited states; perception (object extraction/pairing) is itself inferred and noisy; bounded planner fails on high-branching games; single run per game, no variance over runs; very expensive.
- Noticed: (a) the core heuristic "mixed outcomes = missing variable" directly conflicts with genuinely stochastic environments (like a boss with random move choice) — there it would chase noise forever; (b) exact-replay acceptance is brittle with noise; (c) the comparison confounds base model (Claude vs GPT) with design; (d) ablations only for Ŝ_B, others "unstable to remove".

## 5. Ideas to steal for Reading the Boss
1. **Ontology-error signal adapted to stochastic bosses.** The boss *is* stochastic, so raw outcome entropy is not an error signal. Use instead: **excess entropy** = observed conditional entropy of next-move given the current context features minus the entropy the current posterior predicts, or a *split test*: for candidate context feature u (distance bin, phase, HP band, last-2 moves), accept a new law when splitting by u lowers next-move entropy by ≥ δ with ≥ n samples per sub-row. Copy their gate numbers as starting points (δ = 0.05 nats-normalized, n ≥ 3, modal share ≥ 0.95 for deterministic laws). This is a concrete proposal rule for **law birth** in the SMC rejuvenation step and a target for the intuition net's law-activation head.
2. **Thompson-sampled exploration matrix for scouting attempts.** Maintain Beta/Dirichlet counts over (boss move, player context/action) cells, e.g. "what does the boss do after I heal at mid range". During scouting, pick the context to provoke by sampling 1 − Beta(1 + s, 1 + n − s). This is a cheap, well-defined stand-in for expected information gain in the planner's β-term, and a baseline to compare EIG against.
3. **Checked planning / abort-on-mismatch.** Execute the plan step by step, compare prediction to observation, and on mismatch abort, log the counterexample and fall back. This maps directly onto the ESS-collapse -> early rejuvenation trigger and the metareasoning gate (a mismatch should hand control to the reflex/safe policy and request the slow mind).
4. **Replay-consistency as a sanity metric:** report the fraction of logged fight transitions where the MAP boss program's top predicted move/timing matches (within tolerance). Good diagnostic for E2 ablations.
5. **Action-efficiency vs humans** as the headline sample-efficiency metric: RHAE-like score min(1.15, (human deaths / agent deaths)²) per boss. Directly applicable to E1 (deaths to competence vs. humans), and gives a bounded, comparable number.
6. **Paired, per-game statistics** (paired difference, d_z, exact sign/permutation test, bootstrap CI on completion rate) for ablations across bosses. Adopt for E2/E6.
7. **Two-tier hypotheses (NL/qualitative then executable).** For the boss model: a coarse "proto" hypothesis (e.g., "there is a hidden phase variable") that is tracked before it is turned into an executable law.

## 6. Structures and models of thought
- **Counterexample-guided revision** = learning from surprise. Each death or mispredicted hit is a counterexample, exactly the "learning from death" story.
- **Ontology learning** (inventing new state variables when outcomes are mixed) is the deterministic analogue of the CRP "new move/new family" step and of latent-cause inference. In MiniSouls, mixed outcomes for identical cues should trigger "split this move into two latent moves" (split/merge move) or "add a hidden phase/combo counter."
- **Hybrid belief:** symbolic counts (Dirichlet) for fast epistemic bookkeeping + programs for prediction + language for not-yet-formalized hunches. Reading the Boss already has the first two; the Dirichlet tables are cheap and could feed the gate (high table entropy = plan / scout).
- **Metacognitive control** via "plan only once verified": a trust gate on the model before using it to act.
