# Assessing Adaptive World Models in Machines with Novel Games (Ying et al., 2025)

## 1. Citation and reading status
- **Title:** Assessing Adaptive World Models in Machines with Novel Games
- **Authors:** Lance Ying, Katherine M. Collins, Prafull Sharma, Cédric Colas, Kaiya Ivy Zhao, Adrian Weller, Zenna Tavares, Phillip Isola, Samuel J. Gershman, Jacob D. Andreas, Thomas L. Griffiths, François Chollet, Kelsey R. Allen, Joshua B. Tenenbaum
- **Year and venue:** 2025, arXiv:2507.12821v2 (cs.AI). It is a Perspective paper; no journal is named on the HTML page.
- **Link:** https://arxiv.org/abs/2507.12821
- **What I read:** The full arXiv HTML, in two passes through a fetch tool that summarizes pages, so I saw it in paraphrase, not verbatim. The paper has **no new experiments**. Figures (including the Tsividis et al. learning curves) were described only qualitatively.

## 2. Core idea
- **World-model induction.** Human adaptability comes from rapidly *building and refining* internal world models (rules, objects, causal structure) through interaction. AI evaluation mostly tests static models learned offline.
- **Formal frame.** Hierarchical Bayesian updating: P(Ω|e) ∝ P(e|Ω)P(Ω), with instance-level models Ω₁ constrained by abstract models Ω₂, Ω₃ and so on (Fig. 1). This is learning-to-learn across games.
- **Agents as scientists.** Agents should run hypothesis-driven, "child as scientist" exploration and **plan to learn**, with epistemic goals.
- **Three capabilities to test:**
  1. rapid learning in a new domain;
  2. robust within-domain generalization (new levels, modified rules, changed goals, rules that change mid-play);
  3. cross-domain transfer and meta-learning, for example reusing intuitive physics.
- **Critique of existing benchmarks.** Atari, Go and StarCraft rely on massive training in fixed environments. Procedural or parametric generalization tests measure a *performance change*, not the *adaptation process*, and use simple DSLs.
- **Three desiderata:**
  1. **Structural novelty.** Latent mechanics, objects and objectives must be inferred by experimenting. Games change over time. Abstract principles must transfer.
  2. **Intuitive and learnable for average humans.** This makes human benchmarking possible and constrains generation.
  3. **Diversity** of world-model types (spatial, physical, social/ToM) and of learning channels (pure interaction, instructions, demonstrations, NPCs).
- **Continual generation.** Novelty wears off, so benchmarks should sample continually from a **compositional generative process over games**. Variants change the environment, the number of players, mechanics or objectives, or combine games (Fig. 4).

## 3. Proposed evaluation (Section 4.2), with no formulas given
- **(a) Sample efficiency under a trial budget.** Vary the number of allowed attempts. Report performance after k attempts, or the **number of plays needed to reach average human performance** (following Lake et al. 2017). Plot success against learning samples for humans and models on the same axis (as in Tsividis et al. 2021).
- **(b) Qualitative analysis of exploration.** Is exploration targeted (humans), structured but less efficient (EMPA), or diffuse (DDQN)? This separates understanding the rules from optimizing reward.
- **(c) Probing internal models.**
  - Program-synthesis agents can be read directly.
  - Neural networks can be probed through layers or representations.
  - Foundation models can be asked targeted questions at several levels of abstraction.
  - In all cases, track how the representation *changes during interaction*.
- **Case studies:**
  - **ARC-AGI-3:** no instructions; goals and rules are inferred through interaction.
  - **AutumnBench:** two phases.
    - An *interaction phase*: free exploration with no reward.
    - A *test phase*: masked-frame prediction, defect (anomaly) detection, and planning.
  - **VGDL games (Tsividis et al. 2021):** humans learn in minutes. Humans are more targeted than EMPA, and EMPA more than DDQN.
  - **Virtual Tools (Allen et al. 2020).**
- **Validating games with humans.** Collect engagement, playability and difficulty ratings plus learning curves, and check that average humans can solve the tasks (as ARC-AGI does).
- **Open problems (Discussion):**
  - Is a hierarchical world model *necessary*, or can model-free approaches suffice?
  - How can latent-rule inference be measured beyond task success? Candidates are probes, counterfactual evaluation, and representation analysis.
  - External validity.

## 4. Limitations
- **Reported by the authors:** no empirical results; metrics for internal-model quality are underdeveloped; the necessity of world models is an open question.
- **What I notice:**
  - Sample efficiency is defined only loosely.
  - Nothing addresses how to separate perception or motor failures from world-model failures in real-time games.
  - Fast motor play (souls-like combat) is not discussed. Their examples are turn-based or physics-puzzle games, so our real-time adversary setting is an extension.

## 5. Ideas to steal

1. **The AutumnBench two-phase protocol, adapted to bosses.** Add an optional **observation phase** in which the boss attacks a dummy, or the agent may only dodge and never attack. Follow it with a **test phase** of three probe tasks that measure the model independently of combat skill:
   - (i) **masked-move prediction:** given a cue sequence with a gap, predict the missing move and its duration (frame prediction);
   - (ii) **anomaly detection:** show a fight clip where the boss breaks one of its own laws (for example an impossible transition or wrong timing) and ask the agent to flag it; score the agent's surprise or posterior predictive p-value. This is a direct test that a *law* was induced;
   - (iii) **planning:** win.

   This turns the plan's "induction speed" metric into a battery of tests.
2. **Trials-to-human-level.** For each boss, the number of attempts until the agent's rolling win rate (or HP-removed) reaches the **human median at the same attempt index**. Add it to §5 next to deaths-to-first-win.
3. **Learning curves on human-aligned axes.** Plot win rate against attempts (and against game-seconds of experience) for humans and agents on the same plot, for each split T1 to T4.
4. **Exploration-pattern analysis.** Compare *where* the agent spends its information gathering. The prediction is that our scouting is targeted at high-uncertainty moves. Metrics:
   - the fraction of early-attempt actions that are information-seeking (keeping distance, baiting a specific move);
   - the expected information gain per attempt, computable from our posterior, compared with humans' revealed bait behaviour.
5. **Mid-play rule changes as a test split.** The paper explicitly wants rules that change mid-game. Add a **T5 or T4b split**: a phase change or a self-rule change (shorter i-frames, a new stamina cost) *within* an attempt, unannounced. Measure recovery time in attempts or ticks after the change, from ESS collapse to re-convergence.
6. **Necessity test.** Their open question ("are world models necessary?") is exactly our E6 amortization-gap experiment, and should be framed as an answer to it in the write-up.
7. **Validate bosses with humans.** Before the main study, check with a few humans that every T1 to T4 boss is learnable within about 20 attempts (desideratum 2). Drop bosses humans cannot learn, or report them separately.

## 6. Structures and models of thought
- **A hierarchical Bayesian world model** (Ω₁ for each game, constrained by abstract Ω₂). This maps one-to-one onto our design: Ω₁ is the boss program, Ω₂ is Θ plus the CRP family library, and Ω₃ could be the self-model of game rules shared across "souls-likes".
- **Programs as world models** (EMPA, AutumnSynth, Das et al.) can be inspected directly. Our boss programs are inspectable too, so model quality can be scored by comparing the inferred program to the true one, for example the edit distance between programs or the KL divergence between the predictive distributions over hit times.
- **Planning to learn / epistemic action.** The agent should choose actions for information gain: our scouting term and the planner's value of information.
- **Child as scientist.** Hypothesis testing through targeted experiments. Here that means baiting specific moves to disambiguate competing boss hypotheses with different predictions. A concrete implementation: choose the action that maximizes the expected KL divergence between the top-2 SMC particles' predictions.
