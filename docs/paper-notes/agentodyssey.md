# AgentOdyssey

## 1. Citation and reading status
- **Title:** AgentOdyssey: Open-Ended Long-Horizon Text Game Generation for Test-Time Continual Learning Agents
- **Authors:** Zheyuan Zhang, Zehao Wen, Alvin Zhang, Andrew Wang, Jianwen Xie, Daniel Khashabi, Tianmin Shu (JHU and Lambda)
- **Year and venue:** 2026, NeurIPS 2026 (according to the project page; the arXiv HTML names no venue)
- **Links:** https://arxiv.org/abs/2606.24893 · project page https://agentodyssey.github.io/ · docs https://agentodyssey.github.io/docs · code https://github.com/agentodyssey/agentodyssey
- **What I read:** The arXiv HTML full text, in three targeted passes through a fetch tool that summarizes pages, so I saw it in paraphrase, not verbatim. The main body and Appendix A1 to A3.2 are covered. The HTML cut off partway through A3.3 (entity generation). I did not see A3.3 onward (A4 to A9: generation pipeline details, prompts, SFT/RL hyperparameters, case studies). I also read the docs pages "Getting started", "Evaluation Metrics" and "Agent Interface". Note that Tianmin Shu, the course instructor, is the last author.

## 2. Core idea and method
- **The benchmark.** It tests **test-time continual learning**: learning and acting are interleaved within one long deployment of up to 500 steps. There is no separate train/test phase.
- **The environment.** Each game is a POMDP text RPG:
  - a world graph of areas, objects and NPCs;
  - 10 simulated minutes pass per step;
  - **action rules** (effects the player triggers) and **step rules** (environment dynamics such as a day-night cycle and stochastic NPC spawns);
  - main quests are strictly sequential; side quests are unordered;
  - 17 core verbs, which generated rules can add to or edit.
- **Generation.** An LLM writes new entities, rules and quests as code (program synthesis), conditioned on a human-written base game. Each game then goes through static checks, execution, automated agent playtesting, and a repair loop. Games that pretrained LLMs already answer well on World Knowledge QA are filtered out as contaminated (the threshold is not given).
- **The five target abilities:** exploration, episodic memory, world-knowledge acquisition, skill learning, and long-horizon planning.
- **Agent paradigms tested:**
  - Long Context;
  - Fixed-size memory (STM, MEM1);
  - RAG (Vanilla, Mem0, Raptor, Voyager);
  - SFT (LoRA);
  - RL (PPO with LoRA);
  - Latent memory (MemoryLLM, MPlus).

  Optional add-ons are reflection, summarization and STM. Backbones are GPT-5, GPT-5-mini, Qwen3-4B, LLaMA-3 variants and Claude-Opus-4.6.
- **Agent API.** A subclass of `Agent` overrides `_act(obs) -> action string`. `Action(name, verb, params, description)`, and `available_actions` is recomputed from the rules module.

## 3. Experiments, metrics and numbers

### Metrics (exact)
- **Main reward R^main_t:** the cumulative number of completed main-quest stages.
- **Supplementary reward R^sup_t:** the cumulative sum of side-quest (SQ), exploration (E, new areas), craft (C, *unique* successful crafts) and defeat (D, *unique* NPC types defeated) rewards.
- **Combined progress:** R^comb_t = ½(R^main_t / M_main + R^sup_t / M_sup), where M_main = max over runs and t of R^main_t, and likewise for M_sup, within each game. It lies in [0, 1].
- **Ranking:** by main reward, with supplementary reward breaking ties.
- **World Knowledge QA (WK):**
  - multiple choice, in 7 categories: crafting ingredients, ingredient quantities, NPC inventories, object distributions, spatial connections, action-rule mechanics, step-rule mechanics;
  - ground truth comes from rules over the world definition; for rule mechanics an LLM writes the questions from the rule source code;
  - distractors are roughly 50/50 LLM-proposed plausible fakes and real but wrong entities;
  - items near spawn are oversampled;
  - asked **before gameplay (WK.b) and after (WK.a)**, and in some figures every 100 steps.
- **Episodic Memory QA (Epi):**
  - multiple choice, generated purely by rule from the agent's own log;
  - question types are **sequence completion** ("given these 5 consecutive actions, what came next") and **location recall** ("where did you drop X");
  - distractors are entities the agent never interacted with;
  - 20 samples per category.
- **Coverage:**
  - Object Exploration (OE) is the fraction of objects acquired;
  - Action Exploration (AE) is the fraction of action types executed at least once.
- **Action Diversity:** AD = −Σ_i p_i log p_i / log N, the normalized entropy over N action types. It uses 100-action windows for curves and the full post-tutorial history for tables.
- **Invalid-action rate:** counts parse and format errors only. State-dependent failures are excluded, and malformed JSON becomes a wait action.
- **Cost:** tokens per step and total tokens.
- **QA validation:** 30 sampled QA items were hand-checked and all 30 were correct.

### Protocol
- Runs last up to 500 steps after a tutorial.
- The docs give a default seed of 42 and a default of 300 steps.
- There is essentially one run per configuration; only GPT-5 RAG on Remnant was repeated (3 runs).
- **Human baseline:** one human playtest of 500 steps per game. Instructions to the tester are not described.

### Table 2: completed / total main stages (supplementary reward in parentheses)

| Agent | Remnant | Mark | Metropolis | Saltglass | Quarantine | Robot Kingdom | Mean |
|---|---|---|---|---|---|---|---|
| Long Context GPT-5 | 3/24 (13) | 17/17 (20) | 9/21 (14) | 6/8 (26) | 13/28 (22) | 2/21 (13) | 47.7% |
| RAG+STM GPT-5 | 2/24 (15) | 11/17 (31) | 8/21 (17) | 4/8 (10) | 4/28 (13) | 1/21 (5) | 30.0% |
| Fixed STM GPT-5 | 1/24 (9) | 11/17 (22) | 4/21 (9) | 3/8 (12) | 4/28 (4) | 1/21 (8) | 24.1% |
| SFT+STM Qwen3-4B | 1/24 (6) | 7/17 (16) | 7/21 (6) | 2/8 (8) | 2/28 (8) | 1/21 (8) | 19.3% |

- **Human:**
  - Remnant: Q 9/24, SQ 8, E 10, C 11, D 11.
  - Mark: Q 17/17, E 10, C 10, D 8.
- **Long Context GPT-5 on Remnant:**
  - WK rises from 0.365 before to 0.713 after;
  - Epi 0.918;
  - about 51.9k tokens per step.
- **No-memory GPT-5:** WK.a 0.322, Epi 0.208.
- **STM size sweep (0, 1, 5, 10, 20, 500):**
  - Q = 0, 0, 1, 1, 1, 3;
  - Epi = 0.208, 0.369, 0.587, 0.366, 0.376, 0.918, which is *non-monotonic*.
- **Three GPT-5 RAG runs:** Q = 1, 1, 1; WK.a = 0.487, 0.452, 0.513; Epi = 0.709, 0.614, 0.840. Episodic memory varies a lot from run to run.
- **Vanilla SFT (Qwen3-4B, Mark):** WK.a falls to 0.036 and Epi to 0.051, which the authors read as catastrophic forgetting.
- **Latent-memory agents** (MemoryLLM, MPlus) produced 100% invalid actions.
- **Cost:** Long Context tokens grow roughly quadratically with steps (about 51.6k per step against 2.5–3.4k for the other agents). About 135M tokens were spent on 16 runs.
- **Diagnostics track reward.** WK, Epi, coverage and AD correlate strongly with cumulative reward, and **falling AD coincides with reward plateaus**.

## 4. Failure modes

### Reported by the authors (qualitative taxonomy, no frequencies), by ability
- **Exploration:** narrow object and action coverage; ignoring intermediate crafting ingredients.
- **Episodic memory:** repetitive loops; poor recovery from failure; hallucinated object locations.
- **World knowledge:** hallucinated recipes; failing to update beliefs.
- **Skill learning:** no procedural adaptation, such as never learning to fight better or to write recipe notes.
- **Planning:** losing the main goal after finishing subgoals; poor switching between goals.
- **Cost:** reflection and summarization did not help reasoning models, and reflection used up the context faster.

The authors also list text-only observations, a single agent, and turn-based fixed action durations as limitations.

### What I notice
- n = 1 human and about 1 run per cell, so there are no confidence intervals.
- Max-over-runs normalization makes combined progress depend on which agents are in the comparison set.
- The taxonomy is not coded per trajectory, so it gives no rates.
- "Before" WK is partly a prior-knowledge test.

## 5. Ideas to steal

1. **Model-probe QA before, during and after each boss.** This is AgentOdyssey's WK.b → WK.a, done directly on our posterior instead of through questions. At attempts 0, 1, 2, 5, 10 and 20, score the agent's answers to *templated, rule-generated* probe questions about the boss program, with ground truth read from the generator. Example probes:
   - "Which move follows a 3-hit combo?"
   - "Is there a delayed variant of move k?"
   - "What is the wind-up duration of move k (±2 ticks)?"
   - "Does the boss switch phase below 50% HP?"
   - "Which family is this boss?"

   Report a **knowledge-gain curve** separately from the win curve. Generate distractors from real primitives that the boss *does not* use, the analogue of their "real but wrong entity" distractors. Baselines without an explicit model (PPO, AD) can be probed with linear probes on hidden state.
2. **Episodic probe.** After each attempt, ask "what killed you last attempt, and at what boss HP?" and "what move came after X in attempt 3?". This tests whether the lifelong memory and fight log are used, not just stored. It matters for the LLM-with-memory baseline.
3. **Separate main and supplementary progress.** Main progress is a win, or boss HP removed. Supplementary rewards count *unique* discoveries:
   - distinct moves the agent has seen and correctly labelled (move coverage, an analogue of OE);
   - distinct own actions it has tried (AE);
   - distinct counter-strategies that succeeded (an analogue of C and D).

   Use the combined form ½(main/M_main + sup/M_sup) only as a secondary score, and normalize by the **human or oracle max, not the max over runs**, so scores do not shift when baselines are added.
4. **Action-diversity entropy over a sliding window** of N own actions, as a cheap way to detect when exploration collapses. Prediction: AD collapse signals that PPO or LLM baselines have plateaued, while our gate should keep AD high on novel bosses (T3/T4) and let it fall once the posterior concentrates. Plot AD against posterior entropy.
5. **Invalid-action rate** for the LLM baseline: parse errors only, with malformed output mapped to a no-op.
6. **Cost reporting as a curve.** Report cumulative compute against attempt number. Long Context is quadratic; our SMC is linear in attempts if fight logs are summarized by sufficient statistics. This strengthens the "compute-normalized score" metric in §5 of the plan.
7. **A failure taxonomy organized by capability,** coded per death, adapted from their five abilities (exploration, memory, knowledge, skill, planning). Proposed death codes:
   - (a) unseen-move death (exploration and perception);
   - (b) seen-but-mispredicted timing (world knowledge);
   - (c) correct prediction but wrong or late action (skill or reflex);
   - (d) resource mismanagement, i.e. stamina or heal (planning);
   - (e) repeated identical death (episodic memory failure).

   Unlike AgentOdyssey, *report frequencies*. Our explicit posterior lets us tell (b) apart from (c) automatically, by checking whether the predicted hit-time distribution covered the true hit.
8. **Contamination filter.** Run the probe QA on a fresh agent at attempt 0. Bosses whose properties are answerable *a priori* (for example by the LLM baseline from souls-like lore) should be flagged or regenerated.
9. **Generate–playtest–repair loop for boss programs.** Run the oracle agent on every generated boss and reject bosses that are unwinnable, or trivially winnable by a random policy, before they enter T1–T4. This is the same role as their automated agent playtesting.

## 6. Structures and models of thought
- **Rules as executable programs.** Action rules and step rules are explicit code, so world knowledge equals the rule program. This mirrors our boss programs and self-model rules, and it supports *program-level* ground truth for probes.
- **Memory paradigms as hypotheses about cognition.** Long context is perfect episodic memory at quadratic cost. STM is working memory. RAG is cued retrieval. SFT is consolidation into weights, which caused forgetting. Their result that SFT forgets and STM helps every paradigm argues for our split:
  - a short-horizon working belief (the HSMM filter state);
  - a compressed episodic store (fight logs as sufficient statistics);
  - slow consolidation into Θ and the CRP family library *between* bosses only, never into weights at test time. This also avoids catastrophic forgetting.
- **Exploration collapse as a diagnostic of no epistemic drive.** LLM agents lack directed information gathering. Our planner's scouting and information-gain term is what should prevent the AD collapse.
- **Episodic memory is not the same as knowledge.** High Epi (0.918) with only 3/24 progress shows that remembering is not the same as inducing rules. Our evaluation should likewise separate log recall from model induction.
