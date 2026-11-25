# Formalizing Opponent Modeling with the Rock, Paper, Scissors Game

## 1. Citation and coverage
Erik Brockbank, Edward Vul (UC San Diego). "Formalizing Opponent Modeling with the Rock, Paper, Scissors Game." *Games* 2021, 12(3), 70 (MDPI, special issue "Psychological Perspectives on Simple Games"), published 16 Sept 2021. DOI 10.3390/g12030070. https://mdpi.com/2073-4336/12/3/70
Coverage: the MDPI article page was processed through a web-fetch extraction tool (direct download blocked) in two passes. The main text (intro, framework, results, discussion) was covered; **numeric values that appear only in figures (information gain per dependency, Fig. 4; expected vs observed win differentials, Fig. 5) were not available**, and the reference list was truncated, so the data source (ref. 55; most likely Brockbank & Vul's 2020 CogSci dyad study, unverified) could not be confirmed. Partial read, honestly.

## 2. Core idea and method
- Claim: deviations from Nash random play in RPS reflect adaptive attempts to out-think the opponent, and the real question is which dependencies people *exploit*.
- **Transition coding**: each move relative to the previous one is up (+, beats previous), stay (0), down (-).
- **Dependency hierarchy** (ordered by complexity; higher levels inherit from lower): move base rates -> transition base rates -> move/transition given own previous move -> given opponent's previous move -> given previous outcome (win-stay/lose-shift family) -> higher-order combinations. Eight dependency structures analyzed (Fig. 4); most contexts have 3 values (R/P/S or W/T/L).
- **Information gain** I(A|C) = log2(3) - H(A|C) bits (conditional entropy averaged over contexts).
- **Corrected information gain I***: subtract information already carried by parent (simpler) dependencies, so a complex dependency is not credited for a simpler one it implies.
- **Expected win count differential E[v|C]**: the per-round advantage an opponent would gain by optimally exploiting the dependency (best response to each context's distribution). Captures exploitability, not just entropy (two equal-entropy distributions can give 0.5 vs 0.3 per-round advantage).
- **Corrected regression**: observed dyad win differentials regressed on expected differentials; for higher-level dependencies, residuals after regressing on lower levels are used.

## 3. Data and results
- Reanalysis: 116 participants in 58 stable human dyads, 300 rounds each. Win differentials larger than chance in the original study.
- People *exhibit* dependencies at all levels; information gain grows with complexity, but after correction complex dependencies carry much less unique information.
- Only two dependencies predict actual exploitation: transition base rates (beta = 0.19, p = 0.027) and player's choice given the opponent's previous move (beta = 0.45, p = 0.015). Complex/outcome-based dependencies are present but not exploited over 300 rounds.
- Conclusion: a fundamental limitation of human adversarial reasoning; memory limits suggested (untested).

## 4. Failure modes and limitations
Stated/evident: reanalysis of one dataset; correction method determines attribution; small effects with 58 dyads; memory explanation untested; RPS-specific.
Noticed: human-vs-human dyads mean both players adapt (non-stationary target), conflating "can't learn" with "target moved"; this motivates the later bot study (Brockbank & Vul 2024, used by cross2025-rps.md) with stationary 90%-rule bots; no learning curves reported in the text.

## 5. Ideas to steal
- **Exploitability-weighted metric for boss patterns**: define, per boss law/context, the expected damage differential an optimal responder gains by knowing it (analogue of E[v|C]). Use it to (a) rank difficulty axes, (b) score what the agent has *learned* by regressing its realized advantage per context on the available advantage, as they do. That gives "fraction of available exploitability captured" per dependency order, for agent and humans in E4.
- **Hierarchical correction**: when measuring whether the agent/human exploits a second-order law, residualize against the first-order and base-rate exploitation, so a second-order success is not just a first-order bias.
- **Information gain in bits per context** as a design-time measure of how much a boss's history predicts its next move; report the entropy budget of each test boss (T1-T4) so difficulty is quantified, not asserted.
- **Prediction for E4**: humans will exploit base-rate move biases and responses to the player's last action, but not outcome-conditioned or multi-step laws, even with many attempts. The agent with explicit higher-order laws should exploit them; a memory-limited variant (truncated history window) should reproduce the human ceiling. This tests the memory-limit hypothesis that the paper raises but does not test.
- Use stationary scripted bosses (not adaptive) for the main human comparison; reserve adaptive bosses for a separate condition.

## 6. Structures and models of thought
- **Dependency hierarchy / context-conditioned distributions**: a natural prior over boss laws ordered by context arity (base rate < own transition < player-conditioned < outcome-conditioned < combined), i.e., a complexity prior.
- **Exploitability rather than predictability** as the quantity that matters for an agent: aligns posterior use with decision value (value of information).
- **Bounded memory** as the candidate cause of human limits: implement as a history-window parameter in a human-like agent variant.
