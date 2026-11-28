# Research plan v3 (Nov 28): a faithful souls fight and a player built from first principles

Aryan's direction (Nov 28): make the simulation as close to a real souls fight as possible, then rethink the
player from underlying principles (Bayesian theory of mind, etc.), with hypotheses, math and experiments, toward a
generally capable fighter with lots of freedom.

Where we stand (B77): with ground-truth boss state the mirror agent beat Godrick in 53–58% of fights; with honest
perception and input lag it wins 0–6%. Honest perception alone leaves 11% wins, unmodelled input lag alone 0%.
The agent's wins came from frame-exact timing that no player has.

---------------------------------------------------------------------------------------------------------------

## Part 1. Realism v2 (`ARENA_REAL=2`): mechanics measured from Elden Ring

Sources: eldenring.wiki.gg (Invincibility Frames, Poise), fextralife (Stamina), Bandai Namco's official combat
guide. Our simulator runs at ~30 fps, the same baseline the wiki uses for frame counts. No game assets are copied:
only timings and rules.

| mechanic | Elden Ring (source) | simulator before | realism v2 |
|---|---|---|---|
| roll i-frames | 14 frames at light/medium load, 13 heavy (wiki, 30 fps) | 10 (frames 2–11) | **14 (frames 1–14)** |
| backstep | short hop, 7–8 i-frames only with a talisman, "minor i-frames" | none | **backstep: 10 frames, 0 i-frames, cheap** |
| input buffering | presses are queued for "a short time" and fire at the next chance (official guide) | presses while busy are dropped | **buffer: a press while busy is kept up to 10 frames** |
| player hit reaction | most enemy attacks deal 50/100 poise damage, player poise breakpoints 51/101: an unarmoured player staggers when hit | only attack/heal interrupted | **hitstun 14 frames (can't act) after every hit** |
| boss stance (poise) | bosses 80–120; regen 13/s after a delay of poise/13 s (wiki) | none | **stance 100 (scaled per boss), same regen rule** |
| stance break | enemy stunned "a significant amount of time", critical hit available (wiki, guide) | none | **stun 100 frames; a light attack from the front within 2 m = critical, 3.5× damage, ends the stun** |
| heavy attack | slower, more stance damage; charged heavy / jump heavy / guard counter "large stance damage" (guide) | none | **heavy: 34 frames, 1.6× damage, 2.5× stance damage, 22 stamina** |
| guard counter | heavy attack right after blocking (guide) | none | **heavy within 20 frames of a block: fast (24 frames), 1.4× damage, 3.5× stance** |
| guard at 0 stamina | "leaves you open to a critical attack" (guide) | -10 HP | **guard break: 40-frame stagger** (already in B74's realistic guard) |
| chip damage through a medium shield | not 100% negation | 0 | **25%** (B74) |
| stamina | max 80–170 by Endurance; regen and costs not published on these pages | 100, roll 18, light 12, 45/s | unchanged, flagged as unsourced |
| input latency | not measured on these pages; typical console games 60–130 ms total | 0 or 4–8 frames | **sweep 0–400 ms (B80); headline at ~100 ms (3 frames) + 3-frame perception** |

What these change strategically. Stance + critical turns the fight from "chip damage between swings" into the real
souls loop: read, dodge, punish with heavies or guard counters, break stance, riposte. Hitstun makes strings
lethal (one mistake costs a chain of hits), which is what makes Margit/Godrick feel the way they do. Buffering
removes the need for frame-exact presses (a human presses a little early and the game holds it), and it interacts
with lag.

---------------------------------------------------------------------------------------------------------------

## Part 2. The player from first principles

### 2.1 The fight as a POMDP with an intentional opponent
Hidden state `x_t = (m_t, τ_t, ψ_t, σ_t)`: the boss's current move `m`, the frame within it `τ`, its intent `ψ`
(the string it is in and the move it will choose next) and its stance `σ`. Observations `o_t` are poses seen
`L` frames late with noise. My own action `a_t` takes effect after an input delay `D ~ p_D`.

The boss is an *agent*: it chooses moves by a policy that depends on me,

    π_B(m_{t+1} | m_t, d_t, u_t, φ_t)          d = distance, u = what I am doing (healing, recovering, rolling),
                                              φ = phase.

That is the theory-of-mind part. A souls boss is a simple mind (distance-gated strings, punishes heals and
recovery, delays swings against early rolls), but it *is* a mind, and its choices are caused by mine.

### 2.2 Three hypotheses

**H1 (timing under uncertainty is the bottleneck, and it has a closed form).** For a dodge pressed at time `t₀`,
with input delay `D`, i-frames `[i₀, i₁]` (frames 1–14) and the blade's contact time `T` (from the belief):

    P(dodge) = Σ_d p_D(d) · P(t₀ + d + i₀ ≤ T ≤ t₀ + d + i₁) = (p_D ⋆ 1_[i₀,i₁]) ⋆ p_T  evaluated at t₀.

The best press time maximises this. With 14 i-frames and a 5-frame spread in `D`, a dodge succeeds with
certainty only if the contact-time posterior has spread ≤ 14 − 5 = 9 frames, so **the planner must reduce timing
uncertainty below ~9 frames (300 ms), or use actions whose safety doesn't depend on timing (spacing, backing out
of range, guarding)**. Prediction: wins under lag rise once the planner (a) convolves with `p_D` explicitly and
(b) prefers distance when the contact-time posterior is wide. Test: the B80 latency sweep, and logging the
contact-time spread at each dodge.

**H2 (inverse planning beats passive recognition).** Today the agent only recognises the move in progress. A
Bayesian observer of an agent can also predict the *next* move from what it can see the agent wants: Godrick's
choice depends on distance and on whether I am healing or recovering. Learn `π_B` as Dirichlet counts over
(previous move, distance bin, my-action class) with a hierarchical prior across bosses (bosses share the rule
"punish heals"), and use it (i) for the next move in imagination and (ii) **to choose my behaviour to shape the
boss's choice (baiting)**: stand where the boss's likely opener is one I can punish. Prediction: higher hit
rate after the opener, and the agent's position distribution moves to "bait" ranges. Test: ablate the action
conditioning (counts over distance only).

**H3 (the value of a hit is non-linear: stance is a hidden resource I control).** With stance, the payoff of
landing hits is `Σ dmg + P(break) · critical`; the agent must track the boss's stance (hidden, but inferable
from my own hits and time since, via the published regen rule) and push for the break when close. Prediction:
heavies and guard counters become frequent near the break, ripostes appear, time-to-win drops. Test: belief over
stance on/off.

### 2.3 Architecture (v3 player)
1. **Perception**: delayed noisy pose → episodic recognition (mirror memory) → posterior over (move, frame);
   unchanged, it measured near its ceiling under honest perception (B77 check).
2. **Boss mind model**: `π_B` with action conditioning (H2); reactive behaviours learned as counts, not given.
3. **Timing posterior**: from the imagined futures, the empirical distribution of contact time `T`; convolved
   with the agent's own measured input-delay distribution (it measures its press-to-motion delays) and the
   i-frame window: a closed-form dodge-success term per candidate press time (H1).
4. **Resource beliefs**: boss stance (H3), own stamina and flasks.
5. **Planner**: two-step MPC over imagined futures (re-enactment), with the timing term added to the value and
   uncertainty-aware: value = E[damage dealt + critical·P(break)] − 2.5·E[damage taken], futures sampled from the
   belief, each with its own input delay.

### 2.4 Experiments (in order)
- R1. Build realism v2; unit-check every rule against the table (frame-by-frame traces).
- R2. Baselines under realism v2 + honest mode (~100 ms input + 100 ms perception): mirror planner as is.
- R3. H1: timing term + delay convolution; latency sweep with and without it.
- R4. H3: stance belief and critical in the value; heavy/guard counter in the action set.
- R5. H2: action-conditioned boss mind model + baiting; ablation.
- R6. Freedom: open arena with obstacles, sprint, more actions; generality on the procedural suite, then a frozen
  test on a fresh lockbox of bosses. Wins as the headline, IQM + CIs, Holm.
