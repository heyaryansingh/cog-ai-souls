# A general learner and player: plan (Nov 26)

Aryan's direction: the goal is a general learner and player, not a Margit specialist. It should be built from distinctive ideas about thought, learning, perception and scaling, and tested on how well it generalises across bosses.

## What already works (Arena3D log B11–B20)
- Realistic perception: noisy, delayed 3D keypoints of the boss; a move becomes identifiable about 12 frames after it starts.
- A boss-program model: the move strings, learned by watching, match the true program (B18). Updating it online recovers from a phase-2 moveset switch within one fight (B19).
- Planning by imagined futures over its own actions: the first wins (B20).
- **Two crutches make this not yet general:** it is given the *true move labels* (only corrupted to 70% accuracy), and it imagines with the *true simulator*.

## Architecture: the "Reader"
Each part is an ablatable module grounded in the project's literature reviews (neuro-architectures A1–A5, Anamnesis v5).

| # | Module | Idea | Replaces |
|---|---|---|---|
| R1 | **Label-free move discovery** | Segment the boss's motion at surprise spikes (event segmentation, A1). Cluster onset-aligned keypoint windows with a Chinese-restaurant-process / DP-means prior, so new moves open new clusters. | true move labels |
| R2 | **Program induction** | Sticky hierarchical Dirichlet grammar over discovered moves, conditioned on distance, phase and *my own last action* (input reading). A surprise spike in the grammar's predictions marks a phase change ("unexpected uncertainty"). | hand-coded context table |
| R3 | **Episodic simulation as the world model** | Each discovered move keeps its observed trajectories and contact outcomes. Imagining = replaying a sampled trajectory, retargeted to the current positions (constructive episodic simulation). | true simulator in MPC |
| R4 | **Dual-process control** | Fast path: a reflex on looming / magnocellular cues for unreadable fast moves. Slow path: MPC over own actions with R2 + R3. Arbitration by the slow path's predictive uncertainty. | MPC only |
| R5 | **Cross-boss schema + gated library** | Move-shape prototypes and grammar statistics shared across bosses as a hierarchical prior. A fragment is promoted into the library only if it improves held-out prediction on other bosses (Go-CLS, A3). | learning each boss from scratch |
| R6 | **Sleep between attempts** | Replay surprising segments to refit R1–R3 (prioritised replay, A2), and recombine library moves into "dream bosses" to rehearse on (A4). | nothing (new) |
| R7 | **Amortised intuition** | A small sequence model meta-trained on episodes sampled from the program prior, used as a fast proposal for R2, and compared head-to-head with the explicit learner. | (new comparison) |

## The generalisation suite
- **Procedural program bosses:** random grammars (strings, branches, delays, enders, projectiles, AoE, phase switches) over held-out-subject mocap moves. Train-boss and test-boss splits.
- **Grounded bosses:** Margit-like, plus Godrick-like, both retimed from public gameplay footage (windup, active and recovery frames annotated per move).
- **Metrics:** deaths to first win, damage curve over attempts, and recovery after a phase switch.
- **Scaling:** how fast a new boss is learned as a function of how many bosses the schema has seen (a learning-efficiency scaling curve).
- **Baselines:** PPO (from scratch and fine-tuned), episodic control, threshold planners, and the oracle ladder (B15).

## Order of work
1. R1 + R2 (drops the label crutch). Test: discovered-move purity, and whether grammar learned from discovered moves still gives the B18 gain.
2. Procedural program-boss generator and the generalisation suite.
3. R3 (drops the simulator crutch). Test: MPC with episodic simulation vs the true simulator.
4. R5 + R6: the cross-boss scaling curve.
5. R4, R7: ablations and the head-to-head.
6. Footage retiming of Margit and Godrick, in parallel throughout.
