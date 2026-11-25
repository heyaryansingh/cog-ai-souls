"""Arena3D: a 3D, motion-capture-driven souls-like world (chapter 3).

Bodies are full CMU humanoids (31 bodies, 56 joints) in MuJoCo. The boss is driven by real motion
capture and carries a weapon; harm is decided by per-substep 3D contact between the weapon / boss
body volumes and the player's body-part capsules (no hand-made hitboxes).
"""
