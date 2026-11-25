"""Code every hit the oracle takes: which move, what the player was doing, stamina, and how much
belief the filter had on the true move at impact. Tells us whether failures are belief, timing or acting."""
import sys
from collections import Counter
import numpy as np
sys.path.insert(0, ".")
from rtb.boss import make_splits
from rtb.env import MiniSouls, ACTION_NAMES, ACTIVE
from rtb.policies import OraclePlanner

splits = make_splits(0)
cats, deaths = Counter(), Counter()
for i, boss in enumerate(splits["train"][:16]):
    env = MiniSouls(boss, seed=1000 + 37 * i); obs = env.reset()
    pol = OraclePlanner(12, 45, seed=i); pol.reset(env)
    while True:
        a = pol.act(obs, env) if obs["free"] else 0
        hp, st = env.player.hp, env.player.stamina
        k, stage, j = env.bs.k, env.bs.stage, env.bs.j
        belief_true = pol.filter.move_marginal()[k]
        pa = ACTION_NAMES[a] if obs["free"] else ACTION_NAMES[env.player.action]
        tia = env.player.t_in_action
        obs, _, done, _ = env.step(a); pol.observe(obs)
        if env.player.hp < hp:
            prim = boss.moves[k].primitive.name
            if pa == "dodge":
                why = "dodge mistimed"
            elif pa in ("light", "heavy", "heal"):
                why = f"caught in {pa}"
            elif pa == "block":
                why = "blocked (chip/guard break)" if st > 0 else "guard break"
            elif st < env.R.dodge_cost:
                why = "no stamina to dodge"
            else:
                why = f"{pa}: no defensive action"
            belief = "belief>0.5" if belief_true > 0.5 else "belief<=0.5"
            cats[(why, belief)] += 1
        if done:
            if env.outcome == "death":
                deaths[boss.moves[k].primitive.name] += 1
            break
for (why, b), n in cats.most_common():
    print(f"{n:4d}  {why:30s} {b}")
print("killing moves:", deaths.most_common())
