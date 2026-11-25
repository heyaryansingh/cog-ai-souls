cd /home/user/cog-ai-souls
TAG=_ens python scripts/reflex_eval.py train 40 6 checkpoints/reflex48_it2.pt+checkpoints/reflex48_it4.pt+checkpoints/reflex48_it5.pt --planners 12 > results/log_reflex_eval_ens.txt 2>&1
TAG=_gate python scripts/reflex_eval.py train 40 6 checkpoints/reflex48_it2.pt+checkpoints/reflex48_it4.pt+checkpoints/reflex48_it5.pt --planners 12 --gate 0.5 > results/log_reflex_eval_gate.txt 2>&1
TAG=_T1 python scripts/reflex_eval.py T1 80 2 checkpoints/reflex48_it2.pt+checkpoints/reflex48_it4.pt+checkpoints/reflex48_it5.pt --planners 12,48 > results/log_reflex_eval_T1.txt 2>&1
RTB_REFLEX=checkpoints/reflex48_it2.pt+checkpoints/reflex48_it4.pt+checkpoints/reflex48_it5.pt python scripts/learn_curve.py --split T3 --bosses 80 --n_test 80 --attempts 4 --mode netsup --tag _n80 > results/log_T3_netsup_reflex.txt 2>&1
