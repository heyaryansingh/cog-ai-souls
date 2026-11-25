cd /home/user/cog-ai-souls
python scripts/train_reflex_priv.py --init checkpoints/reflex48_it4.pt --iters 5 --attempts 3 --teacher 48 --tag priv > results/log_priv.txt 2>&1
P=checkpoints/priv_it2.pt+checkpoints/priv_it3.pt+checkpoints/priv_it4.pt
RTB_REFLEX=$P python scripts/learn_curve.py --split T1 --bosses 80 --n_test 80 --attempts 3 --mode mixture --tag _n80_priv > results/log_T1_priv.txt 2>&1
RTB_REFLEX=$P python scripts/learn_curve.py --split T3 --bosses 80 --n_test 80 --attempts 4 --mode netsup --tag _n80_priv > results/log_T3_priv.txt 2>&1
RTB_REFLEX=$P RTB_BLEND=3 python scripts/learn_curve.py --split T1 --bosses 80 --n_test 80 --attempts 3 --mode mixture --tag _n80_priv > results/log_T1_priv_blend3.txt 2>&1
RTB_REFLEX=$P RTB_BLEND=3 python scripts/learn_curve.py --split T3 --bosses 80 --n_test 80 --attempts 4 --mode netsup --tag _n80_priv > results/log_T3_priv_blend3.txt 2>&1
