python scripts/learn_curve.py --split T3 --n_test 80 --bosses 80 --attempts 4 --mode mixture --tag _n80 > results/log_T3_n80_mixture.txt 2>&1
python scripts/learn_curve.py --split T3 --n_test 80 --bosses 80 --attempts 4 --mode netsup --tag _n80 > results/log_T3_n80_netsup.txt 2>&1
RTB_POOL_LAW_COUNTS=1 python scripts/learn_curve.py --bosses 40 --attempts 6 --tag _poollaw > results/log_poollaw.txt 2>&1
