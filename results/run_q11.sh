python scripts/learn_curve.py --split T1 --n_test 80 --bosses 80 --attempts 3 --mode mixture --tag _n80 > results/log_T1_n80_mixture.txt 2>&1
python scripts/learn_curve.py --split T1 --n_test 80 --bosses 80 --attempts 3 --mode online --tag _n80 > results/log_T1_n80_online.txt 2>&1
