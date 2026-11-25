python scripts/learn_curve.py --split T3 --n_test 80 --bosses 80 --attempts 4 --mode online_netsup --tag _n80 > results/log_T3_n80_online_netsup.txt 2>&1
python scripts/learn_curve.py --split T3 --n_test 80 --bosses 80 --attempts 4 --mode oracle --tag _n80 > results/log_T3_n80_oracle.txt 2>&1
