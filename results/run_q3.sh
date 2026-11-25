python scripts/learn_curve.py --split T2 --n_test 30 --bosses 30 --attempts 8 --mode oracle --procs 3 > results/log_T2_oracle.txt 2>&1
python scripts/learn_curve.py --split T2 --n_test 30 --bosses 30 --attempts 8 --mode frozen --procs 3 > results/log_T2_frozen.txt 2>&1
python scripts/learn_curve.py --split T3 --n_test 30 --bosses 30 --attempts 8 --mode frozen --procs 3 > results/log_T3_frozen.txt 2>&1
