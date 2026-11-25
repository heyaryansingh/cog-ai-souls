python scripts/learn_curve.py --split T3 --n_test 30 --bosses 30 --attempts 4 --mode netmix --procs 3 > results/log_T3_netmix.txt 2>&1
python scripts/learn_curve.py --bosses 40 --attempts 4 --mode netmix --procs 3 > results/log_train_netmix.txt 2>&1
