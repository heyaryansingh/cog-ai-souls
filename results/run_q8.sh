python scripts/swap_diag.py > results/log_swap_diag.txt 2>&1
python scripts/learn_curve.py --split T3 --n_test 30 --bosses 30 --attempts 4 --mode netsup > results/log_T3_netsup.txt 2>&1
python scripts/learn_curve.py --bosses 40 --attempts 4 --mode netsup > results/log_train_netsup.txt 2>&1
