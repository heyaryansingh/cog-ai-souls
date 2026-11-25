RTB_DELETE_BIAS=3 python scripts/learn_curve.py --bosses 40 --attempts 4 --tag _del+3 > results/log_del+3.txt 2>&1
RTB_DELETE_BIAS=-inf python scripts/learn_curve.py --bosses 40 --attempts 4 --tag _nodelete > results/log_nodelete.txt 2>&1
