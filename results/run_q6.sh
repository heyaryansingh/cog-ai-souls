while pgrep -f "_v2birth" >/dev/null; do sleep 20; done
python scripts/learn_curve.py --split T4 --n_test 30 --bosses 30 --attempts 8 --self_model active > results/log_T4_active.txt 2>&1
