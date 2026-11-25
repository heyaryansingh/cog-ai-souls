while pgrep -f "run_t1.sh|learn_curve.py" >/dev/null; do sleep 20; done
for sm in train learned true; do
  python scripts/learn_curve.py --split T4 --n_test 30 --bosses 30 --attempts 8 --self_model $sm > results/log_T4_$sm.txt 2>&1
done
