while pgrep -f train_rl.py >/dev/null; do sleep 20; done
for pr in vague results/prior_eb.json star; do
  python scripts/learn_curve.py --split T1 --n_test 30 --bosses 30 --attempts 8 --prior $pr > results/log_T1_$(basename $pr .json).txt 2>&1
done
