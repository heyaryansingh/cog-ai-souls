cd /home/user/cog-ai-souls
while pgrep -f "run_P1.sh|run_after_priv.sh|run_priv.sh" >/dev/null; do sleep 30; done
for A in 2 8; do
RTB_PHANTOM=$A python scripts/learn_curve.py --split T3 --bosses 80 --n_test 80 --attempts 4 --mode netsup --tag _n80 > results/log_T3_phantom$A.txt 2>&1
RTB_PHANTOM=$A python scripts/learn_curve.py --split T1 --bosses 80 --n_test 80 --attempts 3 --mode mixture --tag _n80 > results/log_T1_phantom$A.txt 2>&1
done
