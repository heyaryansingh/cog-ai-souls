while pgrep -f "run_online.sh" >/dev/null; do sleep 20; done
python scripts/learn_curve.py --split T3 --n_test 30 --bosses 30 --attempts 8 > results/log_T3.txt 2>&1
python scripts/learn_curve.py --split T3 --n_test 30 --bosses 30 --attempts 8 --mode oracle > results/log_T3_oracle.txt 2>&1
python scripts/learn_curve.py --split T2 --n_test 30 --bosses 30 --attempts 8 > results/log_T2.txt 2>&1
for sm in proprio iframes learned; do
python scripts/learn_curve.py --split T4 --n_test 30 --bosses 30 --attempts 8 --self_model $sm --tag _v2 > results/log_T4_${sm}_v2.txt 2>&1
done
python scripts/learn_curve.py --split T1 --n_test 30 --bosses 30 --attempts 4 --mode online > results/log_T1_online.txt 2>&1
python scripts/learn_curve.py --split T1 --n_test 30 --bosses 30 --attempts 4 --mode mixture --tag _rep > results/log_T1_mixture_rep.txt 2>&1
