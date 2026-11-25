cd /home/user/cog-ai-souls
while pgrep -f "run_after_priv.sh|run_priv.sh" >/dev/null; do sleep 30; done
export RTB_SPLIT_SEED=1
T1="--split T1 --bosses 80 --n_test 80 --attempts 3 --mode mixture --tag _P1"
T3="--split T3 --bosses 80 --n_test 80 --attempts 4 --mode netsup --tag _P1"
python scripts/learn_curve.py $T1 > results/log_P1_T1_p12.txt 2>&1
python scripts/learn_curve.py $T3 > results/log_P1_T3_p12.txt 2>&1
RTB_REFLEX=checkpoints/reflex48_it2.pt+checkpoints/reflex48_it4.pt+checkpoints/reflex48_it5.pt python scripts/learn_curve.py $T1 > results/log_P1_T1_reflex.txt 2>&1
RTB_REFLEX=checkpoints/reflex48_it2.pt+checkpoints/reflex48_it4.pt+checkpoints/reflex48_it5.pt python scripts/learn_curve.py $T3 > results/log_P1_T3_reflex.txt 2>&1
RTB_REFLEX=checkpoints/reflex48_it2.pt+checkpoints/reflex48_it4.pt+checkpoints/reflex48_it5.pt RTB_BLEND=3 python scripts/learn_curve.py $T1 > results/log_P1_T1_blend3.txt 2>&1
RTB_REFLEX=checkpoints/reflex48_it2.pt+checkpoints/reflex48_it4.pt+checkpoints/reflex48_it5.pt RTB_BLEND=3 python scripts/learn_curve.py $T3 > results/log_P1_T3_blend3.txt 2>&1
RTB_REFLEX=checkpoints/reflex48_it2.pt+checkpoints/reflex48_it4.pt+checkpoints/reflex48_it5.pt RTB_REFLEX_ATTEMPTS=1 python scripts/learn_curve.py $T3 > results/log_P1_T3_ra1.txt 2>&1
python scripts/learn_curve.py $T1 --rollouts 48 --tag _P1_r48 > results/log_P1_T1_p48.txt 2>&1
python scripts/learn_curve.py $T3 --rollouts 48 --tag _P1_r48 > results/log_P1_T3_p48.txt 2>&1
