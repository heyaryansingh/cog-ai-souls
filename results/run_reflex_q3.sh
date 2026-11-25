cd /home/user/cog-ai-souls
for B in 3 10; do
RTB_REFLEX=checkpoints/reflex48_it2.pt+checkpoints/reflex48_it4.pt+checkpoints/reflex48_it5.pt RTB_BLEND=$B python scripts/learn_curve.py --split T1 --bosses 80 --n_test 80 --attempts 3 --mode mixture --tag _n80 > results/log_T1_blend$B.txt 2>&1
RTB_REFLEX=checkpoints/reflex48_it2.pt+checkpoints/reflex48_it4.pt+checkpoints/reflex48_it5.pt RTB_BLEND=$B python scripts/learn_curve.py --split T3 --bosses 80 --n_test 80 --attempts 4 --mode netsup --tag _n80 > results/log_T3_blend$B.txt 2>&1
done
