cd /home/user/cog-ai-souls
while pgrep -f "run_meta.sh|run_phantom.sh|run_P1.sh" >/dev/null; do sleep 30; done
RTB_REFLEX=checkpoints/reflex48_it2.pt+checkpoints/reflex48_it4.pt+checkpoints/reflex48_it5.pt RTB_BLEND_ADAPT=1,20,0.28,0.40 python scripts/learn_curve.py --split T1 --bosses 80 --n_test 80 --attempts 3 --mode mixture --tag _n80 > results/log_T1_adapt.txt 2>&1
RTB_REFLEX=checkpoints/reflex48_it2.pt+checkpoints/reflex48_it4.pt+checkpoints/reflex48_it5.pt RTB_BLEND_ADAPT=1,20,0.28,0.40 python scripts/learn_curve.py --split T3 --bosses 80 --n_test 80 --attempts 4 --mode netsup --tag _n80 > results/log_T3_adapt.txt 2>&1
