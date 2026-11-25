cd /home/user/cog-ai-souls
while pgrep -f "run_phantom.sh|run_P1.sh|run_after_priv.sh" >/dev/null; do sleep 30; done
python scripts/train_metagate.py --reflex checkpoints/reflex48_it2.pt+checkpoints/reflex48_it4.pt+checkpoints/reflex48_it5.pt > results/log_metagate.txt 2>&1
for c in 0 1; do
RTB_REFLEX=checkpoints/reflex48_it2.pt+checkpoints/reflex48_it4.pt+checkpoints/reflex48_it5.pt RTB_METAGATE=checkpoints/metagate.pt RTB_METACOST=$c python scripts/learn_curve.py --split T1 --bosses 80 --n_test 80 --attempts 3 --mode mixture --tag _n80 > results/log_T1_meta$c.txt 2>&1
RTB_REFLEX=checkpoints/reflex48_it2.pt+checkpoints/reflex48_it4.pt+checkpoints/reflex48_it5.pt RTB_METAGATE=checkpoints/metagate.pt RTB_METACOST=$c python scripts/learn_curve.py --split T3 --bosses 80 --n_test 80 --attempts 4 --mode netsup --tag _n80 > results/log_T3_meta$c.txt 2>&1
done
