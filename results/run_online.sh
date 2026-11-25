while pgrep -f "run_t4.sh" >/dev/null; do sleep 20; done
python scripts/learn_curve.py --bosses 40 --attempts 12 --mode online > results/log_online.txt 2>&1
