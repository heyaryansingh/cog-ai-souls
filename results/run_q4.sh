while pgrep -f "run_q3.sh" >/dev/null; do sleep 20; done
python scripts/train_intuition.py --n 40000 --epochs 15 --procs 3 > results/log_intuition.txt 2>&1
python scripts/ceiling_probe.py > results/log_ceiling.txt 2>&1
python scripts/amortization_gap.py > results/log_amortization.txt 2>&1
