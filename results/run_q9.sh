for L in structure timing transitions,laws laws timing,structure,laws; do
  RTB_LESION=$L python scripts/learn_curve.py --bosses 40 --attempts 6 --tag _lesion-${L//,/+} > results/log_lesion_${L//,/+}.txt 2>&1
done
