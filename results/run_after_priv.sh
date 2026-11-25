cd /home/user/cog-ai-souls
while pgrep -f run_priv.sh >/dev/null; do sleep 30; done
python scripts/shield_probe.py > results/log_shield.txt 2>&1
