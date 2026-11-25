"""Recurrent PPO (RL^2) baseline on Arena. python scripts/arena_train_ppo.py UPDATES [THREADS]"""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from arena.rl import ppo_train
updates = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
os.makedirs("checkpoints", exist_ok=True)
ppo_train(updates=updates, ckpt="checkpoints/arena_ppo.pt", log=lambda s: print(s, flush=True))
