"""Evaluate a saved checkpoint and write a GIF.

python evaluate.py runs/sm_dense --episodes 100
"""

import argparse
import json
import os

import torch

from pusht_rl.env import EnvConfig
from pusht_rl.evaluation import evaluate
from pusht_rl.ppo import PPOAgent, PPOConfig
from pusht_rl.supermimic import SMConfig, SuperMimicAgent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    args = json.load(open(os.path.join(a.run, "args.json")))
    ckpt = torch.load(os.path.join(a.run, "checkpoint.pt"), map_location=a.device)
    if args["algo"] == "supermimic":
        agent = SuperMimicAgent(SMConfig(**ckpt["cfg"]), a.device)
        agent.load_state_dict(ckpt)
        fn = lambda s: agent.act(s, 0.0)
    else:
        agent = PPOAgent(PPOConfig(obs_velocity=not args["no_obs_velocity"]), a.device)
        agent.load_state_dict(ckpt)
        fn = lambda s: agent.act(s, deterministic=True)
    env_cfg = EnvConfig(reward=args["reward"], success_threshold=args["success_threshold"], damping=args["damping"])
    for name, stage in [("full Push-T reset", None), ("final curriculum stage", env_cfg.curriculum_stages)]:
        res = evaluate(fn, env_cfg, a.episodes, stage=stage,
                       gif_path=os.path.join(a.run, f"final_{'full' if stage is None else 'curr'}.gif"))
        print(f"{name:25s} success={res['success_rate'] * 100:.1f}%  length={res['episode_length']:.1f} +- {res['episode_length_std']:.1f}")


if __name__ == "__main__":
    main()
