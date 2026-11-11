"""Train SuperMimic-style (world model + BPTT) or PPO on Push-T.

Examples
--------
python train.py --algo supermimic --reward sparse --total-steps 2000000 --run sm_sparse
python train.py --algo supermimic --reward dense  --total-steps 2000000 --run sm_dense
python train.py --algo ppo        --reward sparse --total-steps 2000000 --run ppo_sparse
python train.py --algo supermimic --reward sparse --no-critic          --run sm_sparse_nocritic
"""

import argparse
import csv
import json
import os
import time
from collections import deque

import numpy as np
import torch

from pusht_rl.buffer import ReplayBuffer
from pusht_rl.env import EnvConfig, VecPushT
from pusht_rl.evaluation import evaluate
from pusht_rl.ppo import PPOAgent, PPOConfig, compute_gae
from pusht_rl.supermimic import SMConfig, SuperMimicAgent


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["supermimic", "ppo"], default="supermimic")
    p.add_argument("--reward", choices=["sparse", "dense"], default="sparse")
    p.add_argument("--total-steps", type=float, default=2e6)
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--run", default=None, help="output folder name under runs/")
    p.add_argument("--eval-every", type=float, default=1e5)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--log-every", type=float, default=1e4)
    # environment
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--curriculum-angle", action="store_true", help="also widen the initial angle range (not in the ESM paper)")
    p.add_argument("--success-threshold", type=float, default=0.95)
    p.add_argument("--damping", type=float, default=0.99, help="fraction of block velocity lost per second (table friction); 0 = original Push-T")
    p.add_argument("--no-obs-velocity", action="store_true", help="use the 6-D ESM observation only")
    # supermimic
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--wm-window", type=int, default=8)
    p.add_argument("--no-critic", action="store_true")
    p.add_argument("--steps-per-iter", type=int, default=8, help="env steps per env between updates")
    p.add_argument("--wm-updates", type=int, default=4)
    p.add_argument("--policy-updates", type=int, default=4)
    p.add_argument("--buffer-size", type=float, default=1e6)
    p.add_argument("--warmup-steps", type=float, default=2e4)
    p.add_argument("--action-noise", type=float, default=0.3)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=0.5, help="max actor gradient norm for BPTT")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    # ppo
    p.add_argument("--rollout-len", type=int, default=128)
    p.add_argument("--ppo-epochs", type=int, default=10)
    p.add_argument("--ent-coef", type=float, default=0.0)
    return p.parse_args()


class Logger:
    def __init__(self, path):
        self.path, self.keys, self.f = path, None, None

    def log(self, row):
        if self.keys is None:
            self.keys = list(row.keys())
            self.f = open(self.path, "w", newline="")
            self.w = csv.DictWriter(self.f, fieldnames=self.keys, extrasaction="ignore")
            self.w.writeheader()
        self.w.writerow({k: row.get(k, "") for k in self.keys})
        self.f.flush()


def main():
    args = parse()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run = args.run or f"{args.algo}_{args.reward}_s{args.seed}"
    out = os.path.join("runs", run)
    os.makedirs(out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out, "args.json"), "w"), indent=2)

    env_cfg = EnvConfig(reward=args.reward, success_threshold=args.success_threshold, damping=args.damping,
                        curriculum=not args.no_curriculum, curriculum_angle=args.curriculum_angle)
    venv = VecPushT(env_cfg, args.n_envs, args.seed)
    logger = Logger(os.path.join(out, "log.csv"))
    eval_logger = Logger(os.path.join(out, "eval.csv"))
    recent = deque(maxlen=200)
    t_start = time.time()
    next_log, next_eval = args.log_every, args.eval_every
    obs_vel = not args.no_obs_velocity
    print(f"[{run}] device={device} algo={args.algo} reward={args.reward}")

    best = {"score": -1.0}

    def periodic(extra, policy_fn, agent_state):
        nonlocal next_log, next_eval
        for ep in venv.pop_finished():
            recent.append(ep)
        steps = venv.total_steps
        if steps >= next_log:
            next_log += args.log_every
            row = {"env_steps": steps, "time_s": round(time.time() - t_start, 1),
                   "stage": venv.curriculum.stage,
                   "train_success": np.mean([e[0] for e in recent]) if recent else 0.0,
                   "train_ep_len": np.mean([e[1] for e in recent]) if recent else 0.0,
                   "train_final_coverage": np.mean([e[4] for e in recent]) if recent else 0.0, **extra}
            logger.log(row)
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        if steps >= next_eval or steps >= args.total_steps:
            next_eval += args.eval_every
            gif = os.path.join(out, f"eval_{steps // 1000}k.gif")
            res_full = evaluate(policy_fn, env_cfg, args.eval_episodes, stage=None, gif_path=gif)
            res_cur = evaluate(policy_fn, env_cfg, args.eval_episodes // 2, stage=env_cfg.curriculum_stages)
            row = {"env_steps": steps, "time_s": round(time.time() - t_start, 1), "stage": venv.curriculum.stage,
                   "eval_success_full": res_full["success_rate"], "eval_len_full": res_full["episode_length"],
                   "eval_coverage_full": res_full["final_coverage"],
                   "eval_success_curr_final": res_cur["success_rate"], "eval_len_curr_final": res_cur["episode_length"]}
            eval_logger.log(row)
            print("EVAL", row, flush=True)
            torch.save(agent_state(), os.path.join(out, "checkpoint.pt"))
            score = res_full["success_rate"] + res_full["final_coverage"]  # success first, coverage breaks ties
            if score > best["score"]:
                best["score"] = score
                torch.save(agent_state(), os.path.join(out, "checkpoint_best.pt"))
                print(f"new best checkpoint (success={res_full['success_rate']:.2f}, coverage={res_full['final_coverage']:.3f})")

    if args.algo == "supermimic":
        cfg = SMConfig(reward=args.reward, horizon=args.horizon, wm_window=args.wm_window,
                       use_critic=not args.no_critic, obs_velocity=obs_vel, action_noise=args.action_noise,
                       actor_lr=args.actor_lr, grad_clip=args.grad_clip)
        agent = SuperMimicAgent(cfg, device)
        buf = ReplayBuffer(args.n_envs, int(args.buffer_size), args.seed)
        policy_fn = lambda s: agent.act(s, 0.0)
        info = {}
        while venv.total_steps < args.total_steps:
            # (1) collect data with the frozen policy + action noise
            for _ in range(args.steps_per_iter):
                if venv.total_steps < args.warmup_steps:
                    a = np.random.uniform(-1, 1, (args.n_envs, 2)).astype(np.float32)
                else:
                    a = agent.act(venv.states, cfg.action_noise)
                s, s2, r, suc, end, _ = venv.step(a)
                buf.add(s, a, r, suc, end, s2)
            if buf.size < args.wm_window + 1:
                continue
            # (2) fit world model
            for _ in range(args.wm_updates):
                info.update(agent.update_world_model(buf))
            # (3) improve policy (+ critic) through the frozen world model
            if venv.total_steps >= args.warmup_steps:
                for _ in range(args.policy_updates):
                    info.update(agent.update_policy(buf))
            info["buffer_successes"] = buf.n_success_total
            periodic(dict(info), policy_fn, agent.state_dict)

    else:  # PPO
        pcfg = PPOConfig(obs_velocity=obs_vel, rollout_len=args.rollout_len, epochs=args.ppo_epochs, ent_coef=args.ent_coef)
        agent = PPOAgent(pcfg, device)
        policy_fn = lambda s: agent.act(s, deterministic=True)
        T, N = pcfg.rollout_len, args.n_envs
        while venv.total_steps < args.total_steps:
            S = np.zeros((T, N, 8), np.float32); S2 = np.zeros_like(S)
            A = np.zeros((T, N, 2), np.float32); LP = np.zeros((T, N), np.float32)
            R = np.zeros((T, N), np.float32); SUC = np.zeros_like(R); END = np.zeros_like(R)
            for t in range(T):
                with torch.no_grad():
                    o = agent.obs(venv.states)
                    d = agent.dist(o)
                    a = d.sample()
                    LP[t] = d.log_prob(a).sum(-1).cpu().numpy()
                A[t] = a.cpu().numpy()
                s, s2, r, suc, end, _ = venv.step(np.clip(A[t], -1, 1))
                S[t], S2[t], R[t], SUC[t], END[t] = s, s2, r, suc, end
            with torch.no_grad():
                V = agent.critic(agent.obs(S.reshape(-1, 8))).squeeze(-1).cpu().numpy().reshape(T, N)
                V2 = agent.critic(agent.obs(S2.reshape(-1, 8))).squeeze(-1).cpu().numpy().reshape(T, N)
            adv, ret = compute_gae(R, V, V2, SUC, END, pcfg.gamma, pcfg.gae_lambda)
            f = lambda x: torch.as_tensor(x.reshape(T * N, *x.shape[2:]), device=device)
            info = agent.update(agent.obs(S.reshape(-1, 8)), f(A), f(LP), f(adv), f(ret))
            periodic(info, policy_fn, agent.state_dict)

    print(f"done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
