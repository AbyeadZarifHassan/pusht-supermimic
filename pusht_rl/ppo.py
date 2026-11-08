"""PPO baseline (likelihood-ratio / score-function gradient, on-policy), as in the RL II lecture:

    L = E[ min( ratio * A_GAE, clip(ratio, 1-eps, 1+eps) * A_GAE ) ],  ratio = pi_theta / pi_theta_old

Same observation, action space and environment as the SuperMimic agent.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from . import torch_ops as ops
from .world_model import mlp


@dataclass
class PPOConfig:
    obs_velocity: bool = True
    hidden: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 10
    minibatch: int = 1024
    rollout_len: int = 128
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    init_log_std: float = -0.5


class PPOAgent(nn.Module):
    def __init__(self, cfg: PPOConfig, device):
        super().__init__()
        self.cfg, self.device = cfg, device
        od = ops.obs_dim(cfg.obs_velocity)
        self.actor = mlp(od, 2, cfg.hidden, 3, norm=False)
        self.critic = mlp(od, 1, cfg.hidden, 3, norm=False)
        self.log_std = nn.Parameter(torch.full((2,), cfg.init_log_std))
        self.to(device)
        self.opt = torch.optim.Adam(self.parameters(), cfg.lr, eps=1e-5)

    def obs(self, states):
        s = torch.as_tensor(states, device=self.device)
        return ops.observation(s, self.cfg.obs_velocity)

    def dist(self, o):
        return torch.distributions.Normal(self.actor(o), self.log_std.exp())

    @torch.no_grad()
    def act(self, states, noise: float = 0.0, deterministic: bool = True):
        o = self.obs(states)
        a = self.actor(o) if deterministic else self.dist(o).sample()
        return np.clip(a.cpu().numpy(), -1, 1).astype(np.float32)

    def update(self, obs, actions, logp_old, advantages, returns):
        cfg = self.cfg
        n = len(obs)
        info = {}
        for _ in range(cfg.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, cfg.minibatch):
                mb = perm[i:i + cfg.minibatch]
                d = self.dist(obs[mb])
                logp = d.log_prob(actions[mb]).sum(-1)
                ratio = (logp - logp_old[mb]).exp()
                adv = advantages[mb]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
                v_loss = ((self.critic(obs[mb]).squeeze(-1) - returns[mb]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), cfg.max_grad_norm)
                self.opt.step()
        with torch.no_grad():
            info = {"pg_loss": pg_loss.item(), "v_loss": v_loss.item(), "entropy": ent.item(),
                    "approx_kl": (logp_old[mb] - logp).mean().item(), "log_std": self.log_std.mean().item()}
        return info


def compute_gae(rewards, values, next_values, terminals, ends, gamma, lam):
    """GAE over a [T, N] rollout. ``terminals`` = success (no bootstrap);
    ``ends`` = episode ended for any reason (cuts the GAE recursion)."""
    T = len(rewards)
    adv = np.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] * (1 - terminals[t]) - values[t]
        last = delta + gamma * lam * (1 - ends[t]) * last
        adv[t] = last
    return adv, adv + values
