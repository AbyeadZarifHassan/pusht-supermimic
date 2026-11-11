"""SuperMimic-style agent for Push-T: learned world model + BPTT actor + critic.

Follows the training loop of SuperMimic (Fig. 2 / Sec. 3 of the paper):
  (1) collect: deterministic actor + Gaussian action noise into a replay buffer
  (2) fit world model on k-step windows  (Eq. 1)
  (3) improve policy by back-propagation-through-time through the *frozen*
      world model over H imagined steps, bootstrapped with a critic (Eq. 2);
      critic fitted to lambda-returns of the imagined rollout (Eq. 3).
Soft termination: c_0 = 1, c_{t+1} = c_t (1 - d_hat_{t+1}).
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from . import torch_ops as ops
from .world_model import WorldModel, mlp


@dataclass
class SMConfig:
    reward: str = "sparse"
    horizon: int = 32          # H
    wm_window: int = 8         # k
    gamma: float = 0.99
    lam: float = 0.95
    use_critic: bool = True
    obs_velocity: bool = True
    hidden: int = 256
    wm_lr: float = 3e-4
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    wm_batch: int = 256
    policy_batch: int = 256
    critic_steps: int = 2
    grad_clip: float = 0.5
    action_noise: float = 0.3


class SuperMimicAgent:
    def __init__(self, cfg: SMConfig, device):
        self.cfg, self.device = cfg, device
        od = ops.obs_dim(cfg.obs_velocity)
        self.wm = WorldModel(cfg.hidden).to(device)
        self.actor = nn.Sequential(mlp(od, 2, cfg.hidden, 3), nn.Tanh()).to(device)
        self.critic = mlp(od, 1, cfg.hidden, 3).to(device)
        self.wm_opt = torch.optim.Adam(self.wm.parameters(), cfg.wm_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), cfg.critic_lr)

    def obs(self, s):
        return ops.observation(s, self.cfg.obs_velocity)

    @torch.no_grad()
    def act(self, states: np.ndarray, noise: float = 0.0) -> np.ndarray:
        s = torch.as_tensor(states, device=self.device)
        a = self.actor(self.obs(s)).cpu().numpy()
        if noise > 0:
            a = a + np.random.normal(0, noise, a.shape)
        return np.clip(a, -1, 1).astype(np.float32)

    # -------------------------------------------------------------- (2) world model
    def update_world_model(self, buffer):
        s0, a, s2, suc, valid = buffer.sample_windows(self.cfg.wm_batch, self.cfg.wm_window, self.device)
        pos, rate = buffer.sample_success_states(self.cfg.wm_batch // 4, self.device)
        loss, info = self.wm.loss(s0, a, s2, suc, valid, pos, rate)
        self.wm_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), 10.0)
        self.wm_opt.step()
        return info

    # ------------------------------------------------------------- (3) policy + critic
    def imagine(self, s0, grad: bool):
        """Roll the frozen world model forward H steps with the current actor."""
        cfg = self.cfg
        s, c = s0, torch.ones(len(s0), device=self.device)
        states, rewards, dones, alive = [s], [], [], [c]
        for _ in range(cfg.horizon):
            a = self.actor(self.obs(s))
            s_next = self.wm.step(s, a)
            r, d = self.wm.reward(s, s_next, cfg.reward)
            c = c * (1 - d)
            s = s_next
            states.append(s); rewards.append(r); dones.append(d); alive.append(c)
        return states, rewards, dones, alive

    def update_policy(self, buffer):
        cfg = self.cfg
        s0 = buffer.sample_states(cfg.policy_batch, self.device)
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)

        states, rewards, dones, alive = self.imagine(s0, grad=True)
        # J = sum_t gamma^t c_t r_{t+1} + gamma^H c_H V(s_H)
        J = torch.zeros(len(s0), device=self.device)
        disc = 1.0
        for t in range(cfg.horizon):
            J = J + disc * alive[t] * rewards[t]
            disc *= cfg.gamma
        if cfg.use_critic:
            J = J + disc * alive[-1] * self.critic(self.obs(states[-1])).squeeze(-1)
        actor_loss = -J.mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        gnorm = nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
        self.actor_opt.step()
        self.wm.requires_grad_(True)
        self.critic.requires_grad_(True)

        info = {"J": J.mean().item(), "actor_grad_norm": float(gnorm),
                "imag_success_prob": (1 - alive[-1]).mean().item()}
        if cfg.use_critic:
            info.update(self._update_critic(states, rewards, dones))
        return info

    def _update_critic(self, states, rewards, dones):
        """Fit V to lambda-returns computed backward over the imagined transitions only (Eq. 3)."""
        cfg = self.cfg
        with torch.no_grad():
            S = torch.stack([s.detach() for s in states])          # [H+1, B, 8]
            R = torch.stack([r.detach() for r in rewards])          # [H, B]
            D = torch.stack([d.detach() for d in dones])            # [H, B]
            O = self.obs(S)
            V = self.critic(O).squeeze(-1)                          # [H+1, B]
            targets = torch.zeros_like(R)
            ret = V[-1]
            for t in reversed(range(cfg.horizon)):
                ret = R[t] + cfg.gamma * (1 - D[t]) * ((1 - cfg.lam) * V[t + 1] + cfg.lam * ret)
                targets[t] = ret
        for _ in range(cfg.critic_steps):
            v = self.critic(O[:-1]).squeeze(-1)
            loss = ((v - targets) ** 2).mean()
            self.critic_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
            self.critic_opt.step()
        return {"critic_loss": loss.item(), "V_mean": v.mean().item()}

    # ---------------------------------------------------------------- checkpoints
    def state_dict(self):
        return {"cfg": self.cfg.__dict__, "wm": self.wm.state_dict(), "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict()}

    def load_state_dict(self, d):
        self.wm.load_state_dict(d["wm"]); self.actor.load_state_dict(d["actor"]); self.critic.load_state_dict(d["critic"])
