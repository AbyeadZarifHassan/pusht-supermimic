"""Fixed-size replay buffer stored per environment, so k-step windows stay in time order.

Old transitions are overwritten (as in SuperMimic). Each stored transition is
(s, a, r, success, episode_end, s_next) where s_next is the true successor
before any auto-reset.
"""

import numpy as np
import torch

from .env import ACTION_DIM, STATE_DIM


class ReplayBuffer:
    def __init__(self, n_envs: int, capacity: int, seed: int = 0):
        self.n_envs = n_envs
        self.cap = max(capacity // n_envs, 16)
        self.s = np.zeros((n_envs, self.cap, STATE_DIM), np.float32)
        self.s2 = np.zeros((n_envs, self.cap, STATE_DIM), np.float32)
        self.a = np.zeros((n_envs, self.cap, ACTION_DIM), np.float32)
        self.r = np.zeros((n_envs, self.cap), np.float32)
        self.suc = np.zeros((n_envs, self.cap), np.float32)
        self.end = np.zeros((n_envs, self.cap), np.float32)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)
        self.n_success_total = 0

    def add(self, s, a, r, suc, end, s2):
        p = self.ptr
        self.s[:, p], self.a[:, p], self.r[:, p] = s, a, r
        self.suc[:, p], self.end[:, p], self.s2[:, p] = suc, end, s2
        self.ptr = (p + 1) % self.cap
        self.size = min(self.size + 1, self.cap)
        self.n_success_total += int(suc.sum())

    def __len__(self):
        return self.size * self.n_envs

    def _phys(self, t):
        """Logical time index (0 = oldest) -> physical slot."""
        return (self.ptr - self.size + t) % self.cap

    def sample_windows(self, batch: int, k: int, device):
        """k-step windows: s0, actions[k], next states[k], success[k], valid mask[k]."""
        k = min(k, self.size)
        env = self.rng.integers(0, self.n_envs, batch)
        t0 = self.rng.integers(0, self.size - k + 1, batch)
        idx = self._phys(t0[:, None] + np.arange(k)[None])  # [B, k]
        e = env[:, None]
        end = self.end[e, idx]
        # step j is valid if no episode ended at an earlier step of the window
        valid = np.concatenate([np.ones((batch, 1), np.float32), 1 - np.cumsum(end, 1)[:, :-1].clip(0, 1)], 1)
        f = lambda x: torch.as_tensor(x, device=device)
        return f(self.s[env, idx[:, 0]]), f(self.a[e, idx]), f(self.s2[e, idx]), f(self.suc[e, idx]), f(valid)

    def sample_states(self, batch: int, device):
        env = self.rng.integers(0, self.n_envs, batch)
        idx = self._phys(self.rng.integers(0, self.size, batch))
        return torch.as_tensor(self.s[env, idx], device=device)

    def sample_success_states(self, batch: int, device):
        """Successor states labelled success (for class-balanced training of the success head)."""
        valid = self.suc[:, : self.cap] > 0
        if self.size < self.cap:
            valid = valid.copy()
            used = np.zeros(self.cap, bool)
            used[self._phys(np.arange(self.size))] = True
            valid &= used[None]
        envs, slots = np.nonzero(valid)
        if len(envs) == 0:
            return None, 0.0
        pick = self.rng.integers(0, len(envs), batch)
        rate = len(envs) / (self.size * self.n_envs)
        return torch.as_tensor(self.s2[envs[pick], slots[pick]], device=device), rate
