"""Learned differentiable world model f_w (SuperMimic Sec. 3.2) for Push-T.

* dynamics head  : s_{t+1} = f_w(s_t, a_t)          (MLP, learned)
* success head   : d_t = P(success | s_t)            (MLP, learned; success == termination in Push-T)
* reward head    : sparse -> r = d(s_{t+1});  dense -> analytic potential shaping + d(s_{t+1})
* observation    : analytic (torch_ops.observation)

Fitted on k-step autoregressive windows (Eq. 1 of the paper):
    L = sum_j || s_hat_j - s_j ||^2 + BCE(d_hat_j, d_j)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import torch_ops as ops


def mlp(i, o, hidden=256, layers=3, norm=True):
    mods, d = [], i
    for _ in range(layers):
        mods += [nn.Linear(d, hidden)] + ([nn.LayerNorm(hidden)] if norm else []) + [nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


class WorldModel(nn.Module):
    def __init__(self, hidden=256, layers=3):
        super().__init__()
        self.dyn = mlp(ops.WM_IN_DIM, ops.WM_OUT_DIM, hidden, layers)
        self.succ = mlp(4, 1, 128, 2)
        # log-odds correction for class-balanced training of the success head
        self.register_buffer("prior_logit", torch.tensor(-10.0))

    def step(self, s, a):
        return ops.wm_apply_delta(s, self.dyn(ops.wm_input(s, a)))

    def success_logit(self, s):
        return self.succ(ops.success_input(s)).squeeze(-1) + self.prior_logit

    def success_prob(self, s):
        return torch.sigmoid(self.success_logit(s))

    def reward(self, s, s_next, mode: str):
        p = self.success_prob(s_next)
        if mode == "dense":
            return ops.pose_potential(s_next) - ops.pose_potential(s) + p, p
        return p, p

    # ------------------------------------------------------------------ fitting
    def loss(self, s0, actions, s_next, success, valid, pos_states=None, pos_rate=0.0):
        k = actions.shape[1]
        s = s0
        dyn_err = []
        for j in range(k):
            s = self.step(s, actions[:, j])
            dyn_err.append(ops.state_error(s, s_next[:, j]))
        dyn_err = torch.stack(dyn_err, 1)  # [B, k]
        dyn_loss = (dyn_err * valid).sum() / valid.sum().clamp(min=1)

        # success head: trained on true successor states. When positives exist we
        # use a 50/50 balanced batch and correct the logit by the true prior.
        x = s_next.reshape(-1, s_next.shape[-1])
        y = success.reshape(-1)
        if pos_states is not None:
            n = len(pos_states)
            x = torch.cat([x[:n], pos_states])
            y = torch.cat([y[:n], torch.ones(n, device=x.device)])
            logit = lambda p: math.log(p / (1 - p))
            batch_rate = min(max(y.mean().item(), 1e-6), 1 - 1e-6)
            true_rate = min(max(pos_rate, 1e-6), 0.5)
            self.prior_logit.fill_(logit(true_rate) - logit(batch_rate))
        else:
            self.prior_logit.fill_(0.0)  # no positives yet: the head itself learns "never"
        logit_raw = self.succ(ops.success_input(x)).squeeze(-1)
        succ_loss = F.binary_cross_entropy_with_logits(logit_raw, y)

        with torch.no_grad():
            # k-step prediction error of the block position in pixels (for logging)
            px = torch.sqrt((s[:, 2] - s_next[:, -1, 2]) ** 2 + (s[:, 3] - s_next[:, -1, 3]) ** 2)
            m = valid[:, -1]
            block_err_px = (px * m).sum() / m.sum().clamp(min=1)
        return dyn_loss + succ_loss, {"wm_dyn": dyn_loss.item(), "wm_succ": succ_loss.item(),
                                       "wm_kstep_block_err_px": block_err_px.item()}
