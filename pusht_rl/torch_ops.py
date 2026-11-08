"""Differentiable (torch) versions of the analytic heads.

State layout (see env.py):  s = [ax, ay, bx, by, theta, vx, vy, omega]
theta is kept *unwrapped* inside imagined rollouts; every function below only
uses sin/cos of it, so nothing is discontinuous.
"""

import math

import torch

from .env import GOAL_POSE

GX, GY, GTH = float(GOAL_POSE[0]), float(GOAL_POSE[1]), float(GOAL_POSE[2])
_CG, _SG = math.cos(GTH), math.sin(GTH)


def goal_to_world(a):
    """Action in goal frame -> world-frame direction (unit-free)."""
    return torch.stack([_CG * a[..., 0] - _SG * a[..., 1], _SG * a[..., 0] + _CG * a[..., 1]], -1)


def to_block_frame(vx, vy, th):
    c, s = torch.cos(th), torch.sin(th)
    return c * vx + s * vy, -s * vx + c * vy


def observation(s, with_velocity: bool = True):
    """Objective-centric observation (ESM paper, Sec. III-B), everything in the block frame.

    [agent_pos^B (2), goal_pos^B (2), sin(theta_goal^B), cos(theta_goal^B)] (+ block velocity^B, omega)
    """
    ax, ay, bx, by, th = s[..., 0], s[..., 1], s[..., 2], s[..., 3], s[..., 4]
    pax, pay = to_block_frame(ax - bx, ay - by, th)
    pgx, pgy = to_block_frame(GX - bx, GY - by, th)
    dth = GTH - th
    feats = [pax / 100, pay / 100, pgx / 100, pgy / 100, torch.sin(dth), torch.cos(dth)]
    if with_velocity:
        vbx, vby = to_block_frame(s[..., 5], s[..., 6], th)
        feats += [vbx / 100, vby / 100, s[..., 7] / 2]
    return torch.stack(feats, -1)


def obs_dim(with_velocity: bool = True):
    return 9 if with_velocity else 6


def pose_potential(s, sigma_pos: float = 60.0):
    d = torch.sqrt((s[..., 2] - GX) ** 2 + (s[..., 3] - GY) ** 2 + 1e-6)
    return torch.exp(-d / sigma_pos) * 0.5 * (1 + torch.cos(GTH - s[..., 4]))


# ----------------------------------------------------------------- world model I/O
def wm_input(s, a):
    """Features for the dynamics network: world + block-frame quantities."""
    ax, ay, bx, by, th = s[..., 0], s[..., 1], s[..., 2], s[..., 3], s[..., 4]
    aw = goal_to_world(a)
    pax, pay = to_block_frame(ax - bx, ay - by, th)
    abx, aby = to_block_frame(aw[..., 0], aw[..., 1], th)
    vbx, vby = to_block_frame(s[..., 5], s[..., 6], th)
    return torch.stack([
        (ax - 256) / 256, (ay - 256) / 256, (bx - 256) / 256, (by - 256) / 256,
        torch.sin(th), torch.cos(th),
        vbx / 100, vby / 100, s[..., 7] / 2,
        pax / 100, pay / 100,
        aw[..., 0], aw[..., 1], abx, aby,
    ], -1)


WM_IN_DIM = 15
WM_OUT_DIM = 8


def wm_apply_delta(s, out):
    """Network output (normalised) -> next state.

    out = [d_agent(2)/20, d_block^B(2)/20, d_theta/0.2, v'^B(2)/100, omega'/2]
    Block displacement and velocity are predicted in the block frame and
    rotated back, which makes the model translation/rotation equivariant.
    """
    th = s[..., 4]
    c, sn = torch.cos(th), torch.sin(th)
    dbx = (c * out[..., 2] - sn * out[..., 3]) * 20
    dby = (sn * out[..., 2] + c * out[..., 3]) * 20
    vx = (c * out[..., 5] - sn * out[..., 6]) * 100
    vy = (sn * out[..., 5] + c * out[..., 6]) * 100
    return torch.stack([
        s[..., 0] + out[..., 0] * 20, s[..., 1] + out[..., 1] * 20,
        s[..., 2] + dbx, s[..., 3] + dby,
        th + out[..., 4] * 0.2,
        vx, vy, out[..., 7] * 2,
    ], -1)


def state_error(pred, target):
    """Normalised per-sample squared error between two states (used for the WM loss)."""
    e = [
        (pred[..., 0] - target[..., 0]) / 20, (pred[..., 1] - target[..., 1]) / 20,
        (pred[..., 2] - target[..., 2]) / 20, (pred[..., 3] - target[..., 3]) / 20,
        (torch.sin(pred[..., 4]) - torch.sin(target[..., 4])) / 0.2,
        (torch.cos(pred[..., 4]) - torch.cos(target[..., 4])) / 0.2,
        (pred[..., 5] - target[..., 5]) / 100, (pred[..., 6] - target[..., 6]) / 100,
        (pred[..., 7] - target[..., 7]) / 2,
    ]
    return sum(x ** 2 for x in e)


def success_input(s):
    dth = s[..., 4] - GTH
    return torch.stack([(s[..., 2] - GX) / 20, (s[..., 3] - GY) / 20,
                        torch.sin(dth) / 0.1, (1 - torch.cos(dth)) / 0.01], -1)
