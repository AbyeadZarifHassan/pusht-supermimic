"""Quick checks: python -m pytest tests -q"""

import math

import numpy as np
import torch

from pusht_rl import torch_ops as ops
from pusht_rl.env import GOAL_POSE, EnvConfig, PushTEnv, pose_potential, rot
from pusht_rl.ppo import compute_gae


def test_goal_is_success_and_offset_is_not():
    e = PushTEnv(EnvConfig(), 0)
    e.set_state([100, 100, *GOAL_POSE, 0, 0, 0])
    assert e.is_success()
    e.set_state([100, 100, GOAL_POSE[0] + 10, GOAL_POSE[1], GOAL_POSE[2], 0, 0, 0])
    assert not e.is_success()


def test_agent_pushes_block():
    e = PushTEnv(EnvConfig(), 0)
    e.set_state([150, 256, 256, 240, 0, 0, 0, 0])
    a = rot(GOAL_POSE[2]).T @ np.array([1.0, 0.0])  # +x in world frame, expressed in goal frame
    for _ in range(8):
        s, *_ = e.step(a)
    assert s[2] > 300  # block moved right


def test_torch_potential_matches_numpy():
    s = torch.tensor([[0, 0, 300.0, 200.0, 0.3, 0, 0, 0]])
    assert math.isclose(ops.pose_potential(s).item(), pose_potential(300.0, 200.0, 0.3), rel_tol=1e-5)


def test_observation_at_goal():
    s = torch.tensor([[256.0, 256.0, *GOAL_POSE, 0, 0, 0]], dtype=torch.float32)
    o = ops.observation(s)
    assert torch.allclose(o[0, 2:6], torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-5)


def test_zero_delta_keeps_position():
    s = torch.tensor([[100.0, 120.0, 300.0, 200.0, 0.7, 10.0, -5.0, 0.2]])
    out = torch.zeros(1, ops.WM_OUT_DIM)
    s2 = ops.wm_apply_delta(s, out)
    assert torch.allclose(s2[0, :5], s[0, :5])


def test_gae_no_bootstrap_on_success():
    adv, ret = compute_gae(np.array([[1.0]]), np.array([[0.5]]), np.array([[10.0]]),
                           np.array([[1.0]]), np.array([[1.0]]), 0.99, 0.95)
    assert np.isclose(ret[0, 0], 1.0)
