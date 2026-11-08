"""Push-T environment (state-based, no pygame).

Geometry, physics rates and the 95% coverage success test follow the
original Push-T task (Chi et al., Diffusion Policy; also gym-pusht).
Observation/action design and the reverse curriculum follow
Ma et al. 2026, "A Single Diffusion-Policy Controller for Multi-Task
Block Pushing" (arXiv:2607.10892):

* action  : velocity command of the pusher in the *goal* frame, a in [-1, 1]^2
* obs     : objective-centric, everything expressed in the *block* frame
* reward  : sparse (1 on success) or potential-shaped dense (see below)
* curriculum: block starts within 256 +- (30 + m/M * 100) px of the canvas
  centre at stage m; the stage advances after N successes.

The full simulator state used by the world model is

    s = [agent_x, agent_y, block_x, block_y, block_theta, block_vx, block_vy, block_omega]

in pixels / radians / px-per-second.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pymunk
from shapely.geometry import Point, Polygon
from shapely import affinity

STATE_DIM = 8
ACTION_DIM = 2

CANVAS = 512.0
GOAL_POSE = np.array([256.0, 256.0, math.pi / 4])  # x, y, theta (same as Push-T)

# T-block geometry (Push-T: scale=30, length=4), in the block's local frame.
_SCALE, _LEN = 30.0, 4.0
T_VERTS_BAR = [(-_LEN * _SCALE / 2, _SCALE), (_LEN * _SCALE / 2, _SCALE),
               (_LEN * _SCALE / 2, 0.0), (-_LEN * _SCALE / 2, 0.0)]
T_VERTS_STEM = [(-_SCALE / 2, _SCALE), (-_SCALE / 2, _LEN * _SCALE),
                (_SCALE / 2, _LEN * _SCALE), (_SCALE / 2, _SCALE)]
T_LOCAL = Polygon(T_VERTS_BAR).union(Polygon(T_VERTS_STEM))
AGENT_RADIUS = 15.0


def wrap_angle(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def t_polygon(x, y, theta):
    """T-shape polygon at pose (x, y, theta) in world coordinates."""
    poly = affinity.rotate(T_LOCAL, theta, origin=(0, 0), use_radians=True)
    return affinity.translate(poly, x, y)


_GOAL_POLY = t_polygon(*GOAL_POSE)


def coverage(x, y, theta) -> float:
    """Fraction of the goal T covered by the block (Push-T success metric)."""
    return _GOAL_POLY.intersection(t_polygon(x, y, theta)).area / _GOAL_POLY.area


def pose_potential(bx, by, theta, sigma_pos=60.0):
    """Smooth pose score in [0, 1]; 1 at the goal pose.

    Used for the dense (potential-shaped) reward. The same formula is
    re-implemented in torch (``torch_ops.pose_potential``) so the world
    model can differentiate through it.
    """
    d = math.hypot(bx - GOAL_POSE[0], by - GOAL_POSE[1])
    dth = wrap_angle(GOAL_POSE[2] - theta)
    return math.exp(-d / sigma_pos) * 0.5 * (1.0 + math.cos(dth))


@dataclass
class EnvConfig:
    reward: str = "sparse"          # "sparse" | "dense"
    success_threshold: float = 0.95  # coverage needed for success (Push-T: 0.95)
    max_steps: int = 300            # episode limit used in the ESM paper
    v_max: float = 200.0            # px/s for |a| = 1
    sim_hz: int = 100
    control_hz: int = 10
    damping: float = 0.99           # fraction of block velocity lost per second (0 = original Push-T, block keeps sliding)
    # reverse curriculum (ESM paper, App. B)
    curriculum: bool = True
    curriculum_stages: int = 10     # M
    curriculum_successes: int = 50  # N
    curriculum_base: float = 30.0
    curriculum_span: float = 100.0
    curriculum_angle: bool = False  # extra (not in paper): also widen the initial angle range


class PushTEnv:
    """Single Push-T environment with velocity actions in the goal frame."""

    def __init__(self, cfg: EnvConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.dt = 1.0 / cfg.sim_hz
        self.n_sub = cfg.sim_hz // cfg.control_hz
        self.R_goal = rot(GOAL_POSE[2])
        self._build()
        self.steps = 0

    # ------------------------------------------------------------------ physics
    def _build(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 1.0 - self.cfg.damping if self.cfg.damping > 0 else 1.0
        walls = [((5, 506), (5, 5)), ((5, 5), (506, 5)), ((506, 5), (506, 506)), ((5, 506), (506, 506))]
        for a, b in walls:
            seg = pymunk.Segment(self.space.static_body, a, b, 2)
            seg.friction = 1.0
            self.space.add(seg)

        self.agent = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        agent_shape = pymunk.Circle(self.agent, AGENT_RADIUS)
        agent_shape.friction = 1.0
        self.space.add(self.agent, agent_shape)

        mass = 1.0
        inertia = pymunk.moment_for_poly(mass, T_VERTS_BAR) * 2  # same as Push-T (uses bar twice)
        self.block = pymunk.Body(mass, inertia)
        s1 = pymunk.Poly(self.block, T_VERTS_BAR)
        s2 = pymunk.Poly(self.block, T_VERTS_STEM)
        for s in (s1, s2):
            s.friction = 1.0
        self.block.center_of_gravity = (s1.center_of_gravity + s2.center_of_gravity) / 2
        self.space.add(self.block, s1, s2)

    def get_state(self) -> np.ndarray:
        a, b = self.agent.position, self.block.position
        v = self.block.velocity
        return np.array([a.x, a.y, b.x, b.y, wrap_angle(self.block.angle), v.x, v.y,
                         self.block.angular_velocity], dtype=np.float32)

    def set_state(self, state):
        state = np.asarray(state, dtype=np.float64)
        self.agent.position = (state[0], state[1])
        self.agent.velocity = (0, 0)
        # pymunk rotates about the centre of gravity, so set the angle first and
        # the position second; then block.position is exactly (state[2], state[3]).
        self.block.angle = state[4]
        self.block.position = (state[2], state[3])
        vel = state[5:8] if len(state) >= 8 else (0, 0, 0)
        self.block.velocity = (vel[0], vel[1])
        self.block.angular_velocity = vel[2]
        self.space.reindex_shapes_for_body(self.block)

    # ------------------------------------------------------------- curriculum
    def sample_initial_state(self, stage: int | None) -> np.ndarray:
        cfg = self.cfg
        if stage is None or not cfg.curriculum:
            # original Push-T reset distribution
            bx, by = self.rng.uniform(100, 400, size=2)
            theta = self.rng.uniform(-np.pi, np.pi)
        else:
            half = cfg.curriculum_base + stage / cfg.curriculum_stages * cfg.curriculum_span
            bx, by = self.rng.uniform(256 - half, 256 + half, size=2)
            if cfg.curriculum_angle:
                span = np.pi * (0.25 + 0.75 * stage / cfg.curriculum_stages)
                theta = wrap_angle(GOAL_POSE[2] + self.rng.uniform(-span, span))
            else:
                theta = self.rng.uniform(-np.pi, np.pi)
        block_poly = t_polygon(bx, by, theta).buffer(AGENT_RADIUS + 5)
        for _ in range(100):
            ax, ay = self.rng.uniform(50, 450, size=2)
            if not block_poly.contains(Point(ax, ay)):
                break
        return np.array([ax, ay, bx, by, theta, 0, 0, 0], dtype=np.float32)

    def reset(self, stage: int | None = None, state=None) -> np.ndarray:
        self.set_state(self.sample_initial_state(stage) if state is None else state)
        self.steps = 0
        self.space.step(self.dt)  # settle
        if coverage(*self._pose()) > self.cfg.success_threshold:  # avoid starting solved
            return self.reset(stage, None)
        return self.get_state()

    def _pose(self):
        b = self.block.position
        return b.x, b.y, self.block.angle

    def is_success(self) -> bool:
        x, y, th = self._pose()
        # cheap pre-check before the exact polygon test
        if math.hypot(x - GOAL_POSE[0], y - GOAL_POSE[1]) > 25 or abs(wrap_angle(th - GOAL_POSE[2])) > 0.35:
            return False
        return coverage(x, y, th) > self.cfg.success_threshold

    def step(self, action):
        """Returns next_state, reward, success(terminal), truncated."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        v = self.R_goal @ a * self.cfg.v_max  # goal frame -> world frame
        phi0 = pose_potential(*self._pose())
        for _ in range(self.n_sub):
            p = np.array(self.agent.position)
            target = np.clip(p + v * self.dt, AGENT_RADIUS + 6, CANVAS - AGENT_RADIUS - 6)
            self.agent.velocity = tuple((target - p) / self.dt)
            self.space.step(self.dt)
        self.steps += 1
        success = self.is_success()
        if self.cfg.reward == "sparse":
            reward = 1.0 if success else 0.0
        else:  # potential-based shaping + success bonus
            reward = pose_potential(*self._pose()) - phi0 + (1.0 if success else 0.0)
        truncated = (not success) and self.steps >= self.cfg.max_steps
        return self.get_state(), float(reward), bool(success), bool(truncated)

    # --------------------------------------------------------------- rendering
    def render(self, size: int = 256):
        return render_state(self.get_state(), size)


def render_state(state, size: int = 256):
    from PIL import Image, ImageDraw

    k = size / CANVAS
    img = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)

    def draw_t(x, y, th, fill):
        poly = t_polygon(x, y, th)
        d.polygon([(px * k, py * k) for px, py in poly.exterior.coords], fill=fill)

    draw_t(*GOAL_POSE, (144, 238, 144))
    draw_t(state[2], state[3], state[4], (119, 136, 153))
    r = AGENT_RADIUS * k
    d.ellipse([state[0] * k - r, state[1] * k - r, state[0] * k + r, state[1] * k + r], fill=(65, 105, 225))
    return img


@dataclass
class Curriculum:
    cfg: EnvConfig
    stage: int = 0
    successes_in_stage: int = 0
    history: list = field(default_factory=list)

    def current(self):
        return self.stage if self.cfg.curriculum else None

    def report_success(self, step: int):
        if not self.cfg.curriculum or self.stage >= self.cfg.curriculum_stages:
            return
        self.successes_in_stage += 1
        if self.successes_in_stage >= self.cfg.curriculum_successes:
            self.stage += 1
            self.successes_in_stage = 0
            self.history.append((step, self.stage))


class VecPushT:
    """A list of PushTEnv with auto-reset. Keeps per-env episode stats."""

    def __init__(self, cfg: EnvConfig, n: int, seed: int = 0):
        self.cfg = cfg
        self.envs = [PushTEnv(cfg, seed + 1000 * i) for i in range(n)]
        self.n = n
        self.curriculum = Curriculum(cfg)
        self.total_steps = 0
        self.ep_len = np.zeros(n, dtype=np.int64)
        self.ep_ret = np.zeros(n, dtype=np.float64)
        self.finished = []  # (success, length, return, stage, final coverage) of finished episodes
        self.states = np.stack([e.reset(self.curriculum.current()) for e in self.envs])

    def step(self, actions):
        """Returns (state, next_state, reward, success, episode_end, new_state).

        ``next_state`` is the true successor (before auto-reset); ``new_state``
        is the state to act from next (after auto-reset where needed).
        """
        nxt = np.zeros_like(self.states)
        rew = np.zeros(self.n, dtype=np.float32)
        suc = np.zeros(self.n, dtype=np.float32)
        end = np.zeros(self.n, dtype=np.float32)
        new = np.zeros_like(self.states)
        for i, env in enumerate(self.envs):
            s2, r, success, trunc = env.step(actions[i])
            nxt[i], rew[i], suc[i] = s2, r, float(success)
            self.ep_len[i] += 1
            self.ep_ret[i] += r
            if success or trunc:
                end[i] = 1.0
                self.finished.append((success, int(self.ep_len[i]), float(self.ep_ret[i]), self.curriculum.stage,
                                      float(coverage(s2[2], s2[3], s2[4]))))
                if success:
                    self.curriculum.report_success(self.total_steps)
                self.ep_len[i], self.ep_ret[i] = 0, 0.0
                new[i] = env.reset(self.curriculum.current())
            else:
                new[i] = s2
        self.total_steps += self.n
        prev = self.states
        self.states = new
        return prev, nxt, rew, suc, end, new

    def pop_finished(self):
        out, self.finished = self.finished, []
        return out
