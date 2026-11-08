"""Evaluation: success rate and episode length over randomized test episodes (ESM paper protocol)."""

import numpy as np

from .env import EnvConfig, PushTEnv, coverage, render_state


def evaluate(policy_fn, env_cfg: EnvConfig, n_episodes: int = 100, seed: int = 12345,
             stage=None, gif_path: str | None = None, gif_episodes: int = 4):
    """policy_fn: np.ndarray [N, 8] -> actions [N, 2] (deterministic).

    stage=None -> original Push-T reset distribution (block anywhere in [100, 400]^2, any angle);
    stage=m    -> curriculum stage-m distribution.
    """
    cfg = EnvConfig(**{**env_cfg.__dict__, "curriculum": stage is not None})
    envs = [PushTEnv(cfg, seed + i) for i in range(n_episodes)]
    states = np.stack([e.reset(stage) for e in envs])
    done = np.zeros(n_episodes, bool)
    success = np.zeros(n_episodes, bool)
    length = np.full(n_episodes, cfg.max_steps)
    frames = [[] for _ in range(min(gif_episodes, n_episodes))] if gif_path else None
    for t in range(cfg.max_steps):
        if frames is not None:
            for i, f in enumerate(frames):
                if not done[i]:
                    f.append(render_state(states[i]))
        actions = policy_fn(states)
        for i, e in enumerate(envs):
            if done[i]:
                continue
            s2, _, suc, trunc = e.step(actions[i])
            states[i] = s2
            if suc or trunc:
                done[i], success[i] = True, suc
                length[i] = t + 1
        if done.all():
            break
    if frames is not None:
        _save_gif(frames, gif_path)
    final_cov = np.mean([coverage(s[2], s[3], s[4]) for s in states])
    return {"success_rate": float(success.mean()), "final_coverage": float(final_cov), "episode_length": float(length.mean()),
            "episode_length_std": float(length.std())}


def _save_gif(frames, path):
    from PIL import Image

    T = max(len(f) for f in frames)
    tiles = []
    for t in range(T):
        row = [f[min(t, len(f) - 1)] for f in frames]
        w, h = row[0].size
        canvas = Image.new("RGB", (w * len(row), h), (255, 255, 255))
        for i, im in enumerate(row):
            canvas.paste(im, (i * w, 0))
        tiles.append(canvas)
    tiles[0].save(path, save_all=True, append_images=tiles[1:], duration=100, loop=0)
