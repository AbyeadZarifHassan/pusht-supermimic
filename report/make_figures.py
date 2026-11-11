"""Make the report figures from the recorded evaluation results.

Run from the repository root:  python report/make_figures.py
Optional: if docs/best_model.gif exists, a strip of frames is also written.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pusht_rl.env import AGENT_RADIUS, GOAL_POSE, t_polygon  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

INK, MID, LIGHT, FAINT = "#183247", "#246A9B", "#9CADBD", "#E4ECF2"
TEAL, AMBER, PALE = "#187F83", "#B77A24", "#EDF4FA"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": LIGHT, "axes.labelcolor": INK,
    "xtick.color": "#647181", "ytick.color": "#647181", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": FAINT, "grid.linewidth": 0.6, "legend.frameon": False,
})

# ---------------------------------------------------------------- recorded data (eval.csv files)
H16_2M = np.array([  # steps, stage, success, coverage, length
    [100096, 0, 0.00, 0.106, 300.0], [200192, 0, 0.00, 0.340, 300.0], [300032, 0, 0.01, 0.367, 297.3],
    [400128, 0, 0.09, 0.546, 288.0], [500224, 1, 0.15, 0.554, 274.6], [600064, 1, 0.15, 0.595, 267.2],
    [700160, 2, 0.00, 0.490, 300.0], [800000, 3, 0.14, 0.672, 271.9], [900096, 4, 0.23, 0.582, 270.3],
    [1000192, 5, 0.64, 0.813, 181.7], [1100032, 7, 0.10, 0.774, 279.9], [1200128, 9, 0.61, 0.825, 169.6],
    [1300224, 10, 0.51, 0.821, 194.7], [1400064, 10, 0.47, 0.844, 189.3], [1500160, 10, 0.85, 0.880, 104.8],
    [1600000, 10, 0.77, 0.877, 118.0], [1700096, 10, 0.87, 0.890, 99.8], [1800192, 10, 0.19, 0.503, 264.0],
    [1900032, 10, 0.43, 0.666, 214.4], [2000128, 10, 0.79, 0.934, 115.3]])
SEED1 = np.array([
    [100096, 0, 0.00, 0.133], [200192, 0, 0.00, 0.282], [300032, 0, 0.03, 0.364], [400128, 0, 0.03, 0.369],
    [500224, 1, 0.03, 0.425], [600064, 2, 0.06, 0.321], [700160, 3, 0.11, 0.608], [800000, 3, 0.02, 0.401],
    [900096, 4, 0.00, 0.257]])
NOCRITIC = np.array([[100096, 0, 0.0, 0.055], [200192, 0, 0.0, 0.229], [300032, 0, 0.02, 0.354],
                     [400128, 1, 0.11, 0.556], [500224, 1, 0.02, 0.362]])
SPARSE = np.array([[s * 100000, 0, 0.0, c] for s, c in zip(range(1, 11),
                   [0.062, 0.058, 0.050, 0.054, 0.054, 0.058, 0.054, 0.056, 0.061, 0.058])])


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1. the task
def fig_task():
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    titles = ["Curriculum stage 0", "Curriculum stage 10", "Original Push-T reset"]
    halves = [30, 130, None]
    for ax, title, half in zip(axes, titles, halves):
        ax.set_xlim(0, 512); ax.set_ylim(512, 0); ax.set_aspect("equal"); ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(LIGHT)
        if half is None:
            ax.add_patch(plt.Rectangle((100, 100), 300, 300, fill=False, ls=":", ec=MID, lw=0.8))
            pos = rng.uniform(100, 400, size=(3, 2))
        else:
            ax.add_patch(plt.Rectangle((256 - half, 256 - half), 2 * half, 2 * half, fill=False, ls=":", ec=MID, lw=0.8))
            pos = rng.uniform(256 - half, 256 + half, size=(3, 2))
        goal = t_polygon(*GOAL_POSE)
        ax.add_patch(Polygon(np.array(goal.exterior.coords), closed=True, fc="#DDEFE9", ec=TEAL, lw=1.2, ls="--"))
        for i, (x, y) in enumerate(pos):
            poly = t_polygon(x, y, rng.uniform(-np.pi, np.pi))
            ax.add_patch(Polygon(np.array(poly.exterior.coords), closed=True, fc=MID if i == 0 else LIGHT, ec="none", alpha=0.95 if i == 0 else 0.55))
        ax.add_patch(Circle((70, 440), AGENT_RADIUS, fc=AMBER))
        ax.set_title(title, fontsize=9, color=INK)
    axes[0].text(70, 485, "pusher", ha="center", fontsize=7, color=MID)
    save(fig, "task.pdf")


# ---------------------------------------------------------------- 2. comparison at 500k
def fig_compare():
    names = ["Do nothing", "PPO, sparse", "PPO, dense", "SuperMimic, sparse, H=16",
             "SuperMimic, H=32", "SuperMimic, H=16, no critic", "SuperMimic, H=8", "SuperMimic, H=16"]
    succ = [0, 0, 0, 0, 0, 2, 15, 15]
    cov = [0.043, 0.052, 0.064, 0.054, 0.121, 0.362, 0.507, 0.554]
    y = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    shade = [LIGHT] * 4 + [MID] * 3 + [TEAL]
    ax[0].barh(y, succ, color=shade, height=0.6)
    ax[1].barh(y, cov, color=shade, height=0.6)
    ax[0].set_yticks(y); ax[0].set_yticklabels(names)
    ax[0].set_xlabel("Evaluation success (%)"); ax[1].set_xlabel("Final goal coverage")
    for a in ax:
        a.grid(axis="y", visible=False)
    for i, v in enumerate(succ):
        ax[0].text(v + 0.4, i, f"{v}%", va="center", fontsize=7, color=MID)
    for i, v in enumerate(cov):
        ax[1].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=7, color=MID)
    ax[0].set_xlim(0, 20); ax[1].set_xlim(0, 0.7)
    save(fig, "compare_500k.pdf")


# ---------------------------------------------------------------- 3. long run
def fig_long():
    s = H16_2M[:, 0] / 1e6
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
    ax[0].plot(s, H16_2M[:, 2] * 100, "-o", color=MID, ms=3, lw=1.3, label="Seed 0, 2M steps")
    ax[0].plot(SEED1[:, 0] / 1e6, SEED1[:, 2] * 100, "-s", color=AMBER, ms=3, lw=1.2, label="Seed 1, shorter run")
    ax[0].set_ylabel("Evaluation success (%)"); ax[0].set_xlabel("Environment steps (millions)")
    ax[0].legend(fontsize=7, loc="upper left")
    ax[1].plot(s, H16_2M[:, 3], "-o", color=MID, ms=3, lw=1.3)
    ax[1].plot(SEED1[:, 0] / 1e6, SEED1[:, 3], "-s", color=AMBER, ms=3, lw=1.2)
    ax[1].set_ylabel("Final goal coverage"); ax[1].set_xlabel("Environment steps (millions)")
    tw = ax[1].twinx()
    tw.step(s, H16_2M[:, 1], where="post", color=TEAL, lw=1.0, ls=":")
    tw.set_ylabel("Curriculum stage (dotted)", color=TEAL); tw.set_ylim(0, 11); tw.grid(False)
    tw.spines["right"].set_visible(True); tw.spines["top"].set_visible(False)
    for a in ax:
        a.set_xlim(0, 2.05)
    save(fig, "long_run.pdf")


# ---------------------------------------------------------------- 4. critic and sparse
def fig_ablate():
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ax.plot(H16_2M[:5, 0] / 1e3, H16_2M[:5, 3], "-o", color=MID, ms=3, label="H=16 with critic")
    ax.plot(NOCRITIC[:, 0] / 1e3, NOCRITIC[:, 3], "-s", color=AMBER, ms=3, label="H=16 without critic")
    ax.plot(SPARSE[:5, 0] / 1e3, SPARSE[:5, 3], "-^", color=TEAL, ms=3, label="H=16 sparse reward")
    ax.axhline(0.043, color=LIGHT, lw=0.8, ls="--")
    ax.text(505, 0.055, "do nothing", fontsize=7, color=MID, ha="right")
    ax.set_xlabel("Environment steps (thousands)"); ax.set_ylabel("Final goal coverage")
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "ablations.pdf")


# ---------------------------------------------------------------- 5. frames of the best policy
def fig_frames():
    gif = os.path.join(os.path.dirname(__file__), "..", "docs", "best_model.gif")
    if not os.path.exists(gif):
        print("docs/best_model.gif not found, skipping rollout frames")
        return
    from PIL import Image
    im = Image.open(gif)
    n = im.n_frames
    picks = np.linspace(0, n - 1, 6).astype(int)
    frames = []
    for k in picks:
        im.seek(int(k))
        f = im.convert("RGB")
        w = f.size[0] // 4  # first of the four side-by-side episodes
        frames.append(f.crop((0, 0, w, f.size[1])))
    fig, ax = plt.subplots(1, 6, figsize=(7.0, 1.4))
    for a, f, k in zip(ax, frames, picks):
        a.imshow(f); a.set_xticks([]); a.set_yticks([]); a.grid(False)
        a.set_title(f"step {k}", fontsize=7, color=MID)
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_color(LIGHT)
    save(fig, "rollout_frames.png")


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fig_ci():
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    for data, col, lab, mk in [(H16_2M, MID, "Seed 0", "o"), (SEED1, AMBER, "Seed 1", "s")]:
        s = data[:, 0] / 1e6
        k = np.round(data[:, 2] * 100)
        lo, hi = wilson(k, 100)
        ax.fill_between(s, lo * 100, hi * 100, color=col, alpha=0.18, lw=0)
        ax.plot(s, k, "-" + mk, color=col, ms=3, lw=1.2, label=lab + " (95% Wilson interval shaded)")
    ax.set_xlabel("Environment steps (millions)"); ax.set_ylabel("Evaluation success (%)")
    ax.set_xlim(0, 2.05); ax.set_ylim(0, 100)
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "success_ci.pdf")


def fig_ci_bars():
    rows = [("PPO, dense", 0), ("SuperMimic, H=32", 0), ("SuperMimic, H=16, no critic", 2),
            ("SuperMimic, H=8", 15), ("SuperMimic, H=16", 15)]
    fig, ax = plt.subplots(figsize=(5.2, 2.2))
    for i, (name, k) in enumerate(rows):
        lo, hi = wilson(k, 100)
        ax.errorbar(k, i, xerr=[[k - lo * 100], [hi * 100 - k]], fmt="o", color=TEAL if k >= 15 else MID,
                    ecolor=LIGHT, elinewidth=2, capsize=3, ms=5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Evaluation success at 500k steps (%), 95% Wilson interval"); ax.set_xlim(-1, 30)
    ax.grid(axis="y", visible=False)
    save(fig, "ci_500k.pdf")


if __name__ == "__main__":
    fig_ci(); fig_ci_bars()
    fig_task(); fig_compare(); fig_long(); fig_ablate(); fig_frames()
