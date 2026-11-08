"""Plot learning curves of all runs: python plot.py runs/sm_sparse runs/ppo_sparse ... (default: every run)."""

import csv
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def col(rows, key):
    return [float(r[key]) for r in rows if r.get(key) not in (None, "")]


def main():
    runs = sys.argv[1:] or sorted(glob.glob("runs/*"))
    fig, ax = plt.subplots(1, 4, figsize=(19, 4))
    for run in runs:
        name = os.path.basename(run.rstrip("/"))
        ev, lg = read(os.path.join(run, "eval.csv")), read(os.path.join(run, "log.csv"))
        if ev:
            ax[0].plot(col(ev, "env_steps"), col(ev, "eval_success_full"), marker="o", label=name)
        if lg:
            ax[1].plot(col(lg, "env_steps"), col(lg, "train_final_coverage"), label=name)
            ax[2].plot(col(lg, "env_steps"), col(lg, "train_success"), label=name)
            ax[3].plot(col(lg, "env_steps"), col(lg, "stage"), label=name)
    ax[0].set_title("Eval success (full Push-T reset)")
    ax[1].set_title("Goal coverage at episode end (training)")
    ax[1].axhline(0.237, ls="--", c="gray", lw=1, label="do nothing (stage 0)")
    ax[2].set_title("Training success (rolling, curriculum)")
    ax[3].set_title("Curriculum stage")
    for a in ax:
        a.set_xlabel("environment steps")
        a.grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results.png", dpi=120)
    print("saved results.png")


if __name__ == "__main__":
    main()
