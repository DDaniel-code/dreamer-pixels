"""Build report figures from the downloaded Kaggle pilot artifacts.

This script only reads the exported CSV files.  It does not run an
environment or start training.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "kaggle_100k_final"
OUT = ROOT / "output" / "pdf" / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    progress = read_csv(ARTIFACTS / "train_run" / "run_0" / "progress.csv")
    env_steps = np.array([int(row["Diagnostics/CumSteps"]) * 2 for row in progress])
    return_average = np.array([float(row["ReturnAverage"]) for row in progress])
    valid = np.isfinite(return_average)

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=180)
    ax.plot(env_steps[valid], return_average[valid], color="#1f77b4", linewidth=1.8)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Training return (window average)")
    ax.set_title("Cartpole swingup - single-seed 100k-env-step pilot")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "training_return_vs_env_steps.png", bbox_inches="tight")
    plt.close(fig)

    rows = read_csv(ARTIFACTS / "eval_episodes.csv")
    labels = {"random": "Random", "initial": "Training-initial", "dreamer": "Final Dreamer"}
    order = ["random", "initial", "dreamer"]
    values = {
        policy: np.array([float(row["return"]) for row in rows if row["policy"] == policy])
        for policy in order
    }
    if any(len(values[policy]) != 10 for policy in order):
        raise ValueError("expected ten evaluation episodes for every policy")
    means = np.array([values[policy].mean() for policy in order])
    sample_stds = np.array([values[policy].std(ddof=1) for policy in order])

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=180)
    x = np.arange(len(order))
    ax.bar(x, means, yerr=sample_stds, capsize=4, color="#377eb8", alpha=0.88)
    rng = np.random.default_rng(0)
    for index, policy in enumerate(order):
        jitter = rng.uniform(-0.08, 0.08, size=len(values[policy]))
        ax.scatter(
            np.full(len(values[policy]), index) + jitter,
            values[policy],
            color="#222222",
            s=13,
            zorder=3,
            label="Raw episode return" if index == 0 else None,
        )
    ax.set_xticks(x, [labels[policy] for policy in order])
    ax.set_ylabel("Episodic return")
    ax.set_title("Fixed-seed evaluation (seeds 1000-1009)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "evaluation_comparison.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
