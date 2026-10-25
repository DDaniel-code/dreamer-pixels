#!/usr/bin/env python3
"""Plot the independent 100k baseline and overnight seed-0 run.

The runs are shown separately. This script intentionally does not concatenate
their training curves or treat them as independent training seeds.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_progress(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    steps = np.array([float(row["Diagnostics/CumSteps"]) * 2 for row in rows])
    returns = np.array([float(row["ReturnAverage"]) for row in rows])
    return steps, returns


def read_eval(path: Path):
    grouped: dict[str, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["policy"], []).append(float(row["return"]))
    return grouped


def plot_training(old_progress: Path, new_progress: Path, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for axis, progress, title in (
        (axes[0], old_progress, "Independent 100k run"),
        (axes[1], new_progress, "Independent overnight run"),
    ):
        steps, returns = read_progress(progress)
        axis.plot(steps, returns, color="#245b8a", linewidth=1.2)
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("Environment steps", fontsize=8)
        axis.grid(alpha=0.25)
        axis.tick_params(labelsize=7)
    axes[0].set_ylabel("Training ReturnAverage", fontsize=8)
    fig.suptitle("Training diagnostics (runs are not concatenated)", fontsize=10)
    fig.tight_layout(pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_evaluation(old_eval: Path, new_eval: Path, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for axis, evaluation, title in (
        (axes[0], old_eval, "100k baseline"),
        (axes[1], new_eval, "Overnight bounded run"),
    ):
        grouped = read_eval(evaluation)
        labels = ["random", "initial", "dreamer"]
        positions = np.arange(len(labels))
        means = [statistics.mean(grouped[label]) for label in labels]
        stds = [statistics.stdev(grouped[label]) for label in labels]
        axis.errorbar(positions, means, yerr=stds, fmt="none", ecolor="#333333", capsize=3)
        for position, label in zip(positions, labels):
            axis.scatter(np.full(len(grouped[label]), position), grouped[label],
                         color="#111111", s=10, zorder=3)
        axis.scatter(positions, means, color="#c44e52", s=24, zorder=4)
        axis.set_xticks(positions, ["Random", "Initial", "Final"])
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("Policy", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(labelsize=7)
    axes[0].set_ylabel("Episodic return", fontsize=8)
    fig.suptitle("Fixed-seed evaluation (mean, sample std., and raw episodes)", fontsize=10)
    fig.tight_layout(pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot_training(
        args.old_root / "train_run/run_0/progress.csv",
        args.new_root / "long_train_run/run_0/progress.csv",
        args.output_dir / "training_return_runs.png",
    )
    plot_evaluation(
        args.old_root / "eval_episodes.csv",
        args.new_root / "eval_episodes.csv",
        args.output_dir / "evaluation_comparison_runs.png",
    )


if __name__ == "__main__":
    main()
