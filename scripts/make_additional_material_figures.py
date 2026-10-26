"""Build figures for the additional Dreamer midterm material."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "figures" / "additional_material_returns.png"

# Seed 0 values are independently present in the local overnight artifacts.
# Seed 1 is an aggregate supplied by a teammate, so no per-episode points are
# fabricated for that row.
labels = ["Seed 0\n100k env", "Seed 0\n330k snapshot", "Hao Deng\nseed 1, 100k"]
means = np.array([181.7011, 322.0840, 227.0931])
stds = np.array([27.5705, 9.6868, 29.4737])
colors = ["#4c78a8", "#f58518", "#54a24b"]

fig, axes = plt.subplots(
    1, 2, figsize=(8.2, 3.2), gridspec_kw={"width_ratios": [1.2, 1]}
)

ax = axes[0]
x = np.arange(len(labels))
ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xticks(x, labels, fontsize=8)
ax.set_ylabel("Episodic return")
ax.set_title("Final Dreamer evaluation")
ax.set_ylim(0, 380)
ax.grid(axis="y", alpha=0.25)
for i, (mean, std) in enumerate(zip(means, stds)):
    ax.text(i, mean + std + 8, f"{mean:.1f} +/- {std:.1f}", ha="center", va="bottom", fontsize=7)

ax = axes[1]
budget_x = np.array([100_000, 330_000])
budget_y = np.array([181.7011, 322.0840])
budget_e = np.array([27.5705, 9.6868])
ax.errorbar(budget_x, budget_y, yerr=budget_e, fmt="o-", color="#4c78a8", capsize=4)
ax.set_xlabel("Seed-0 environment steps")
ax.set_ylabel("Return")
ax.set_title("Seed-0 budget comparison")
ax.set_xlim(80_000, 350_000)
ax.set_ylim(0, 380)
ax.ticklabel_format(axis="x", style="plain")
ax.grid(alpha=0.25)

fig.suptitle("Cartpole swingup; error bars are episode sample standard deviation", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.92))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=220, bbox_inches="tight")
plt.close(fig)
print(OUT)
