"""Create a pyplot comparison of the upgraded LSTM and published results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path(__file__).resolve().parents[1] / "output1" / "figures"

models = ["New LSTM", "AlgTeam", "Ensemble", "AEC-Net", "Tricog"]
metrics = {
    "AUROC": [0.72676, 0.764, 0.790, 0.789, 0.721],
    "AUPRC": [0.08095, 0.084, 0.094, 0.093, 0.045],
    "F1-score": [0.13231, 0.139, 0.113, 0.131, 0.113],
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    colors = ["#1f77b4"] + ["#9aa0a6"] * (len(models) - 1)
    x = np.arange(len(models))

    for axis, (metric, values) in zip(axes, metrics.items()):
        bars = axis.bar(x, values, color=colors, edgecolor="#333333", linewidth=0.6)
        axis.set_title(metric, fontsize=13, fontweight="bold")
        axis.set_xticks(x, models, rotation=35, ha="right")
        axis.set_ylim(0, max(values) * 1.25)
        axis.set_ylabel("Score")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(values) * 0.025,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(
        "Upgraded LSTM Compared with Selected PhysioNet/CinC 2019 Results",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "New LSTM: internal test split; published models: reported Test Set A values",
        ha="center",
        fontsize=9,
    )
    fig.savefig(OUTPUT / "lstm_competition_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

