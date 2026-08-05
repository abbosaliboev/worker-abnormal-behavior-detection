"""
Generate one metrics chart per detector (Accuracy + Precision + Recall + F1).

Usage:
    python generate_plots.py
"""

import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results", exist_ok=True)

DETECTORS = [
    {
        "name":    "Fall Detection",
        "file":    "results/fall_metrics.png",
        "color":   "#E74C3C",
        "dataset": "UP-Fall Dataset  |  4 Subjects  |  LOOCV",
        "metrics": {
            "Accuracy":  92.40,
            "Precision": 87.6,
            "Recall":    98.2,
            "F1-score":  92.3,
        },
    },
    {
        "name":    "Unsafe Running Detection",
        "file":    "results/running_metrics.png",
        "color":   "#F39C12",
        "dataset": "KTH Action Dataset  |  25 Subjects  |  LOOCV",
        "metrics": {
            "Accuracy":  90.99,
            "Precision": 95.6,
            "Recall":    86.1,
            "F1-score":  90.5,
        },
    },
    {
        "name":    "Long-time Inactivity Detection",
        "file":    "results/inactivity_metrics.png",
        "color":   "#2ECC71",
        "dataset": "UP-Fall Dataset  |  4 Subjects  |  LOOCV",
        "metrics": {
            "Accuracy":  95.83,
            "Precision": 96.4,
            "Recall":    95.8,
            "F1-score":  95.8,
        },
    },
]

for d in DETECTORS:
    labels = list(d["metrics"].keys())
    values = list(d["metrics"].values())

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(labels, values, color=d["color"], width=0.5, alpha=0.88)
    ax.axhline(90, color="black", linestyle="--", linewidth=2, label="90% target")

    ax.set_ylim(75, 105)
    ax.set_ylabel("Score (%)", fontsize=13)
    ax.set_title(f"{d['name']}\n{d['dataset']}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=13)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(d["file"], dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {d['file']}")
