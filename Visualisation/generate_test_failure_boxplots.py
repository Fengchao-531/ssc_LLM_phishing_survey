from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def draw_jittered_points(axis, values_by_group: list[np.ndarray], colors: list[str]):
    rng = np.random.default_rng(7)
    for index, (values, color) in enumerate(zip(values_by_group, colors, strict=True), start=1):
        if len(values) == 0:
            continue
        jitter = rng.normal(0.0, 0.05, size=len(values))
        axis.scatter(
            np.full(len(values), index, dtype=float) + jitter,
            values,
            s=8,
            c=color,
            alpha=0.18,
            linewidths=0,
        )


def draw_fig4_score_distribution(frame, output_path: Path, group_order: list[str]):
    figure, axes = plt.subplots(1, 2, figsize=(14.8, 6.4), sharex=True)
    group_colors = ["#bbbbbb", "#2ca02c", "#bbbbbb", "#1f77b4"]

    score_values = [frame.loc[frame["group"] == group, "surrogate_score"].to_numpy() for group in group_order]
    margin_values = [frame.loc[frame["group"] == group, "decision_margin"].to_numpy() for group in group_order]

    score_box = axes[0].boxplot(score_values, patch_artist=True, tick_labels=group_order, showfliers=False)
    margin_box = axes[1].boxplot(margin_values, patch_artist=True, tick_labels=group_order, showfliers=False)
    for patch, color in zip(score_box["boxes"], group_colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    for patch, color in zip(margin_box["boxes"], group_colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)

    draw_jittered_points(axes[0], score_values, group_colors)
    draw_jittered_points(axes[1], margin_values, group_colors)

    axes[0].axhline(0.5, color="black", linewidth=1.3, linestyle="--")
    axes[1].axhline(0.0, color="black", linewidth=1.3, linestyle="--")
    axes[0].set_title("Surrogate score", fontsize=12)
    axes[1].set_title("Decision margin", fontsize=12)
    axes[0].set_ylabel("Value", fontsize=10)
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(alpha=0.10, linewidth=0.5)

    figure.suptitle("Figure 4. Score / Distance-to-Threshold Distribution", y=0.98, fontsize=15)
    figure.tight_layout(rect=(0, 0.02, 1, 0.95))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def draw_fig7_mixed_score_distribution(mixed_frame: pd.DataFrame, test_ids: np.ndarray, output_path: Path):
    test_frame = mixed_frame[mixed_frame["row_id"].isin(test_ids)].copy()
    group_order = ["HW-benign", "HW-phishing", "LLM-benign", "LLM-phishing"]
    colors = ["#7f7f7f", "#2ca02c", "#9c9c9c", "#1f77b4"]
    values = [test_frame.loc[test_frame["truth_group"] == group, "surrogate_score_holdout"].to_numpy() for group in group_order]

    figure, axis = plt.subplots(figsize=(10.8, 6.2))
    box = axis.boxplot(values, patch_artist=True, tick_labels=group_order, showfliers=False)
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    draw_jittered_points(axis, values, colors)
    axis.axhline(0.5, color="black", linewidth=1.3, linestyle="--")
    axis.set_ylabel("Surrogate score", fontsize=10)
    axis.set_title("Figure 7. Mixed-data Surrogate Score by True Label", fontsize=14)
    axis.tick_params(axis="x", rotation=20)
    axis.grid(alpha=0.12, linewidth=0.5)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
