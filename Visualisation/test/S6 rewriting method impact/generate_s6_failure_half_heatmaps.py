#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


ROOT = Path(__file__).resolve().parent
ASSOCIATION_CSV = ROOT / "s6_failure_associated_pair_changes.csv"
OUTPUT_PNG = ROOT / "fig_s6_fn_associated_pair_changes_half_heatmaps.png"
OUTPUT_PDF = ROOT / "fig_s6_fn_associated_pair_changes_half_heatmaps.pdf"
VALUES_CSV = ROOT / "s6_fn_associated_pair_changes_half_heatmap_values.csv"

METHODS = ["Fuzzer", "UTA", "MPG"]
DETECTORS = ["SecureNet", "V3"]
DETECTOR_LABELS = {"SecureNet": "SN", "V3": "V3"}
PRINCIPLES = [
    ("A", "Authority"),
    ("L", "Liking"),
    ("R", "Reciprocity"),
    ("SP", "Social Proof"),
    ("S", "Scarcity"),
    ("C", "Commitment"),
]


def load_values() -> pd.DataFrame:
    assoc = pd.read_csv(ASSOCIATION_CSV)
    assoc = assoc[
        assoc["method"].isin(METHODS)
        & assoc["detector_label"].isin(DETECTORS)
    ].copy()
    assoc["display_effect"] = np.where(
        assoc["q_value"].lt(0.05),
        assoc["rank_biserial_effect"],
        np.nan,
    )
    assoc.to_csv(VALUES_CSV, index=False)
    return assoc


def draw_half_heatmap(
    axis: plt.Axes,
    data: pd.DataFrame,
    *,
    method: str,
    detector: str,
    cmap: LinearSegmentedColormap,
    norm: TwoSlopeNorm,
) -> None:
    size = len(PRINCIPLES)
    matrix = np.full((size, size), np.nan)
    q_matrix = np.full((size, size), np.nan)
    for row_index, (row_short, _) in enumerate(PRINCIPLES):
        for col_index, (col_short, _) in enumerate(PRINCIPLES):
            if col_index > row_index:
                continue
            pair = f"{col_short}-{row_short}" if col_index != row_index else f"{row_short}-{row_short}"
            selected = data[
                data["method"].eq(method)
                & data["detector_label"].eq(detector)
                & data["pair"].eq(pair)
            ]
            if selected.empty:
                continue
            matrix[row_index, col_index] = float(selected["display_effect"].iloc[0])
            q_matrix[row_index, col_index] = float(selected["q_value"].iloc[0])

    masked = np.ma.masked_invalid(matrix)
    image = axis.imshow(masked, cmap=cmap, norm=norm, aspect="equal")

    for row_index in range(size):
        for col_index in range(size):
            if col_index > row_index:
                axis.add_patch(
                    plt.Rectangle(
                        (col_index - 0.5, row_index - 0.5),
                        1,
                        1,
                        facecolor="white",
                        edgecolor="white",
                        linewidth=0,
                    )
                )
                continue
            value = matrix[row_index, col_index]
            if np.isnan(value):
                axis.text(
                    col_index,
                    row_index,
                    "·",
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="#666666",
                )
            else:
                axis.text(
                    col_index,
                    row_index,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=11.5,
                    color="black",
                )

    axis.set_xticks(np.arange(size))
    axis.set_yticks(np.arange(size))
    axis.set_xticklabels([label for _, label in PRINCIPLES], fontsize=12, rotation=35, ha="right")
    axis.set_yticklabels([label for _, label in PRINCIPLES], fontsize=12)
    axis.tick_params(axis="x", pad=3)
    axis.tick_params(axis="y", pad=3)
    axis.set_xticks(np.arange(-0.5, size, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, size, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.4)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.set_title(f"{method}, {DETECTOR_LABELS[detector]}", fontsize=15, pad=8)
    for spine in axis.spines.values():
        spine.set_visible(False)
    return image


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "axes.linewidth": 1.0,
        }
    )
    data = load_values()
    cmap = LinearSegmentedColormap.from_list(
        "ai_difference_blue_yellow",
        [
            (0.0, "#f6d681"),
            (0.47, "#fff7df"),
            (0.50, "#eeeeee"),
            (0.53, "#eaf2f1"),
            (1.0, "#003b4d"),
        ],
    )
    cmap.set_bad("#eeeeee")
    norm = TwoSlopeNorm(vmin=-0.32, vcenter=0.0, vmax=0.32)

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(13.6, 8.5),
        gridspec_kw={"wspace": 0.05, "hspace": 0.38},
    )
    image = None
    for row_index, detector in enumerate(DETECTORS):
        for col_index, method in enumerate(METHODS):
            image = draw_half_heatmap(
                axes[row_index, col_index],
                data,
                method=method,
                detector=detector,
                cmap=cmap,
                norm=norm,
            )
            if col_index > 0:
                axes[row_index, col_index].set_yticklabels([])
            if row_index == 0:
                axes[row_index, col_index].set_xticklabels([])

    assert image is not None
    colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.026, pad=0.018)
    colorbar.set_label("Rank-biserial effect size\n(gray/dot = q >= 0.05)", fontsize=14, labelpad=10)
    colorbar.ax.tick_params(labelsize=13)

    figure.suptitle(
        "FN-associated rewriting-induced persuasion-pair changes",
        fontsize=20,
        y=0.985,
    )
    figure.subplots_adjust(left=0.08, right=0.90, bottom=0.08, top=0.91)
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.06)
    figure.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)


if __name__ == "__main__":
    main()
