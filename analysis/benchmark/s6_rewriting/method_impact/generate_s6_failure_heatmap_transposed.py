#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


ROOT = Path(__file__).resolve().parent
ASSOCIATION_CSV = ROOT / "s6_failure_associated_pair_changes.csv"
OUTPUT_PNG = ROOT / "fig_s6_fn_associated_pair_changes_transposed.png"
OUTPUT_PDF = ROOT / "fig_s6_fn_associated_pair_changes_transposed.pdf"
VALUES_CSV = ROOT / "s6_fn_associated_pair_changes_transposed_values.csv"

METHODS = ["Fuzzer", "UTA", "MPG"]
DETECTORS = ["SecureNet", "V3"]
PAIR_ORDER = [
    "A-A",
    "A-L",
    "A-R",
    "A-SP",
    "A-S",
    "A-C",
    "L-L",
    "L-R",
    "L-SP",
    "L-S",
    "L-C",
    "R-R",
    "R-SP",
    "R-S",
    "R-C",
    "SP-SP",
    "SP-S",
    "SP-C",
    "S-S",
    "S-C",
    "C-C",
]

PAIR_LABELS = [f"({pair.split('-', 1)[0]}, {pair.split('-', 1)[1]})" for pair in PAIR_ORDER]


def load_values():
    rows = []
    with ASSOCIATION_CSV.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            if row["method"] not in METHODS:
                continue
            if row["detector_label"] not in DETECTORS:
                continue
            if row["pair"] not in PAIR_ORDER:
                continue
            q_value = float(row["q_value"])
            row["row_label"] = row["method"] + ", " + (
                "SN" if row["detector_label"] == "SecureNet" else row["detector_label"]
            )
            row["display_effect"] = (
                str(float(row["rank_biserial_effect"])) if q_value < 0.05 else ""
            )
            rows.append(row)
    with VALUES_CSV.open("w", newline="", encoding="utf-8") as output_file:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
        }
    )
    assoc = load_values()
    rows = [(method, detector) for method in METHODS for detector in DETECTORS]
    row_labels = [f"{method}, {'SN' if detector == 'SecureNet' else detector}" for method, detector in rows]

    matrix = np.full((len(rows), len(PAIR_ORDER)), np.nan)
    for row_index, (method, detector) in enumerate(rows):
        for col_index, pair in enumerate(PAIR_ORDER):
            selected = [
                row for row in assoc
                if row["method"] == method
                and row["detector_label"] == detector
                and row["pair"] == pair
            ]
            if not selected or not selected[0]["display_effect"]:
                continue
            matrix[row_index, col_index] = float(selected[0]["display_effect"])

    masked = np.ma.masked_invalid(matrix)
    cmap = LinearSegmentedColormap.from_list(
        "fn_effect_ai_difference_blue_yellow",
        [
            (0.0, "#f6d681"),
            (0.47, "#fff7df"),
            (0.50, "#f7f7f7"),
            (0.53, "#eaf2f1"),
            (1.0, "#003b4d"),
        ],
    )
    cmap.set_bad("#eeeeee")
    norm = TwoSlopeNorm(vmin=-0.32, vcenter=0.0, vmax=0.32)

    figure, axis = plt.subplots(figsize=(18.6, 5.9))
    image = axis.imshow(masked, cmap=cmap, norm=norm, aspect="auto")

    axis.set_xticks(np.arange(len(PAIR_ORDER)))
    axis.set_xticklabels(PAIR_LABELS, fontsize=14.5, rotation=0, ha="center")
    axis.set_yticks(np.arange(len(rows)))
    axis.set_yticklabels(row_labels, fontsize=21)
    axis.tick_params(axis="x", pad=6)
    axis.tick_params(axis="y", pad=7)

    axis.set_xticks(np.arange(-0.5, len(PAIR_ORDER), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", bottom=False, left=False)

    for row_index in range(len(rows)):
        for col_index in range(len(PAIR_ORDER)):
            value = matrix[row_index, col_index]
            if np.isnan(value):
                axis.text(col_index, row_index, "·", ha="center", va="center", fontsize=17, color="#666666")
            else:
                axis.text(
                    col_index,
                    row_index,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="black",
                )

    for row_index in [1.5, 3.5]:
        axis.axhline(row_index, color="black", linewidth=1.0)

    axis.set_xlabel("Persuasion pair", fontsize=21, labelpad=12)
    axis.set_ylabel("Method, detector", fontsize=22, labelpad=12)
    axis.set_title(
        "Rewriting-induced pair changes enriched among false negatives",
        fontsize=26,
        pad=18,
    )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.022, pad=0.015)
    colorbar.set_label("Rank-biserial effect size\n(gray/dot = q >= 0.05)", fontsize=20, labelpad=10)
    colorbar.ax.tick_params(labelsize=18)

    figure.subplots_adjust(left=0.115, right=0.925, bottom=0.19, top=0.82)
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.06)
    figure.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)


if __name__ == "__main__":
    main()
