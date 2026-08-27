#!/usr/bin/env python3
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


ROOT = Path(__file__).resolve().parent
FEATURE_SHIFT_CSV = ROOT / "s6_method_pair_feature_shift.csv"
FN_TP_CSV = ROOT / "s6_rewritten_fn_tp_pair_characteristics.csv"

OUTPUT_A_PNG = ROOT / "fig_s6_rewriting_two_panel_main_a.png"
OUTPUT_A_PDF = ROOT / "fig_s6_rewriting_two_panel_main_a.pdf"
OUTPUT_B_PNG = ROOT / "fig_s6_rewriting_two_panel_main_b.png"
OUTPUT_B_PDF = ROOT / "fig_s6_rewriting_two_panel_main_b.pdf"
HEATMAP_VALUES_CSV = ROOT / "s6_two_panel_heatmap_values.csv"

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
FOCUS_PAIRS = {"A-S", "S-S", "SP-S"}


def read_rows(path):
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def to_float(value):
    try:
        if value is None or str(value).strip() == "":
            return math.nan
        return float(value)
    except ValueError:
        return math.nan


def stars(q_value):
    if not math.isfinite(q_value) or q_value >= 0.05:
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    return "*"


def load_panel_a():
    rows = read_rows(FEATURE_SHIFT_CSV)
    lookup = {
        (row["method"], row["pair"]): to_float(row["delta_rewritten_minus_original"])
        for row in rows
    }
    matrix = np.full((len(METHODS), len(PAIR_ORDER)), np.nan)
    for row_index, method in enumerate(METHODS):
        for col_index, pair in enumerate(PAIR_ORDER):
            matrix[row_index, col_index] = lookup.get((method, pair), math.nan)
    return matrix


def load_panel_b():
    rows = read_rows(FN_TP_CSV)
    lookup = {
        (row["method"], row["detector_label"], row["pair"]): row
        for row in rows
    }
    row_order = [(method, detector) for method in METHODS for detector in DETECTORS]
    matrix = np.full((len(row_order), len(PAIR_ORDER)), np.nan)
    q_matrix = np.full((len(row_order), len(PAIR_ORDER)), np.nan)
    for row_index, (method, detector) in enumerate(row_order):
        for col_index, pair in enumerate(PAIR_ORDER):
            row = lookup.get((method, detector, pair))
            if not row:
                continue
            matrix[row_index, col_index] = to_float(row["rank_biserial_effect"])
            q_matrix[row_index, col_index] = to_float(row["q_value"])
    return row_order, matrix, q_matrix


def export_values(delta_matrix, row_order, effect_matrix, q_matrix):
    rows = []
    for row_index, method in enumerate(METHODS):
        for col_index, pair in enumerate(PAIR_ORDER):
            rows.append(
                {
                    "panel": "a_rewriting_induced_persuasion_difference",
                    "row": method,
                    "method": method,
                    "detector": "",
                    "pair": pair,
                    "value": delta_matrix[row_index, col_index],
                    "q_value": "",
                    "stars": "",
                }
            )
    for row_index, (method, detector) in enumerate(row_order):
        for col_index, pair in enumerate(PAIR_ORDER):
            q_value = q_matrix[row_index, col_index]
            rows.append(
                {
                    "panel": "b_fn_tp_detector_failure_association",
                    "row": "{}-{}".format(method, detector),
                    "method": method,
                    "detector": detector,
                    "pair": pair,
                    "value": effect_matrix[row_index, col_index],
                    "q_value": q_value,
                    "stars": stars(q_value),
                }
            )
    with HEATMAP_VALUES_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["panel", "row", "method", "detector", "pair", "value", "q_value", "stars"],
        )
        writer.writeheader()
        writer.writerows(rows)


def draw_grid(axis, n_rows, n_cols):
    axis.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.6)
    axis.tick_params(which="minor", bottom=False, left=False)
    for col_index, pair in enumerate(PAIR_ORDER):
        if pair in FOCUS_PAIRS:
            axis.axvline(col_index - 0.5, color="#222222", linewidth=1.15)
            axis.axvline(col_index + 0.5, color="#222222", linewidth=1.15)
    for spine in ["left", "bottom", "top", "right"]:
        axis.spines[spine].set_linewidth(1.2)


def draw_panel_a(axis, matrix, cmap):
    vmax = max(0.035, float(np.nanmax(np.abs(matrix))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    image = axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    axis.set_yticks(np.arange(len(METHODS)))
    axis.set_yticklabels(METHODS, fontsize=22)
    axis.set_xticks(np.arange(len(PAIR_ORDER)))
    axis.set_xticklabels(PAIR_ORDER, fontsize=22, rotation=45, ha="right")
    axis.tick_params(axis="x", pad=8)
    axis.tick_params(axis="y", pad=8)
    axis.set_xlabel("Persuasion pair", fontsize=24, labelpad=13)
    axis.set_ylabel("Method", fontsize=23, labelpad=18)
    draw_grid(axis, len(METHODS), len(PAIR_ORDER))
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            axis.text(
                col_index,
                row_index,
                "{:+.3f}".format(value),
                ha="center",
                va="center",
                fontsize=12.5,
                color="black",
            )
    return image


def draw_panel_b(axis, row_order, matrix, q_matrix, cmap):
    vmax = max(0.45, float(np.nanmax(np.abs(matrix))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    image = axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    row_labels = [
        "{}, {}".format(method, "SN" if detector == "SecureNet" else detector)
        for method, detector in row_order
    ]
    axis.set_yticks(np.arange(len(row_order)))
    axis.set_yticklabels(row_labels, fontsize=25)
    axis.set_xticks(np.arange(len(PAIR_ORDER)))
    axis.set_xticklabels(PAIR_ORDER, fontsize=26, rotation=45, ha="right")
    axis.tick_params(axis="x", pad=8)
    axis.tick_params(axis="y", pad=8)
    draw_grid(axis, len(row_order), len(PAIR_ORDER))
    for row_index in [1.5, 3.5]:
        axis.axhline(row_index, color="#222222", linewidth=1.2)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            marker = stars(q_matrix[row_index, col_index])
            rgba = cmap(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if luminance < 0.48 else "black"
            if marker:
                axis.text(
                    col_index,
                    row_index - 0.12,
                    "{:+.2f}".format(value),
                    ha="center",
                    va="center",
                    fontsize=16.5,
                    color=text_color,
                )
                axis.text(
                    col_index,
                    row_index + 0.23,
                    marker,
                    ha="center",
                    va="center",
                    fontsize=16.5,
                    color=text_color,
                )
            else:
                axis.text(
                    col_index,
                    row_index,
                    "{:+.2f}".format(value),
                    ha="center",
                    va="center",
                    fontsize=16.5,
                    color=text_color,
                )
    return image


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
        }
    )
    delta_matrix = load_panel_a()
    row_order, effect_matrix, q_matrix = load_panel_b()
    export_values(delta_matrix, row_order, effect_matrix, q_matrix)

    cmap = LinearSegmentedColormap.from_list(
        "s6_signature_diverging",
        [
            (0.0, "#f3c957"),
            (0.45, "#fff1bf"),
            (0.50, "#ffffff"),
            (0.55, "#d9ecec"),
            (1.0, "#003b4d"),
        ],
    )
    cmap.set_bad("#eeeeee")

    figure_a, axis_a = plt.subplots(figsize=(21.0, 4.2))
    image_a = draw_panel_a(axis_a, delta_matrix, cmap)
    cbar_a = figure_a.colorbar(image_a, ax=axis_a, fraction=0.018, pad=0.010)
    cbar_a.set_label("Delta pair score", fontsize=18, labelpad=10)
    cbar_a.ax.tick_params(labelsize=16)
    figure_a.subplots_adjust(left=0.095, right=0.925, top=0.98, bottom=0.38)
    figure_a.savefig(OUTPUT_A_PNG, dpi=450, bbox_inches="tight", pad_inches=0.05)
    figure_a.savefig(OUTPUT_A_PDF, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure_a)

    figure_b, axis_b = plt.subplots(figsize=(21.0, 7.2))
    image_b = draw_panel_b(axis_b, row_order, effect_matrix, q_matrix, cmap)
    cbar_b = figure_b.colorbar(image_b, ax=axis_b, fraction=0.018, pad=0.010)
    cbar_b.set_label("Rank-biserial effect, FN - TP", fontsize=22, labelpad=10)
    cbar_b.ax.tick_params(labelsize=20)
    figure_b.subplots_adjust(left=0.080, right=0.925, top=0.98, bottom=0.25)
    figure_b.savefig(OUTPUT_B_PNG, dpi=450, bbox_inches="tight", pad_inches=0.05)
    figure_b.savefig(OUTPUT_B_PDF, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure_b)

    print("Wrote {}".format(OUTPUT_A_PNG))
    print("Wrote {}".format(OUTPUT_A_PDF))
    print("Wrote {}".format(OUTPUT_B_PNG))
    print("Wrote {}".format(OUTPUT_B_PDF))
    print("Wrote {}".format(HEATMAP_VALUES_CSV))


if __name__ == "__main__":
    main()
