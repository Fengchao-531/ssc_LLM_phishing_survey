#!/usr/bin/env python3

from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SAMPLE_CSV = ROOT / "s6_sample_level_paired_rewriting_scores.csv"
OUTPUT_PNG = ROOT / "fig_s6_sample_persuasion_shift_by_outcome.png"
OUTPUT_PDF = ROOT / "fig_s6_sample_persuasion_shift_by_outcome.pdf"
SHIFT_VALUES_CSV = ROOT / "s6_sample_persuasion_shift_by_outcome_values.csv"
STATS_CSV = ROOT / "s6_sample_persuasion_shift_by_outcome_stats.csv"

METHODS = ["Fuzzer", "UTA", "MPG"]
DETECTORS = ["SecureNet", "V3"]
DETECTOR_MARKERS = {"SecureNet": "o", "V3": "^"}
OUTCOME_COLORS = {"TP": "#8cc4df", "FN": "#003b4d"}
EDGE_COLOR = "#111111"


def rank_biserial_from_u(u_stat: float, n_a: int, n_b: int) -> float:
    if n_a == 0 or n_b == 0:
        return float("nan")
    return 2.0 * u_stat / (n_a * n_b) - 1.0


def significance_stars(q_value: float) -> str:
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return "ns"


def read_csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_shift_values():
    values = read_csv_rows(SHIFT_VALUES_CSV)
    for row in values:
        row["per_sample_persuasion_shift"] = float(row["per_sample_persuasion_shift"])
    return values


def build_stats(values):
    return read_csv_rows(STATS_CSV)


def select_values(values, method, detector, outcome):
    return np.asarray(
        [
            row["per_sample_persuasion_shift"]
            for row in values
            if row["method"] == method
            and row["detector_label"] == detector
            and row["outcome"] == outcome
        ],
        dtype=float,
    )


def get_stat_row(stats, method, detector):
    for row in stats:
        if row["method"] == method and row["detector_label"] == detector:
            return row
    raise RuntimeError("Missing stats row for {} {}".format(method, detector))


def draw_box(
    axis: plt.Axes,
    values: np.ndarray,
    position: float,
    color: str,
    marker: str,
    significant: bool,
    rng: np.random.Generator,
) -> None:
    axis.boxplot(
        values,
        positions=[position],
        widths=0.23,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": EDGE_COLOR, "linewidth": 1.4},
        boxprops={"facecolor": color, "edgecolor": EDGE_COLOR, "linewidth": 1.1},
        whiskerprops={"color": EDGE_COLOR, "linewidth": 1.0},
        capprops={"color": EDGE_COLOR, "linewidth": 1.0},
    )
    if len(values) > 450:
        plot_values = rng.choice(values, size=450, replace=False)
    else:
        plot_values = values
    jitter = rng.normal(0, 0.035, size=len(plot_values))
    axis.scatter(
        np.full(len(plot_values), position) + jitter,
        plot_values,
        s=13,
        marker=marker,
        facecolor=color if significant else "none",
        edgecolor=EDGE_COLOR,
        linewidth=0.55,
        alpha=0.38,
        zorder=2,
    )


def draw_figure(values, stats) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
        }
    )
    rng = np.random.default_rng(7)
    figure, axis = plt.subplots(figsize=(16.2, 9.8))

    base_positions = {"Fuzzer": 0.0, "UTA": 1.08, "MPG": 2.16}
    offsets = {
        ("SecureNet", "TP"): -0.36,
        ("SecureNet", "FN"): -0.12,
        ("V3", "TP"): 0.12,
        ("V3", "FN"): 0.36,
    }

    for method in METHODS:
        for detector in DETECTORS:
            for outcome in ["TP", "FN"]:
                stat_row = get_stat_row(stats, method, detector)
                significant = stat_row["significance"] != "ns"
                subset = select_values(values, method, detector, outcome)
                position = base_positions[method] + offsets[(detector, outcome)]
                draw_box(
                    axis,
                    subset,
                    position,
                    OUTCOME_COLORS[outcome],
                    DETECTOR_MARKERS[detector],
                    significant,
                    rng,
                )

    for method in METHODS:
        for detector, y_offset in [("SecureNet", 0.006), ("V3", 0.026)]:
            row = get_stat_row(stats, method, detector)
            x1 = base_positions[method] + offsets[(detector, "TP")]
            x2 = base_positions[method] + offsets[(detector, "FN")]
            y = 0.365 + y_offset
            axis.plot([x1, x1, x2, x2], [y, y + 0.004, y + 0.004, y], color=EDGE_COLOR, linewidth=1.0)
            axis.text(
                (x1 + x2) / 2,
                y + 0.005,
                "n.s." if row["significance"] == "ns" else row["significance"],
                ha="center",
                va="bottom",
                fontsize=31 if row["significance"] == "ns" else 36,
            )

    axis.set_xticks([base_positions[method] for method in METHODS])
    axis.set_xticklabels(METHODS, fontsize=42)
    axis.set_ylabel("Per-sample\npersuasion shift", fontsize=42, labelpad=20)
    axis.set_xlabel("Rewriting method", fontsize=42, labelpad=14)
    axis.tick_params(axis="y", labelsize=40)
    axis.set_xlim(-0.62, 2.78)
    axis.set_ylim(0.0, 0.47)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor=OUTCOME_COLORS["TP"],
            markeredgecolor=EDGE_COLOR,
            markersize=17,
            label="TP (detected)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor=OUTCOME_COLORS["FN"],
            markeredgecolor=EDGE_COLOR,
            markersize=17,
            label="FN (missed)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker=DETECTOR_MARKERS["SecureNet"],
            linestyle="none",
            markerfacecolor="#ffffff",
            markeredgecolor=EDGE_COLOR,
            markersize=17,
            label="SecureNet",
        ),
        plt.Line2D(
            [0],
            [0],
            marker=DETECTOR_MARKERS["V3"],
            linestyle="none",
            markerfacecolor="#ffffff",
            markeredgecolor=EDGE_COLOR,
            markersize=17,
            label="V3",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#666666",
            markeredgecolor=EDGE_COLOR,
            markersize=17,
            label="Significant",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=EDGE_COLOR,
            markersize=17,
            label="Not significant",
        ),
    ]
    axis.legend(
        handles=legend_handles,
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.30),
        fontsize=31,
        columnspacing=0.70,
        labelspacing=0.24,
        handletextpad=0.25,
        borderaxespad=0.0,
    )

    figure.subplots_adjust(left=0.17, right=0.99, bottom=0.18, top=0.76)
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.06)
    figure.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)


def main() -> None:
    values = load_shift_values()
    stats = build_stats(values)
    draw_figure(values, stats)


if __name__ == "__main__":
    main()
