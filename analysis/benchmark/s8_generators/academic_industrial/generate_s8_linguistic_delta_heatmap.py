#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parent
RATES_CSV = ROOT / "s8_generator_linguistic_feature_rates.csv"
STATS_CSV = ROOT / "s8_generator_linguistic_feature_stats.csv"
OUTPUT_CSV = ROOT / "s8_generator_linguistic_feature_delta_heatmap_values.csv"
OUTPUT_PNG = ROOT / "fig_s8_generator_linguistic_feature_delta_heatmap.png"
OUTPUT_PDF = ROOT / "fig_s8_generator_linguistic_feature_delta_heatmap.pdf"

GROUP_FN = "A-TP / I-FN"
GROUP_TP = "A-TP / I-TP"

# Keep five generators with stable A-TP/I-TP sample sizes. S8-deepseek has only
# six A-TP/I-TP samples in the current detector-disagreement slice.
GENERATORS = [
    "S8-ministral",
    "S8-llama",
    "S8-gemini",
    "S8-gpt",
    "S8-claude",
]

GENERATOR_LABELS = {
    "S8-ministral": "Ministral",
    "S8-llama": "Llama",
    "S8-gemini": "Gemini",
    "S8-gpt": "GPT",
    "S8-claude": "Claude",
}

FEATURE_ORDER = [
    "Explicit action request",
    "Click/open request",
    "Direct URL/page instruction",
    "Information submission",
    "Login/account action",
    "Urgency wording",
    "Softened request",
    "Conversational wording",
]

FEATURE_LABELS = {
    "Explicit action request": "Explicit\naction",
    "Click/open request": "Click/\nopen",
    "Direct URL/page instruction": "URL/page\ninstruction",
    "Information submission": "Info\nsubmission",
    "Login/account action": "Login/\naccount",
    "Urgency wording": "Urgency",
    "Softened request": "Softened\nrequest",
    "Conversational wording": "Conversational\nwording",
}


def significance_stars(q_value: float) -> str:
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return ""


def read_csv_rows(path):
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def build_delta_table():
    rates = read_csv_rows(RATES_CSV)
    stats = read_csv_rows(STATS_CSV)

    rate_lookup = {}
    for row in rates:
        generator = row["generator"]
        feature = row["feature"]
        group = row["group"]
        if generator in GENERATORS and feature in FEATURE_ORDER and group in [GROUP_FN, GROUP_TP]:
            rate_lookup[(generator, feature, group)] = to_float(row["rate"])

    stat_lookup = {}
    for row in stats:
        generator = row["generator"]
        feature = row["feature"]
        if generator in GENERATORS and feature in FEATURE_ORDER:
            stat_lookup[(generator, feature)] = row

    table = []
    for generator in GENERATORS:
        for feature in FEATURE_ORDER:
            fn_rate = rate_lookup.get((generator, feature, GROUP_FN), np.nan)
            tp_rate = rate_lookup.get((generator, feature, GROUP_TP), np.nan)
            delta = fn_rate - tp_rate
            stat = stat_lookup.get((generator, feature), {})
            q_value = to_float(stat.get("q_value"))
            table.append(
                {
                    "generator": generator,
                    "feature": feature,
                    "A-TP / I-FN": fn_rate,
                    "A-TP / I-TP": tp_rate,
                    "delta_a_tp_i_fn_minus_a_tp_i_tp": delta,
                    "p_value": to_float(stat.get("p_value")),
                    "q_value": q_value,
                    "stars": significance_stars(q_value),
                    "generator_label": GENERATOR_LABELS[generator],
                    "feature_label": FEATURE_LABELS[feature],
                }
            )

    fieldnames = [
        "generator",
        "feature",
        "A-TP / I-FN",
        "A-TP / I-TP",
        "delta_a_tp_i_fn_minus_a_tp_i_tp",
        "p_value",
        "q_value",
        "stars",
        "generator_label",
        "feature_label",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)
    return table


def draw_heatmap(table) -> None:
    value_lookup = {
        (row["generator"], row["feature"]): row["delta_a_tp_i_fn_minus_a_tp_i_tp"]
        for row in table
    }
    q_lookup = {
        (row["generator"], row["feature"]): row["q_value"]
        for row in table
    }
    star_lookup = {
        (row["generator"], row["feature"]): row["stars"]
        for row in table
    }
    matrix = np.array(
        [
            [value_lookup[(generator, feature)] for feature in FEATURE_ORDER]
            for generator in GENERATORS
        ],
        dtype=float,
    )
    q_values = np.array(
        [
            [q_lookup[(generator, feature)] for feature in FEATURE_ORDER]
            for generator in GENERATORS
        ],
        dtype=float,
    )
    stars = [
        [star_lookup[(generator, feature)] for feature in FEATURE_ORDER]
        for generator in GENERATORS
    ]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
        }
    )

    values = matrix
    vmax = max(0.42, float(np.nanmax(np.abs(values))) * 1.05)
    cmap = LinearSegmentedColormap.from_list(
        "s8_delta_blue_pale_yellow",
        [
            (0.0, "#f6d681"),
            (0.48, "#fff7df"),
            (0.50, "#ffffff"),
            (0.52, "#eaf2f1"),
            (1.0, "#003b4d"),
        ],
    )
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    figure, axis = plt.subplots(figsize=(18.2, 6.5))
    image = axis.imshow(values, cmap=cmap, norm=norm, aspect="auto")

    axis.set_xticks(np.arange(len(FEATURE_ORDER)))
    axis.set_xticklabels(
        [FEATURE_LABELS[feature] for feature in FEATURE_ORDER],
        fontsize=22,
        rotation=10,
        ha="center",
    )
    axis.set_yticks(np.arange(len(GENERATORS)))
    axis.set_yticklabels(
        [GENERATOR_LABELS[generator] for generator in GENERATORS],
        fontsize=22,
    )
    axis.tick_params(axis="x", pad=8)
    axis.tick_params(axis="y", pad=6)

    axis.set_xticks(np.arange(-0.5, len(FEATURE_ORDER), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(GENERATORS), 1), minor=True)
    axis.grid(which="minor", color="white", linestyle="-", linewidth=2.0)
    axis.tick_params(which="minor", bottom=False, left=False)

    for row_index, generator in enumerate(GENERATORS):
        for col_index, feature in enumerate(FEATURE_ORDER):
            value = float(matrix[row_index, col_index])
            star = stars[row_index][col_index]
            q_value = float(q_values[row_index, col_index])
            if q_value < 0.001:
                q_label = "q<0.001"
            elif q_value < 0.05:
                q_label = f"q={q_value:.3f}"
            else:
                q_label = ""
            if q_label:
                label = f"{value * 100:+.1f}%{star}\n{q_label}"
            else:
                label = f"{value * 100:+.1f}%"
            axis.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=18.5,
                color="black",
                linespacing=0.92,
            )

    for spine in axis.spines.values():
        spine.set_linewidth(1.0)

    colorbar = figure.colorbar(image, ax=axis, fraction=0.028, pad=0.018)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    colorbar.set_label(r"$\Delta$ feature rate", fontsize=21, labelpad=13)
    colorbar.ax.tick_params(labelsize=19)

    figure.subplots_adjust(left=0.095, right=0.93, bottom=0.18, top=0.96)
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.05)
    figure.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)


def main() -> None:
    table = build_delta_table()
    draw_heatmap(table)


if __name__ == "__main__":
    main()
