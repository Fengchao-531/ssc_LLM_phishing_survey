#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "7"
INPUT_SCORED = SCRIPT_DIR / "8" / "scored_inputs" / "llm_malicious_turns_scored.csv"
OUTPUT_CSV = OUTPUT_DIR / "llm_multiturn_pair_turn_distribution.csv"
OUTPUT_PNG = OUTPUT_DIR / "llm_multiturn_pair_turn_boxplots.png"
OUTPUT_TREND_CSV = OUTPUT_DIR / "llm_multiturn_pair_turn_mean_trends.csv"
OUTPUT_TREND_PNG = OUTPUT_DIR / "llm_multiturn_pair_turn_heatmap.png"
OUTPUT_PRINCIPLE_TREND_CSV = OUTPUT_DIR / "llm_multiturn_principle_turn_mean_trends.csv"
OUTPUT_PRINCIPLE_TREND_PNG = OUTPUT_DIR / "llm_multiturn_principle_turn_heatmap.png"
OUTPUT_METADATA = OUTPUT_DIR / "llm_multiturn_pair_turn_metadata.json"
HEATMAP_VALUE_FONT_SIZE = 17

PRINCIPLE_SPECS = [
    ("Authority", "principle_authority"),
    ("Liking", "principle_liking"),
    ("Reciprocity", "principle_reciprocity"),
    ("Social Proof", "principle_social_proof"),
    ("Scarcity", "principle_scarcity"),
    ("Commitment", "principle_commitment"),
]

SELECTED_PAIR_GROUPS = [
    ("Authority", "Authority", "A-A"),
    ("Authority", "Liking", "A-L"),
    ("Authority", "Reciprocity", "A-R"),
    ("Authority", "Social Proof", "A-SP"),
    ("Liking", "Liking", "L-L"),
    ("Liking", "Reciprocity", "L-R"),
    ("Liking", "Social Proof", "L-SP"),
    ("Reciprocity", "Reciprocity", "R-R"),
    ("Reciprocity", "Social Proof", "R-SP"),
    ("Social Proof", "Social Proof", "SP-SP"),
]

TURN_COLOR = "#2f5d50"
PLOT_ROUND_LIMIT = 10
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "folder7_pair_turn_green",
    ["#f5faf7", "#d6e9df", "#9fc5b2", "#5d8f79", "#2f5d50"],
    N=256,
)


def load_scored_turns() -> pd.DataFrame:
    if not INPUT_SCORED.exists():
        raise FileNotFoundError(f"Scored turn input not found: {INPUT_SCORED}")
    frame = pd.read_csv(INPUT_SCORED, low_memory=False).copy()
    frame["round_number"] = pd.to_numeric(frame["round_number"], errors="coerce")
    frame = frame[frame["round_number"].notna()].copy()
    frame["round_number"] = frame["round_number"].astype(int)
    frame = frame[frame["round_number"] <= PLOT_ROUND_LIMIT].copy()
    if frame.empty:
        raise ValueError("No LLM multi-turn rows were available after round filtering.")
    return frame.reset_index(drop=True)


def build_distribution_table(scored: pd.DataFrame) -> pd.DataFrame:
    principle_lookup = {name: column_name for name, column_name in PRINCIPLE_SPECS}
    rows: list[dict[str, object]] = []
    for _, row in scored.iterrows():
        for left_name, right_name, short_label in SELECTED_PAIR_GROUPS:
            left_value = pd.to_numeric(row[principle_lookup[left_name]], errors="coerce")
            right_value = pd.to_numeric(row[principle_lookup[right_name]], errors="coerce")
            if pd.isna(left_value) or pd.isna(right_value):
                continue
            pair_value = float(left_value if left_name == right_name else left_value * right_value)
            rows.append(
                {
                    "round_number": int(row["round_number"]),
                    "pair_group": short_label,
                    "left_principle": left_name,
                    "right_principle": right_name,
                    "value": pair_value,
                    "dialogue_id": row["dialogue_id"],
                    "round_label": row.get("round_label", f"R{int(row['round_number'])}"),
                }
            )
    return pd.DataFrame(rows)


def draw_boxplot(distribution_table: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(45, 15.5), sharey=True)
    figure.patch.set_facecolor("#ffffff")
    axes = axes.flatten()
    group_centers = np.arange(1, PLOT_ROUND_LIMIT + 1, dtype=float)

    for axis, (_left_name, _right_name, short_label) in zip(axes, SELECTED_PAIR_GROUPS, strict=True):
        pair_table = distribution_table[distribution_table["pair_group"] == short_label].copy()
        box_data: list[np.ndarray] = []
        for round_number in range(1, PLOT_ROUND_LIMIT + 1):
            values = pair_table[pair_table["round_number"] == round_number]["value"].to_numpy(dtype=float)
            box_data.append(values)

        boxplot = axis.boxplot(
            box_data,
            positions=group_centers,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#ffffff", "linewidth": 2.2},
            whiskerprops={"color": TURN_COLOR, "linewidth": 1.5},
            capprops={"color": TURN_COLOR, "linewidth": 1.5},
            boxprops={"edgecolor": TURN_COLOR, "linewidth": 1.8},
        )
        for patch in boxplot["boxes"]:
            patch.set_facecolor(TURN_COLOR)
            patch.set_alpha(0.92)

        axis.set_title(short_label, fontsize=29, pad=12, color="#111111")
        axis.grid(axis="y", color="#dddddd", linewidth=0.85, alpha=0.9)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#555555")
        axis.spines["bottom"].set_color("#555555")
        axis.tick_params(axis="x", labelsize=22, colors="#222222")
        axis.tick_params(axis="y", labelsize=22, colors="#222222")
        axis.set_xticks(group_centers)
        axis.set_xticklabels([str(round_number) for round_number in range(1, PLOT_ROUND_LIMIT + 1)], fontsize=22)
        axis.set_ylim(0.0, 1.02)

    for axis in axes[5:]:
        axis.set_xlabel("Malicious turn round", fontsize=25, color="#111111", labelpad=12)
    for axis in [axes[0], axes[5]]:
        axis.set_ylabel("Pair co-occurrence value", fontsize=25, color="#111111")

    figure.subplots_adjust(left=0.07, right=0.995, bottom=0.13, top=0.93, wspace=0.22, hspace=0.30)
    figure.savefig(OUTPUT_PNG, dpi=260, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def build_turn_summary(distribution_table: pd.DataFrame) -> pd.DataFrame:
    summary = (
        distribution_table.groupby(["pair_group", "round_number"], as_index=False)
        .agg(
            mean_value=("value", "mean"),
            median_value=("value", "median"),
            sample_count=("value", "count"),
        )
        .reset_index(drop=True)
    )
    return summary


def build_principle_turn_summary(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for principle_name, principle_column in PRINCIPLE_SPECS:
        summary = (
            scored.groupby("round_number", as_index=False)
            .agg(
                mean_value=(principle_column, "mean"),
                median_value=(principle_column, "median"),
                sample_count=(principle_column, "count"),
            )
            .reset_index(drop=True)
        )
        summary["principle"] = principle_name
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def draw_turn_heatmap(summary_table: pd.DataFrame) -> None:
    pair_order = [short_label for _, _, short_label in SELECTED_PAIR_GROUPS]
    pivot = (
        summary_table.pivot(index="pair_group", columns="round_number", values="mean_value")
        .reindex(pair_order)
        .reindex(columns=range(1, PLOT_ROUND_LIMIT + 1))
    )
    matrix = pivot.to_numpy(dtype=float)
    vmax = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0

    figure, axis = plt.subplots(figsize=(16.5, 7.4))
    figure.patch.set_facecolor("#ffffff")
    image = axis.imshow(matrix, cmap=HEATMAP_CMAP, vmin=0.0, vmax=vmax, aspect="auto")

    axis.set_xticks(range(PLOT_ROUND_LIMIT))
    axis.set_xticklabels([str(round_number) for round_number in range(1, PLOT_ROUND_LIMIT + 1)], fontsize=18)
    axis.set_yticks(range(len(pair_order)))
    axis.set_yticklabels(pair_order, fontsize=18)
    axis.set_xlabel("Malicious turn round", fontsize=20, color="#111111", labelpad=10)
    axis.set_ylabel("Pair persuasion group", fontsize=20, color="#111111", labelpad=10)
    axis.tick_params(axis="x", colors="#222222", length=0)
    axis.tick_params(axis="y", colors="#222222", length=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if not np.isfinite(value):
                continue
            text_color = "#ffffff" if value >= max(vmax * 0.62, 0.45) else "#111111"
            axis.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=HEATMAP_VALUE_FONT_SIZE,
                color=text_color,
            )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label("Mean pair co-occurrence", fontsize=17)
    colorbar.ax.tick_params(labelsize=15)

    figure.subplots_adjust(left=0.12, right=0.92, bottom=0.16, top=0.97)
    figure.savefig(OUTPUT_TREND_PNG, dpi=260, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def draw_principle_turn_heatmap(summary_table: pd.DataFrame) -> None:
    principle_order = [name for name, _ in PRINCIPLE_SPECS]
    pivot = (
        summary_table.pivot(index="principle", columns="round_number", values="mean_value")
        .reindex(principle_order)
        .reindex(columns=range(1, PLOT_ROUND_LIMIT + 1))
    )
    matrix = pivot.to_numpy(dtype=float)
    vmax = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0

    figure, axis = plt.subplots(figsize=(16.5, 5.8))
    figure.patch.set_facecolor("#ffffff")
    image = axis.imshow(matrix, cmap=HEATMAP_CMAP, vmin=0.0, vmax=vmax, aspect="auto")

    axis.set_xticks(range(PLOT_ROUND_LIMIT))
    axis.set_xticklabels([str(round_number) for round_number in range(1, PLOT_ROUND_LIMIT + 1)], fontsize=18)
    axis.set_yticks(range(len(principle_order)))
    axis.set_yticklabels(principle_order, fontsize=18)
    axis.set_xlabel("Malicious turn round", fontsize=20, color="#111111", labelpad=10)
    axis.set_ylabel("Persuasion principle", fontsize=20, color="#111111", labelpad=10)
    axis.tick_params(axis="x", colors="#222222", length=0)
    axis.tick_params(axis="y", colors="#222222", length=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if not np.isfinite(value):
                continue
            text_color = "#ffffff" if value >= max(vmax * 0.62, 0.45) else "#111111"
            axis.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=HEATMAP_VALUE_FONT_SIZE,
                color=text_color,
            )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label("Mean principle strength", fontsize=17)
    colorbar.ax.tick_params(labelsize=15)

    figure.subplots_adjust(left=0.16, right=0.92, bottom=0.18, top=0.97)
    figure.savefig(OUTPUT_PRINCIPLE_TREND_PNG, dpi=260, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def write_metadata(scored: pd.DataFrame, distribution_table: pd.DataFrame) -> None:
    round_counts = (
        scored.groupby("round_number")["dialogue_id"]
        .nunique()
        .sort_index()
        .to_dict()
    )
    metadata = {
        "input_scored_csv": str(INPUT_SCORED),
        "output_png": OUTPUT_PNG.name,
        "output_csv": OUTPUT_CSV.name,
        "output_trend_png": OUTPUT_TREND_PNG.name,
        "output_trend_csv": OUTPUT_TREND_CSV.name,
        "output_principle_trend_png": OUTPUT_PRINCIPLE_TREND_PNG.name,
        "output_principle_trend_csv": OUTPUT_PRINCIPLE_TREND_CSV.name,
        "plot_round_limit": PLOT_ROUND_LIMIT,
        "source_dataset": "LLM multi-turn phishing",
        "dialogue_count": int(scored["dialogue_id"].nunique()),
        "turn_row_count": int(len(scored)),
        "pair_distribution_rows": int(len(distribution_table)),
        "selected_pair_groups": [short_label for _, _, short_label in SELECTED_PAIR_GROUPS],
        "round_counts": {str(key): int(value) for key, value in round_counts.items()},
    }
    OUTPUT_METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scored = load_scored_turns()
    distribution_table = build_distribution_table(scored)
    distribution_table.to_csv(OUTPUT_CSV, index=False)
    summary_table = build_turn_summary(distribution_table)
    summary_table.to_csv(OUTPUT_TREND_CSV, index=False)
    principle_summary_table = build_principle_turn_summary(scored)
    principle_summary_table.to_csv(OUTPUT_PRINCIPLE_TREND_CSV, index=False)
    draw_boxplot(distribution_table)
    draw_turn_heatmap(summary_table)
    draw_principle_turn_heatmap(principle_summary_table)
    write_metadata(scored, distribution_table)
    print(f"Wrote {OUTPUT_PNG}")
    print(f"Wrote {OUTPUT_TREND_PNG}")
    print(f"Wrote {OUTPUT_PRINCIPLE_TREND_PNG}")


if __name__ == "__main__":
    main()
