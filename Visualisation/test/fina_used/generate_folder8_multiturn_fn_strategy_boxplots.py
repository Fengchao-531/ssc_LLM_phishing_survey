#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "8"

HW_SCORED_CSV = OUTPUT_DIR / "scored_inputs" / "hw_malicious_turns_scored.csv"
LLM_SCORED_CSV = OUTPUT_DIR / "scored_inputs" / "llm_malicious_turns_scored.csv"

# Current usable FN slices come from the first completed llm_guard chunks:
# - LLM: first 500 utterance rows from the current LLM-first round benchmark
# - HW: first 500 utterance rows from the earlier HW-first round benchmark
LLM_CHUNK_INPUT_CSV = (
    SCRIPT_DIR.parents[2]
    / "Detectors"
    / "Industry"
    / "email_detectors"
    / "output"
    / "logs"
    / "tmp"
    / "multiturn-vishing-r6_detector_suite_kl69fmf5"
    / "llm_guard_000001_000500"
    / "chunk_input.csv"
)
LLM_CHUNK_SUMMARY_CSV = (
    SCRIPT_DIR.parents[2]
    / "Detectors"
    / "Industry"
    / "email_detectors"
    / "output"
    / "logs"
    / "tmp"
    / "multiturn-vishing-r6_detector_suite_kl69fmf5"
    / "llm_guard_000001_000500"
    / "summary"
    / "llm_guard_summary.csv"
)
HW_CHUNK_INPUT_CSV = (
    SCRIPT_DIR.parents[2]
    / "Detectors"
    / "Industry"
    / "email_detectors"
    / "output"
    / "logs"
    / "tmp"
    / "multiturn-vishing-r6_detector_suite_ly9uqy3v"
    / "llm_guard_000001_000500"
    / "chunk_input.csv"
)
HW_CHUNK_SUMMARY_CSV = (
    SCRIPT_DIR.parents[2]
    / "Detectors"
    / "Industry"
    / "email_detectors"
    / "output"
    / "logs"
    / "tmp"
    / "multiturn-vishing-r6_detector_suite_ly9uqy3v"
    / "llm_guard_000001_000500"
    / "summary"
    / "llm_guard_summary.csv"
)

OUTPUT_CSV = OUTPUT_DIR / "multiturn_round_strategy_distribution_fn.csv"
OUTPUT_PNG = OUTPUT_DIR / "multiturn_round_strategy_boxplots_fn.png"
PAIR_OUTPUT_CSV = OUTPUT_DIR / "multiturn_round_pair_group_distribution_fn.csv"
PAIR_OUTPUT_PNG = OUTPUT_DIR / "multiturn_round_pair_group_boxplots_fn.png"
OUTPUT_METADATA = OUTPUT_DIR / "multiturn_round_strategy_fn_metadata.json"
X_AXIS_LABEL_FONT_SIZE = 33
TICK_LABEL_FONT_SIZE = 29

TURN_COLORS = {
    "HW": "#143d73",
    "LLM": "#2f5d50",
}
PRINCIPLE_SPECS = [
    ("Authority", "principle_authority", "#143d73"),
    ("Reciprocity", "principle_reciprocity", "#35628d"),
    ("Commitment", "principle_commitment", "#4f8696"),
    ("Scarcity", "principle_scarcity", "#77a97c"),
    ("Social Proof", "principle_social_proof", "#b0b96b"),
    ("Liking", "principle_liking", "#d59f63"),
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
PLOT_ROUND_LIMIT = 6


def load_fn_keys(dataset_name: str, chunk_input_csv: Path, chunk_summary_csv: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    chunk_frame = pd.read_csv(chunk_input_csv, low_memory=False).copy()
    summary_frame = pd.read_csv(chunk_summary_csv, low_memory=False).copy()

    chunk_frame["row_number"] = np.arange(1, len(chunk_frame) + 1, dtype=int)
    merged = chunk_frame.merge(
        summary_frame[["row_number", "overall_is_valid", "flagged_scanners"]],
        on="row_number",
        how="left",
    )
    merged["label"] = pd.to_numeric(merged["label"], errors="coerce")
    merged["prediction"] = merged["overall_is_valid"].map({True: 0, False: 1})

    fn_rows = merged[(merged["label"] == 1) & (merged["prediction"] == 0)].copy()
    fn_rows["round_number"] = pd.to_numeric(fn_rows["round_number"], errors="coerce").astype(int)
    fn_rows = fn_rows[fn_rows["round_number"] <= PLOT_ROUND_LIMIT].copy()

    key_table = fn_rows[["dialogue_id", "round_number"]].drop_duplicates().copy()
    key_table["dataset"] = dataset_name

    summary = {
        "dataset": dataset_name,
        "chunk_input_csv": str(chunk_input_csv),
        "chunk_summary_csv": str(chunk_summary_csv),
        "evaluated_rows": int(len(merged)),
        "evaluated_positive_rows": int((merged["label"] == 1).sum()),
        "fn_rows": int(len(fn_rows)),
        "fn_round_counts": {
            str(round_number): int(count)
            for round_number, count in fn_rows.groupby("round_number").size().items()
        },
    }
    return key_table, summary


def build_distribution_table() -> tuple[pd.DataFrame, dict[str, object]]:
    fn_key_tables: dict[str, pd.DataFrame] = {}
    evaluation_summaries: list[dict[str, object]] = []
    for dataset_name, chunk_input_csv, chunk_summary_csv in [
        ("HW", HW_CHUNK_INPUT_CSV, HW_CHUNK_SUMMARY_CSV),
        ("LLM", LLM_CHUNK_INPUT_CSV, LLM_CHUNK_SUMMARY_CSV),
    ]:
        key_table, summary = load_fn_keys(dataset_name, chunk_input_csv, chunk_summary_csv)
        fn_key_tables[dataset_name] = key_table
        evaluation_summaries.append(summary)

    rows: list[dict[str, object]] = []
    for dataset_name, scored_csv in [("HW", HW_SCORED_CSV), ("LLM", LLM_SCORED_CSV)]:
        scored = pd.read_csv(scored_csv, low_memory=False).copy()
        scored["round_number"] = pd.to_numeric(scored["round_number"], errors="coerce").astype(int)
        scored = scored[scored["round_number"] <= PLOT_ROUND_LIMIT].copy()
        key_table = fn_key_tables[dataset_name]
        filtered = scored.merge(key_table, on=["dialogue_id", "round_number"], how="inner")

        for _, row in filtered.iterrows():
            for principle_name, column_name, _ in PRINCIPLE_SPECS:
                value = pd.to_numeric(row[column_name], errors="coerce")
                if pd.isna(value):
                    continue
                rows.append(
                    {
                        "dataset": dataset_name,
                        "round_number": int(row["round_number"]),
                        "principle": principle_name,
                        "value": float(value),
                        "dialogue_id": row["dialogue_id"],
                    }
                )

    distribution = pd.DataFrame(rows)
    metadata = {
        "plot_round_limit": PLOT_ROUND_LIMIT,
        "detector": "llm_guard",
        "subset_definition": "False negatives among malicious utterances only (label=1 and prediction=0).",
        "subset_note": "Current FN plot is based on the first completed 500 evaluated utterance rows per dataset from the round-level benchmark.",
        "evaluation_summaries": evaluation_summaries,
        "scored_inputs": {
            "HW": str(HW_SCORED_CSV),
            "LLM": str(LLM_SCORED_CSV),
        },
        "distribution_rows": int(len(distribution)),
    }
    return distribution, metadata


def build_pair_distribution_table(distribution_table: pd.DataFrame) -> pd.DataFrame:
    principle_lookup = {name: principle_table for name, principle_table in distribution_table.groupby("principle")}
    rows: list[dict[str, object]] = []

    for dataset_name in ["HW", "LLM"]:
        for round_number in range(1, PLOT_ROUND_LIMIT + 1):
            merged: pd.DataFrame | None = None
            for principle_name, principle_frame in principle_lookup.items():
                subset = principle_frame[
                    (principle_frame["dataset"] == dataset_name)
                    & (principle_frame["round_number"] == round_number)
                ][["dialogue_id", "value"]].copy()
                subset = subset.rename(columns={"value": principle_name})
                if merged is None:
                    merged = subset
                else:
                    merged = merged.merge(subset, on="dialogue_id", how="inner")

            if merged is None or merged.empty:
                continue

            for left_name, right_name, short_label in SELECTED_PAIR_GROUPS:
                left_values = pd.to_numeric(merged[left_name], errors="coerce")
                right_values = pd.to_numeric(merged[right_name], errors="coerce")
                pair_values = left_values if left_name == right_name else left_values * right_values
                for dialogue_id, pair_value in zip(merged["dialogue_id"], pair_values, strict=True):
                    if pd.isna(pair_value):
                        continue
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "round_number": round_number,
                            "pair_group": short_label,
                            "left_principle": left_name,
                            "right_principle": right_name,
                            "value": float(pair_value),
                            "dialogue_id": dialogue_id,
                        }
                    )

    return pd.DataFrame(rows)


def draw_boxplot(distribution_table: pd.DataFrame, output_png: Path) -> None:
    font_scale = 1.5
    figure, axes = plt.subplots(2, 3, figsize=(34, 19.5), sharey=True)
    figure.patch.set_facecolor("#ffffff")
    axes = axes.flatten()

    group_centers = np.arange(1, PLOT_ROUND_LIMIT + 1, dtype=float)
    box_width = 0.32
    offset = 0.19
    positions = {
        "HW": group_centers - offset,
        "LLM": group_centers + offset,
    }

    for axis, (principle_name, _, _) in zip(axes, PRINCIPLE_SPECS, strict=True):
        principle_table = distribution_table[distribution_table["principle"] == principle_name].copy()
        axis.set_facecolor("#ffffff")

        for dataset_name in ["HW", "LLM"]:
            dataset_table = principle_table[principle_table["dataset"] == dataset_name].copy()
            box_data: list[np.ndarray] = []
            for round_number in range(1, PLOT_ROUND_LIMIT + 1):
                values = dataset_table[dataset_table["round_number"] == round_number]["value"].to_numpy(dtype=float)
                box_data.append(values)

            boxplot = axis.boxplot(
                box_data,
                positions=positions[dataset_name],
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#ffffff", "linewidth": 2.3},
                whiskerprops={"color": TURN_COLORS[dataset_name], "linewidth": 1.6},
                capprops={"color": TURN_COLORS[dataset_name], "linewidth": 1.6},
                boxprops={"edgecolor": TURN_COLORS[dataset_name], "linewidth": 1.8},
            )
            for patch in boxplot["boxes"]:
                patch.set_facecolor(TURN_COLORS[dataset_name])
                patch.set_alpha(0.92)

        axis.set_title(principle_name, fontsize=24 * font_scale, pad=12, color="#111111")
        axis.grid(axis="y", color="#dddddd", linewidth=0.85, alpha=0.9)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#555555")
        axis.spines["bottom"].set_color("#555555")
        axis.tick_params(axis="x", labelsize=TICK_LABEL_FONT_SIZE, colors="#222222")
        axis.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE, colors="#222222")
        axis.set_xticks(group_centers)
        axis.set_xticklabels(
            [str(round_number) for round_number in range(1, PLOT_ROUND_LIMIT + 1)],
            fontsize=TICK_LABEL_FONT_SIZE,
        )
        axis.set_ylim(0.0, 1.02)

    for axis in axes[3:]:
        axis.set_xlabel("Malicious turn round", fontsize=X_AXIS_LABEL_FONT_SIZE, color="#111111", labelpad=12)
    for axis in [axes[0], axes[3]]:
        axis.set_ylabel("WVAE strategy score within FN turns", fontsize=23 * font_scale, color="#111111")

    commitment_axis = axes[2]
    commitment_axis.legend(
        handles=[
            Patch(facecolor=TURN_COLORS["HW"], edgecolor=TURN_COLORS["HW"], label="HW multi-turn FN"),
            Patch(facecolor=TURN_COLORS["LLM"], edgecolor=TURN_COLORS["LLM"], label="LLM multi-turn FN"),
        ],
        loc="upper right",
        frameon=True,
        fontsize=20 * font_scale,
        facecolor="#ffffff",
        edgecolor="#d0d0d0",
        borderaxespad=0.35,
    )

    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.16, top=0.93, wspace=0.26, hspace=0.28)
    figure.savefig(output_png, dpi=260, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def draw_pair_group_boxplot(distribution_table: pd.DataFrame, output_png: Path) -> None:
    font_scale = 1.5
    figure, axes = plt.subplots(2, 5, figsize=(52, 18.5), sharey=True)
    figure.patch.set_facecolor("#ffffff")
    axes = axes.flatten()

    group_centers = np.arange(1, PLOT_ROUND_LIMIT + 1, dtype=float)
    box_width = 0.32
    offset = 0.19
    positions = {
        "HW": group_centers - offset,
        "LLM": group_centers + offset,
    }

    for axis, (_left_name, _right_name, short_label) in zip(axes, SELECTED_PAIR_GROUPS, strict=True):
        pair_table = distribution_table[distribution_table["pair_group"] == short_label].copy()
        axis.set_facecolor("#ffffff")

        for dataset_name in ["HW", "LLM"]:
            dataset_table = pair_table[pair_table["dataset"] == dataset_name].copy()
            box_data: list[np.ndarray] = []
            for round_number in range(1, PLOT_ROUND_LIMIT + 1):
                values = dataset_table[dataset_table["round_number"] == round_number]["value"].to_numpy(dtype=float)
                box_data.append(values)

            boxplot = axis.boxplot(
                box_data,
                positions=positions[dataset_name],
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#ffffff", "linewidth": 2.3},
                whiskerprops={"color": TURN_COLORS[dataset_name], "linewidth": 1.6},
                capprops={"color": TURN_COLORS[dataset_name], "linewidth": 1.6},
                boxprops={"edgecolor": TURN_COLORS[dataset_name], "linewidth": 1.8},
            )
            for patch in boxplot["boxes"]:
                patch.set_facecolor(TURN_COLORS[dataset_name])
                patch.set_alpha(0.92)

        axis.set_title(short_label, fontsize=24 * font_scale, pad=12, color="#111111")
        axis.grid(axis="y", color="#dddddd", linewidth=0.85, alpha=0.9)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#555555")
        axis.spines["bottom"].set_color("#555555")
        axis.tick_params(axis="x", labelsize=TICK_LABEL_FONT_SIZE, colors="#222222")
        axis.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE, colors="#222222")
        axis.set_xticks(group_centers)
        axis.set_xticklabels(
            [str(round_number) for round_number in range(1, PLOT_ROUND_LIMIT + 1)],
            fontsize=TICK_LABEL_FONT_SIZE,
        )
        axis.set_ylim(0.0, 1.02)

    for axis in axes[5:]:
        axis.set_xlabel("Malicious turn round", fontsize=X_AXIS_LABEL_FONT_SIZE, color="#111111", labelpad=12)
    for axis in [axes[0], axes[5]]:
        axis.set_ylabel("Pair co-occurrence value within FN turns", fontsize=23 * font_scale, color="#111111")

    figure.subplots_adjust(left=0.07, right=0.995, bottom=0.17, top=0.92, wspace=0.24, hspace=0.28)
    figure.savefig(output_png, dpi=260, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    distribution_table, metadata = build_distribution_table()
    if distribution_table.empty:
        raise RuntimeError("No FN malicious-turn persuasion rows were found for folder 8.")

    pair_distribution_table = build_pair_distribution_table(distribution_table)
    distribution_table.to_csv(OUTPUT_CSV, index=False)
    pair_distribution_table.to_csv(PAIR_OUTPUT_CSV, index=False)
    draw_boxplot(distribution_table, OUTPUT_PNG)
    draw_pair_group_boxplot(pair_distribution_table, PAIR_OUTPUT_PNG)
    metadata["pair_distribution_rows"] = int(len(pair_distribution_table))
    metadata["pair_output_csv"] = str(PAIR_OUTPUT_CSV)
    metadata["pair_output_png"] = str(PAIR_OUTPUT_PNG)
    metadata["selected_pair_groups"] = [short_label for _, _, short_label in SELECTED_PAIR_GROUPS]
    OUTPUT_METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
