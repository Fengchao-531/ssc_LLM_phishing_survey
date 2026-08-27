#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
OUTPUT_DIR = SCRIPT_DIR / "8"

HW_INPUT_CSV = (
    REPO_ROOT
    / "Datasets"
    / "sublist"
    / "S7-Cross-channel Expansion"
    / "HW-Vishing-Multi-ScamBaiter.csv"
)
LLM_INPUT_CSV = (
    REPO_ROOT
    / "Datasets"
    / "sublist"
    / "S7-Cross-channel Expansion"
    / "LLM-Vishing-Multi-BothBosu.csv"
)

WVAE_CODE_DIR = REPO_ROOT / "Visualization" / "persuasion_strategy_wvae" / "code"
WVAE_MODEL_PATH = (
    REPO_ROOT / "Visualization" / "persuasion_strategy_wvae" / "output" / "cialdini_wvae_full_v1" / "model.pkl"
)

OUTPUT_CSV = OUTPUT_DIR / "multiturn_round_strategy_distribution.csv"
ROUND_COUNTS_CSV = OUTPUT_DIR / "multiturn_round_counts.csv"
ROUND_SUMMARY_CSV = OUTPUT_DIR / "multiturn_dialogue_round_summary.csv"
OUTPUT_METADATA = OUTPUT_DIR / "multiturn_round_strategy_metadata.json"
OUTPUT_PNG = OUTPUT_DIR / "multiturn_round_strategy_boxplots.png"
PAIR_BOX_OUTPUT_CSV = OUTPUT_DIR / "multiturn_round_pair_group_distribution.csv"
PAIR_BOX_OUTPUT_PNG = OUTPUT_DIR / "multiturn_round_pair_group_boxplots.png"
STACKED_OUTPUT_CSV = OUTPUT_DIR / "multiturn_round_strategy_stacked_bars.csv"
STACKED_OUTPUT_PNG = OUTPUT_DIR / "multiturn_round_strategy_stacked_bars.png"
PAIR_STACKED_OUTPUT_CSV = OUTPUT_DIR / "multiturn_round_pair_stacked_bars.csv"
PAIR_STACKED_OUTPUT_PNG = OUTPUT_DIR / "multiturn_round_pair_stacked_bars.png"
X_AXIS_LABEL_FONT_SIZE = 33
TICK_LABEL_FONT_SIZE = 29
BOXPLOT_PRINCIPLE_ORDER = [
    "Authority",
    "Liking",
    "Reciprocity",
    "Social Proof",
    "Commitment",
    "Scarcity",
]

TURN_INPUTS = {
    "HW": HW_INPUT_CSV,
    "LLM": LLM_INPUT_CSV,
}
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
PAIR_COLORS = [
    "#143d73",
    "#2e5e95",
    "#4b7fb2",
    "#6a9cc3",
    "#88b6cf",
    "#a9cbd9",
    "#2f7f7f",
    "#5d9d8c",
    "#83b57b",
    "#a8c66b",
    "#c9cf77",
    "#d8bd74",
    "#e2aa76",
    "#e79a74",
    "#d97d62",
    "#be6851",
    "#b85a69",
    "#9b6f8f",
    "#7f73a8",
    "#676da8",
    "#4d78b7",
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
SPEAKER_PATTERN = re.compile(r"^(S|P1):\s*(.*)$")


def parse_utterances(body_text: str) -> list[dict[str, str]]:
    utterances: list[dict[str, str]] = []
    current_speaker: str | None = None
    current_lines: list[str] = []

    for raw_line in str(body_text).splitlines():
        match = SPEAKER_PATTERN.match(raw_line)
        if match:
            if current_speaker is not None:
                text = "\n".join(current_lines).strip()
                if text:
                    utterances.append({"speaker": current_speaker, "text": text})
            current_speaker = match.group(1)
            first_line = match.group(2)
            current_lines = [first_line] if first_line else []
            continue

        if current_speaker is not None:
            current_lines.append(raw_line)

    if current_speaker is not None:
        text = "\n".join(current_lines).strip()
        if text:
            utterances.append({"speaker": current_speaker, "text": text})

    return utterances


def build_turn_input(dataset_name: str, source_csv: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_csv(source_csv, low_memory=False)
    rows: list[dict[str, object]] = []
    total_turn_counts: list[int] = []
    malicious_turn_counts: list[int] = []

    passthrough_columns = [
        "label",
        "category",
        "data source",
        "thread_title",
        "thread_location",
        "thread_scammer",
        "thread_baiter",
        "message_count",
        "source_json",
    ]

    for row_index, row in frame.iterrows():
        utterances = parse_utterances(row.get("Body", ""))
        if not utterances:
            continue

        total_turn_count = len(utterances)
        malicious_utterances = [entry["text"] for entry in utterances if entry["speaker"] == "S"]
        malicious_turn_count = len(malicious_utterances)
        if malicious_turn_count == 0:
            continue

        total_turn_counts.append(total_turn_count)
        malicious_turn_counts.append(malicious_turn_count)

        for round_number, malicious_text in enumerate(malicious_utterances, start=1):
            extracted_row: dict[str, object] = {
                "dataset": dataset_name,
                "dialogue_id": f"{dataset_name}-{row_index:05d}",
                "source_row_index": row_index,
                "round_number": round_number,
                "round_label": f"R{round_number}",
                "total_turns": total_turn_count,
                "total_malicious_turns": malicious_turn_count,
                "Body": malicious_text,
            }
            for column_name in passthrough_columns:
                if column_name in frame.columns:
                    extracted_row[column_name] = row.get(column_name, "")
            rows.append(extracted_row)

    extracted = pd.DataFrame(rows)
    summary = {
        "dataset": dataset_name,
        "dialogue_count": int(len(total_turn_counts)),
        "malicious_turn_rows": int(len(extracted)),
        "total_turn_min": int(min(total_turn_counts)),
        "total_turn_median": float(pd.Series(total_turn_counts).median()),
        "total_turn_p90": float(pd.Series(total_turn_counts).quantile(0.9)),
        "total_turn_max": int(max(total_turn_counts)),
        "malicious_turn_min": int(min(malicious_turn_counts)),
        "malicious_turn_median": float(pd.Series(malicious_turn_counts).median()),
        "malicious_turn_p90": float(pd.Series(malicious_turn_counts).quantile(0.9)),
        "malicious_turn_max": int(max(malicious_turn_counts)),
    }
    return extracted, summary


def ensure_scored_turns(input_csv: Path, output_csv: Path, device: str) -> None:
    metadata_json = output_csv.with_suffix(".metadata.json")
    if output_csv.exists() and metadata_json.exists():
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(WVAE_CODE_DIR / "score_email_csv.py"),
        "--input-csv",
        str(input_csv),
        "--output-csv",
        str(output_csv),
        "--model-path",
        str(WVAE_MODEL_PATH),
        "--body-column",
        "Body",
        "--device",
        device,
        "--metadata-json",
        str(metadata_json),
    ]
    subprocess.run(command, cwd=WVAE_CODE_DIR, check=True)


def build_distribution_table(scored_frames: list[pd.DataFrame], max_round: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scored in scored_frames:
        dataset_name = str(scored["dataset"].iloc[0])
        for _, row in scored.iterrows():
            round_number = int(pd.to_numeric(row["round_number"], errors="coerce"))
            if round_number > max_round:
                continue
            for principle_name, column_name, _ in PRINCIPLE_SPECS:
                value = pd.to_numeric(row[column_name], errors="coerce")
                if pd.isna(value):
                    continue
                rows.append(
                    {
                        "dataset": dataset_name,
                        "round_number": round_number,
                        "principle": principle_name,
                        "value": float(value),
                        "dialogue_id": row["dialogue_id"],
                    }
                )
    return pd.DataFrame(rows)


def build_round_counts(scored_frames: list[pd.DataFrame], max_round: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for scored in scored_frames:
        dataset_name = str(scored["dataset"].iloc[0])
        round_counts = (
            scored[scored["round_number"].astype(int) <= max_round]
            .groupby("round_number")
            .size()
            .rename("turn_count")
            .reset_index()
        )
        round_counts["dataset"] = dataset_name
        frames.append(round_counts[["dataset", "round_number", "turn_count"]])
    return pd.concat(frames, ignore_index=True)


def draw_boxplot(distribution_table: pd.DataFrame, max_round: int) -> None:
    title_font_size = 39
    tick_font_size = TICK_LABEL_FONT_SIZE + 5
    x_label_font_size = X_AXIS_LABEL_FONT_SIZE + 6
    y_label_font_size = 41
    legend_font_size = 33
    figure, axes = plt.subplots(2, 3, figsize=(34, 19.5), sharey=True)
    figure.patch.set_facecolor("#ffffff")
    axes = axes.flatten()

    group_centers = np.arange(1, max_round + 1, dtype=float)
    box_width = 0.32
    offset = 0.19
    positions = {
        "HW": group_centers - offset,
        "LLM": group_centers + offset,
    }

    principle_specs_by_name = {name: (name, column_name, color) for name, column_name, color in PRINCIPLE_SPECS}
    ordered_specs = [principle_specs_by_name[name] for name in BOXPLOT_PRINCIPLE_ORDER]

    for axis, (principle_name, _, _) in zip(axes, ordered_specs, strict=True):
        principle_table = distribution_table[distribution_table["principle"] == principle_name].copy()
        axis.set_facecolor("#ffffff")

        for dataset_name in ["HW", "LLM"]:
            dataset_table = principle_table[principle_table["dataset"] == dataset_name].copy()
            box_data: list[np.ndarray] = []
            for round_number in range(1, max_round + 1):
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

        axis.set_title(principle_name, fontsize=title_font_size, pad=12, color="#111111")
        axis.grid(axis="y", color="#dddddd", linewidth=0.85, alpha=0.9)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#555555")
        axis.spines["bottom"].set_color("#555555")
        axis.tick_params(axis="x", labelsize=tick_font_size, colors="#222222")
        axis.tick_params(axis="y", labelsize=tick_font_size, colors="#222222")
        axis.set_xticks(group_centers)
        axis.set_xticklabels(
            [str(round_number) for round_number in range(1, max_round + 1)],
            fontsize=tick_font_size,
        )
        axis.set_ylim(0.0, 1.02)

    for axis in axes[3:]:
        axis.set_xlabel("Malicious turn round", fontsize=x_label_font_size, color="#111111", labelpad=12)
    for axis in [axes[0], axes[3]]:
        axis.set_ylabel("WVAE strategy score", fontsize=y_label_font_size, color="#111111")

    scarcity_axis = axes[5]
    scarcity_axis.legend(
        handles=[
            Patch(facecolor=TURN_COLORS["HW"], edgecolor=TURN_COLORS["HW"], label="HW multi-turn"),
            Patch(facecolor=TURN_COLORS["LLM"], edgecolor=TURN_COLORS["LLM"], label="LLM multi-turn"),
        ],
        loc="upper right",
        frameon=True,
        fontsize=legend_font_size,
        facecolor="#ffffff",
        edgecolor="#d0d0d0",
        borderaxespad=0.35,
    )
    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.16, top=0.93, wspace=0.14, hspace=0.22)
    figure.savefig(OUTPUT_PNG, dpi=260, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def build_pair_box_distribution_table(scored_frames: list[pd.DataFrame], max_round: int) -> pd.DataFrame:
    principle_lookup = {name: column_name for name, column_name, _ in PRINCIPLE_SPECS}
    rows: list[dict[str, object]] = []
    for scored in scored_frames:
        dataset_name = str(scored["dataset"].iloc[0])
        limited = scored[scored["round_number"].astype(int) <= max_round].copy()
        for _, row in limited.iterrows():
            round_number = int(pd.to_numeric(row["round_number"], errors="coerce"))
            for left_name, right_name, short_label in SELECTED_PAIR_GROUPS:
                left_value = pd.to_numeric(row[principle_lookup[left_name]], errors="coerce")
                right_value = pd.to_numeric(row[principle_lookup[right_name]], errors="coerce")
                if pd.isna(left_value) or pd.isna(right_value):
                    continue
                pair_value = float(left_value if left_name == right_name else left_value * right_value)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "round_number": round_number,
                        "pair_group": short_label,
                        "left_principle": left_name,
                        "right_principle": right_name,
                        "value": pair_value,
                        "dialogue_id": row["dialogue_id"],
                    }
                )
    return pd.DataFrame(rows)


def draw_pair_group_boxplot(distribution_table: pd.DataFrame, max_round: int) -> None:
    font_scale = 1.5
    figure, axes = plt.subplots(2, 5, figsize=(52, 18.5), sharey=True)
    figure.patch.set_facecolor("#ffffff")
    axes = axes.flatten()

    group_centers = np.arange(1, max_round + 1, dtype=float)
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
            for round_number in range(1, max_round + 1):
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
            [str(round_number) for round_number in range(1, max_round + 1)],
            fontsize=TICK_LABEL_FONT_SIZE,
        )
        axis.set_ylim(0.0, 1.02)

    for axis in axes[5:]:
        axis.set_xlabel("Malicious turn round", fontsize=X_AXIS_LABEL_FONT_SIZE, color="#111111", labelpad=12)
    for axis in [axes[0], axes[5]]:
        axis.set_ylabel("Pair co-occurrence value", fontsize=23 * font_scale, color="#111111")

    figure.subplots_adjust(left=0.07, right=0.995, bottom=0.17, top=0.92, wspace=0.24, hspace=0.28)
    figure.savefig(PAIR_BOX_OUTPUT_PNG, dpi=260, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def build_stacked_summary(distribution_table: pd.DataFrame, max_round: int) -> pd.DataFrame:
    summary = (
        distribution_table[distribution_table["round_number"] <= max_round]
        .groupby(["dataset", "round_number", "principle"], as_index=False)["value"]
        .mean()
    )
    pivot = (
        summary.pivot(index=["dataset", "round_number"], columns="principle", values="value")
        .reset_index()
        .fillna(0.0)
    )
    pivot.columns.name = None
    pivot["stack_total"] = 0.0
    for principle_name, _, _ in PRINCIPLE_SPECS:
        if principle_name not in pivot.columns:
            pivot[principle_name] = 0.0
        pivot["stack_total"] += pivot[principle_name]
    return pivot


def build_pair_specs() -> list[tuple[str, str, str, str, str]]:
    pair_specs: list[tuple[str, str, str, str, str]] = []
    color_index = 0
    for left_index, (left_name, left_column, _) in enumerate(PRINCIPLE_SPECS):
        left_short = "".join(part[0] for part in left_name.split())
        for right_name, right_column, _ in PRINCIPLE_SPECS[left_index:]:
            right_short = "".join(part[0] for part in right_name.split())
            pair_specs.append(
                (
                    f"{left_name} | {right_name}",
                    f"{left_short}-{right_short}",
                    left_column,
                    right_column,
                    PAIR_COLORS[color_index],
                )
            )
            color_index += 1
    return pair_specs


def draw_stacked_bars(summary: pd.DataFrame, max_round: int) -> None:
    group_centers = np.arange(max_round, dtype=float) * 2.0 + 0.5
    bar_positions = {
        "HW": np.arange(0, max_round * 2, 2, dtype=float),
        "LLM": np.arange(1, max_round * 2, 2, dtype=float),
    }
    bar_width = 0.82

    figure, axis = plt.subplots(figsize=(23, 13))
    figure.patch.set_facecolor("#ffffff")
    axis.set_facecolor("#ffffff")

    max_total = float(summary["stack_total"].max()) if not summary.empty else 0.0
    y_limit = max(3.2, max_total + 0.25)

    for xpos in bar_positions["HW"]:
        axis.add_patch(
            Rectangle(
                (xpos - (bar_width + 0.12) / 2, 0.0),
                bar_width + 0.12,
                y_limit,
                facecolor="#cfcfcf",
                edgecolor="none",
                alpha=0.95,
                zorder=1.2,
            )
        )

    for dataset_name in ["HW", "LLM"]:
        source_frame = (
            summary[summary["dataset"] == dataset_name]
            .set_index("round_number")
            .reindex(range(1, max_round + 1))
            .reset_index()
            .fillna(0.0)
        )
        x_positions = bar_positions[dataset_name]
        bottoms = np.zeros(max_round, dtype=float)

        for principle_name, _, color in PRINCIPLE_SPECS:
            heights = source_frame[principle_name].to_numpy(dtype=float)
            axis.bar(
                x_positions,
                heights,
                width=bar_width,
                bottom=bottoms,
                color=color,
                edgecolor="#b4b4b4" if dataset_name == "HW" else "#fffdf8",
                linewidth=0.25 if dataset_name == "HW" else 0.35,
                linestyle="solid",
                label=principle_name if dataset_name == "HW" else None,
                zorder=3,
            )
            bottoms += heights

    for divider_index in range(max_round - 1):
        midpoint = (group_centers[divider_index] + group_centers[divider_index + 1]) / 2
        axis.axvline(midpoint, color="#ddd5c9", linewidth=0.8, zorder=1)

    axis.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.9, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#888888")
    axis.spines["bottom"].set_color("#888888")
    axis.tick_params(axis="x", length=0, pad=12, labelsize=TICK_LABEL_FONT_SIZE, colors="#444444")
    axis.tick_params(axis="y", colors="#444444", labelsize=TICK_LABEL_FONT_SIZE)

    axis.set_xticks(group_centers)
    axis.set_xticklabels(
        [str(round_number) for round_number in range(1, max_round + 1)],
        fontsize=TICK_LABEL_FONT_SIZE,
    )
    axis.set_xlim(-0.55, float(bar_positions["LLM"][-1]) + 0.55)
    axis.set_ylim(0.0, y_limit)
    axis.set_ylabel("Mean WVAE strategy strength", fontsize=36, color="#333333")
    axis.set_xlabel("Malicious turn round", fontsize=X_AXIS_LABEL_FONT_SIZE, color="#333333", labelpad=18)

    color_legend = axis.legend(
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0.012, 0.992),
        frameon=True,
        fontsize=22,
        handlelength=2.2,
        columnspacing=1.2,
        borderaxespad=0.35,
        facecolor="#ffffff",
        edgecolor="#cfcfcf",
    )
    axis.add_artist(color_legend)

    style_handles = [
        Patch(facecolor="#cfcfcf", edgecolor="#9a9a9a", label="HW multi-turn"),
        Patch(facecolor="#ffffff", edgecolor="#7f7f7f", linewidth=1.6, label="LLM multi-turn"),
    ]
    style_legend = axis.legend(
        handles=style_handles,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(0.58, 0.955),
        frameon=True,
        fontsize=24,
        borderaxespad=0.35,
        facecolor="#ffffff",
        edgecolor="#cfcfcf",
        handlelength=2.5,
    )
    axis.add_artist(style_legend)

    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.98)
    figure.savefig(STACKED_OUTPUT_PNG, dpi=260, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def build_pair_stacked_summary(scored_frames: list[pd.DataFrame], max_round: int) -> pd.DataFrame:
    pair_specs = build_pair_specs()
    rows: list[dict[str, object]] = []
    for scored in scored_frames:
        dataset_name = str(scored["dataset"].iloc[0])
        limited = scored[scored["round_number"].astype(int) <= max_round].copy()
        for round_number in range(1, max_round + 1):
            round_frame = limited[limited["round_number"].astype(int) == round_number].copy()
            if round_frame.empty:
                continue
            row: dict[str, object] = {
                "dataset": dataset_name,
                "round_number": round_number,
                "turn_count": int(len(round_frame)),
                "stack_total": 0.0,
            }
            for full_label, short_label, left_column, right_column, _ in pair_specs:
                left_values = pd.to_numeric(round_frame[left_column], errors="coerce").fillna(0.0)
                right_values = pd.to_numeric(round_frame[right_column], errors="coerce").fillna(0.0)
                if left_column == right_column:
                    value = float(left_values.mean())
                else:
                    value = float((left_values * right_values).mean())
                row[full_label] = value
                row[short_label] = value
                row["stack_total"] += value
            rows.append(row)
    return pd.DataFrame(rows)


def draw_pair_stacked_bars(summary: pd.DataFrame, max_round: int) -> None:
    pair_specs = build_pair_specs()
    group_centers = np.arange(max_round, dtype=float) * 2.0 + 0.5
    bar_positions = {
        "HW": np.arange(0, max_round * 2, 2, dtype=float),
        "LLM": np.arange(1, max_round * 2, 2, dtype=float),
    }
    bar_width = 0.82

    figure, axis = plt.subplots(figsize=(26, 15))
    figure.patch.set_facecolor("#ffffff")
    axis.set_facecolor("#ffffff")

    max_total = float(summary["stack_total"].max()) if not summary.empty else 0.0
    y_limit = max(5.0, max_total + 0.45)

    for xpos in bar_positions["HW"]:
        axis.add_patch(
            Rectangle(
                (xpos - (bar_width + 0.12) / 2, 0.0),
                bar_width + 0.12,
                y_limit,
                facecolor="#cfcfcf",
                edgecolor="none",
                alpha=0.95,
                zorder=1.2,
            )
        )

    for dataset_name in ["HW", "LLM"]:
        source_frame = (
            summary[summary["dataset"] == dataset_name]
            .set_index("round_number")
            .reindex(range(1, max_round + 1))
            .reset_index()
            .fillna(0.0)
        )
        x_positions = bar_positions[dataset_name]
        bottoms = np.zeros(max_round, dtype=float)

        for full_label, short_label, _, _, color in pair_specs:
            heights = source_frame[full_label].to_numpy(dtype=float)
            axis.bar(
                x_positions,
                heights,
                width=bar_width,
                bottom=bottoms,
                color=color,
                edgecolor="#b4b4b4" if dataset_name == "HW" else "#fffdf8",
                linewidth=0.25 if dataset_name == "HW" else 0.35,
                linestyle="solid",
                label=short_label if dataset_name == "HW" else None,
                zorder=3,
            )
            bottoms += heights

    for divider_index in range(max_round - 1):
        midpoint = (group_centers[divider_index] + group_centers[divider_index + 1]) / 2
        axis.axvline(midpoint, color="#ddd5c9", linewidth=0.8, zorder=1)

    axis.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.9, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#888888")
    axis.spines["bottom"].set_color("#888888")
    axis.tick_params(axis="x", length=0, pad=12, labelsize=TICK_LABEL_FONT_SIZE, colors="#444444")
    axis.tick_params(axis="y", colors="#444444", labelsize=TICK_LABEL_FONT_SIZE)

    axis.set_xticks(group_centers)
    axis.set_xticklabels(
        [str(round_number) for round_number in range(1, max_round + 1)],
        fontsize=TICK_LABEL_FONT_SIZE,
    )
    axis.set_xlim(-0.55, float(bar_positions["LLM"][-1]) + 0.55)
    axis.set_ylim(0.0, y_limit)
    axis.set_ylabel("Mean pair co-occurrence strength", fontsize=36, color="#333333")
    axis.set_xlabel("Malicious turn round", fontsize=X_AXIS_LABEL_FONT_SIZE, color="#333333", labelpad=18)

    color_legend = axis.legend(
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0.012, 0.995),
        frameon=True,
        fontsize=27,
        handlelength=2.5,
        columnspacing=1.35,
        borderaxespad=0.35,
        facecolor="#ffffff",
        edgecolor="#cfcfcf",
    )
    axis.add_artist(color_legend)

    style_handles = [
        Patch(facecolor="#cfcfcf", edgecolor="#9a9a9a", label="HW multi-turn"),
        Patch(facecolor="#ffffff", edgecolor="#7f7f7f", linewidth=1.6, label="LLM multi-turn"),
    ]
    style_legend = axis.legend(
        handles=style_handles,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(0.46, 0.955),
        frameon=True,
        fontsize=28,
        borderaxespad=0.35,
        facecolor="#ffffff",
        edgecolor="#cfcfcf",
        handlelength=2.8,
    )
    axis.add_artist(style_legend)

    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.98)
    figure.savefig(PAIR_STACKED_OUTPUT_PNG, dpi=260, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extracted_dir = OUTPUT_DIR / "turn_inputs"
    scored_dir = OUTPUT_DIR / "scored_inputs"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    scored_dir.mkdir(parents=True, exist_ok=True)

    dialogue_summaries: list[dict[str, object]] = []
    scored_frames: list[pd.DataFrame] = []

    for dataset_name, input_csv in TURN_INPUTS.items():
        extracted, summary = build_turn_input(dataset_name, input_csv)
        dialogue_summaries.append(summary)

        extracted_csv = extracted_dir / f"{dataset_name.lower()}_malicious_turns.csv"
        extracted.to_csv(extracted_csv, index=False)

        scored_csv = scored_dir / f"{dataset_name.lower()}_malicious_turns_scored.csv"
        ensure_scored_turns(extracted_csv, scored_csv, device="cuda")
        scored = pd.read_csv(scored_csv, low_memory=False)
        scored["dataset"] = dataset_name
        scored["round_number"] = pd.to_numeric(scored["round_number"], errors="coerce").astype(int)
        scored_frames.append(scored)

    distribution_table = build_distribution_table(scored_frames, max_round=PLOT_ROUND_LIMIT)
    if distribution_table.empty:
        raise RuntimeError("No round-level persuasion rows were generated for folder 8.")

    pair_box_distribution_table = build_pair_box_distribution_table(scored_frames, max_round=PLOT_ROUND_LIMIT)
    round_counts = build_round_counts(scored_frames, max_round=PLOT_ROUND_LIMIT)
    stacked_summary = build_stacked_summary(distribution_table, max_round=PLOT_ROUND_LIMIT)
    pair_stacked_summary = build_pair_stacked_summary(scored_frames, max_round=PLOT_ROUND_LIMIT)
    distribution_table.to_csv(OUTPUT_CSV, index=False)
    pair_box_distribution_table.to_csv(PAIR_BOX_OUTPUT_CSV, index=False)
    round_counts.to_csv(ROUND_COUNTS_CSV, index=False)
    stacked_summary.to_csv(STACKED_OUTPUT_CSV, index=False)
    pair_stacked_summary.to_csv(PAIR_STACKED_OUTPUT_CSV, index=False)
    pd.DataFrame(dialogue_summaries).to_csv(ROUND_SUMMARY_CSV, index=False)
    draw_boxplot(distribution_table, max_round=PLOT_ROUND_LIMIT)
    draw_pair_group_boxplot(pair_box_distribution_table, max_round=PLOT_ROUND_LIMIT)
    draw_stacked_bars(stacked_summary, max_round=PLOT_ROUND_LIMIT)
    draw_pair_stacked_bars(pair_stacked_summary, max_round=PLOT_ROUND_LIMIT)

    metadata = {
        "plot_round_limit": PLOT_ROUND_LIMIT,
        "source_inputs": {name: str(path) for name, path in TURN_INPUTS.items()},
        "dialogue_summaries": dialogue_summaries,
        "principles": [name for name, _, _ in PRINCIPLE_SPECS],
        "distribution_rows": int(len(distribution_table)),
        "round_counts_csv": ROUND_COUNTS_CSV.name,
        "round_summary_csv": ROUND_SUMMARY_CSV.name,
        "output_csv": OUTPUT_CSV.name,
        "output_png": OUTPUT_PNG.name,
        "pair_box_output_csv": PAIR_BOX_OUTPUT_CSV.name,
        "pair_box_output_png": PAIR_BOX_OUTPUT_PNG.name,
        "stacked_output_csv": STACKED_OUTPUT_CSV.name,
        "stacked_output_png": STACKED_OUTPUT_PNG.name,
        "pair_stacked_output_csv": PAIR_STACKED_OUTPUT_CSV.name,
        "pair_stacked_output_png": PAIR_STACKED_OUTPUT_PNG.name,
        "selected_pair_groups": [short_label for _, _, short_label in SELECTED_PAIR_GROUPS],
        "pair_labels": [short_label for _, short_label, _, _, _ in build_pair_specs()],
        "scored_dir": scored_dir.name,
    }
    OUTPUT_METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
