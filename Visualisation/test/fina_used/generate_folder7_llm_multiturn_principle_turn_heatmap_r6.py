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
OUTPUT_CSV = OUTPUT_DIR / "llm_multiturn_principle_turn_mean_trends_r6.csv"
OUTPUT_PNG = OUTPUT_DIR / "llm_multiturn_principle_turn_heatmap_r6.png"
OUTPUT_METADATA = OUTPUT_DIR / "llm_multiturn_principle_turn_heatmap_r6_metadata.json"
HEATMAP_VALUE_FONT_SIZE = 17

PRINCIPLE_SPECS = [
    ("Authority", "principle_authority"),
    ("Liking", "principle_liking"),
    ("Reciprocity", "principle_reciprocity"),
    ("Social Proof", "principle_social_proof"),
    ("Scarcity", "principle_scarcity"),
    ("Commitment", "principle_commitment"),
]

PLOT_ROUND_LIMIT = 6
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "folder7_principle_turn_green_r6",
    ["#f5faf7", "#d6e9df", "#9fc5b2", "#5d8f79", "#2f5d50"],
    N=256,
)


def load_scored_turns() -> pd.DataFrame:
    if not INPUT_SCORED.exists():
        raise FileNotFoundError(f"Scored turn input not found: {INPUT_SCORED}")
    frame = pd.read_csv(INPUT_SCORED, low_memory=False).copy()
    frame["round_number"] = pd.to_numeric(frame["round_number"], errors="coerce")
    frame["total_malicious_turns"] = pd.to_numeric(frame["total_malicious_turns"], errors="coerce")
    frame = frame[frame["round_number"].notna()].copy()
    frame["round_number"] = frame["round_number"].astype(int)
    eligible_dialogues = (
        frame[["dialogue_id", "total_malicious_turns"]]
        .drop_duplicates()
        .query("total_malicious_turns >= @PLOT_ROUND_LIMIT")["dialogue_id"]
        .tolist()
    )
    frame = frame[frame["dialogue_id"].isin(eligible_dialogues)].copy()
    frame = frame[frame["round_number"] <= PLOT_ROUND_LIMIT].copy()
    if frame.empty:
        raise ValueError("No scammer-only LLM multi-turn rows were available after round filtering.")
    return frame.reset_index(drop=True)


def build_principle_turn_summary(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
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


def draw_heatmap(summary_table: pd.DataFrame, full_dataset_median_turns: float, cohort_dialogue_count: int) -> None:
    principle_order = [name for name, _ in PRINCIPLE_SPECS]
    pivot = (
        summary_table.pivot(index="principle", columns="round_number", values="mean_value")
        .reindex(principle_order)
        .reindex(columns=range(1, PLOT_ROUND_LIMIT + 1))
    )
    matrix = pivot.to_numpy(dtype=float)
    vmax = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0

    figure, axis = plt.subplots(figsize=(11.8, 5.8))
    figure.patch.set_facecolor("#ffffff")
    image = axis.imshow(matrix, cmap=HEATMAP_CMAP, vmin=0.0, vmax=vmax, aspect="auto")

    axis.set_xticks(range(PLOT_ROUND_LIMIT))
    axis.set_xticklabels([str(round_number) for round_number in range(1, PLOT_ROUND_LIMIT + 1)], fontsize=18)
    axis.set_yticks(range(len(principle_order)))
    axis.set_yticklabels(principle_order, fontsize=18)
    axis.set_xlabel("Scammer turn round", fontsize=20, color="#111111", labelpad=10)
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

    figure.subplots_adjust(left=0.20, right=0.92, bottom=0.18, top=0.95)
    figure.savefig(OUTPUT_PNG, dpi=260, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def write_metadata(
    scored: pd.DataFrame,
    summary_table: pd.DataFrame,
    full_dataset_median_turns: float,
    cohort_median_turns: float,
) -> None:
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
        "plot_round_limit": PLOT_ROUND_LIMIT,
        "source_dataset": "LLM multi-turn phishing",
        "view": "scammer_only",
        "cohort_rule": "only dialogues with at least 6 malicious scammer turns",
        "full_dataset_median_malicious_turns": float(full_dataset_median_turns),
        "cohort_median_malicious_turns": float(cohort_median_turns),
        "dialogue_count_total": int(scored["dialogue_id"].nunique()),
        "turn_row_count_plotted": int(len(scored)),
        "summary_row_count": int(len(summary_table)),
        "round_counts": {str(key): int(value) for key, value in round_counts.items()},
    }
    OUTPUT_METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scored = load_scored_turns()
    summary_table = build_principle_turn_summary(scored)
    summary_table.to_csv(OUTPUT_CSV, index=False)
    full_frame = pd.read_csv(INPUT_SCORED, usecols=["dialogue_id", "total_malicious_turns"])
    full_frame["total_malicious_turns"] = pd.to_numeric(full_frame["total_malicious_turns"], errors="coerce")
    full_dataset_median_turns = float(full_frame.drop_duplicates()["total_malicious_turns"].median())
    cohort_median_turns = float(scored[["dialogue_id", "total_malicious_turns"]].drop_duplicates()["total_malicious_turns"].median())
    draw_heatmap(summary_table, full_dataset_median_turns, int(scored["dialogue_id"].nunique()))
    write_metadata(scored, summary_table, full_dataset_median_turns, cohort_median_turns)
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
