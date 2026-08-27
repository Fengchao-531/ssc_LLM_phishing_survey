#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


SCRIPT_DIR = Path(__file__).resolve().parent
STAGES_DIR = SCRIPT_DIR.parent / "stages"
OUTPUT_DIR = SCRIPT_DIR / "6"
OUTPUT_CSV = OUTPUT_DIR / "stagewise_hw_llm_fn_selected_group_boxplots.csv"
OUTPUT_METADATA = OUTPUT_DIR / "stagewise_hw_llm_fn_selected_group_boxplots_metadata.json"
X_AXIS_LABEL_FONT_SIZE = 45
Y_AXIS_LABEL_FONT_SIZE = 46
TICK_LABEL_FONT_SIZE = 41
STAGE_ORDER = [
    "S1",
    "S2",
    "S4",
    "S5",
    "S6-MPG",
    "S6-UTA",
    "S6-fuzzer",
    "S8-claude",
    "S8-deepseek",
    "S8-gemini",
    "S8-gpt",
    "S8-llama",
    "S8-ministral",
]
SOURCE_ORDER = ["HW", "LLM"]
PRINCIPLE_COLUMNS = {
    "Authority": "principle_authority",
    "Liking": "principle_liking",
    "Reciprocity": "principle_reciprocity",
    "Social Proof": "principle_social_proof",
    "Scarcity": "principle_scarcity",
    "Commitment": "principle_commitment",
}
SELECTED_GROUPS = [
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


def build_distribution_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage_name in STAGE_ORDER:
        stage_csv = STAGES_DIR / stage_name / "projected_points.csv"
        if not stage_csv.exists():
            continue
        frame = pd.read_csv(stage_csv, low_memory=False)
        for source_name in SOURCE_ORDER:
            subset = frame[(frame["source"] == source_name) & frame["is_fn"].astype(bool)].copy()
            if subset.empty:
                continue
            for left_name, right_name, short_label in SELECTED_GROUPS:
                left_values = pd.to_numeric(subset[PRINCIPLE_COLUMNS[left_name]], errors="coerce")
                right_values = pd.to_numeric(subset[PRINCIPLE_COLUMNS[right_name]], errors="coerce")
                group_values = left_values * right_values
                valid_values = group_values[group_values.notna()].to_numpy(dtype=float)
                for value in valid_values:
                    rows.append(
                        {
                            "stage": stage_name,
                            "source": source_name,
                            "principle_group": short_label,
                            "left_principle": left_name,
                            "right_principle": right_name,
                            "value": float(value),
                        }
                    )
    return pd.DataFrame(rows)


def draw_stage_boxplot(stage_name: str, stage_table: pd.DataFrame) -> dict[str, int]:
    output_png = OUTPUT_DIR / f"{stage_name}_hw_llm_fn_selected_group_boxplots.png"
    figure, axis = plt.subplots(figsize=(23, 12.5))
    figure.patch.set_facecolor("#ffffff")
    axis.set_facecolor("#ffffff")

    group_centers = np.arange(len(SELECTED_GROUPS), dtype=float) * 2.0 + 0.5
    box_width = 0.62
    offset = 0.38
    positions = {"HW": group_centers - offset, "LLM": group_centers + offset}
    fill_colors = {"HW": "#143d73", "LLM": "#b85a69"}
    edge_colors = {"HW": "#143d73", "LLM": "#b85a69"}
    counts: dict[str, int] = {}

    for source_name in SOURCE_ORDER:
        box_data = []
        counts[source_name] = int(len(stage_table[stage_table["source"] == source_name]["value"]))
        for _, _, short_label in SELECTED_GROUPS:
            values = stage_table[
                (stage_table["source"] == source_name)
                & (stage_table["principle_group"] == short_label)
            ]["value"].to_numpy(dtype=float)
            box_data.append(values)

        boxplot = axis.boxplot(
            box_data,
            positions=positions[source_name],
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#ffffff", "linewidth": 2.4},
            whiskerprops={"color": edge_colors[source_name], "linewidth": 1.6},
            capprops={"color": edge_colors[source_name], "linewidth": 1.6},
            boxprops={"edgecolor": edge_colors[source_name], "linewidth": 1.8},
        )
        for patch in boxplot["boxes"]:
            patch.set_facecolor(fill_colors[source_name])
            patch.set_alpha(0.92)

    axis.grid(axis="y", color="#dddddd", linewidth=0.85, alpha=0.9)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#555555")
    axis.spines["bottom"].set_color("#555555")
    axis.tick_params(axis="x", length=0, pad=12, labelsize=TICK_LABEL_FONT_SIZE, colors="#222222")
    axis.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE, colors="#222222")

    axis.set_xticks(group_centers)
    axis.set_xticklabels(
        [short_label for _, _, short_label in SELECTED_GROUPS],
        rotation=0,
        ha="center",
        fontsize=TICK_LABEL_FONT_SIZE,
        color="#222222",
    )
    axis.set_ylabel("Per-email co-occurrence value", fontsize=Y_AXIS_LABEL_FONT_SIZE, color="#111111")
    axis.set_xlabel(
        f"{stage_name} selected persuasion principle groups",
        fontsize=X_AXIS_LABEL_FONT_SIZE,
        color="#111111",
        labelpad=18,
    )
    axis.set_title("")

    legend = axis.legend(
        handles=[
            Patch(facecolor=fill_colors["HW"], edgecolor=edge_colors["HW"], label="HW-P FN"),
            Patch(facecolor=fill_colors["LLM"], edgecolor=edge_colors["LLM"], label="LLM-P FN"),
        ],
        loc="upper right",
        frameon=True,
        fontsize=34,
        facecolor="#ffffff",
        edgecolor="#d0d0d0",
    )
    axis.add_artist(legend)

    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.20, top=0.97)
    figure.savefig(output_png, dpi=260, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return {"hw_value_count": counts["HW"], "llm_value_count": counts["LLM"], "output_png": output_png.name}


def combine_stage_figures(output_name: str, stage_names: list[str]) -> str:
    image_paths = [OUTPUT_DIR / f"{stage_name}_hw_llm_fn_selected_group_boxplots.png" for stage_name in stage_names]
    images = [mpimg.imread(path) for path in image_paths]
    figure, axes = plt.subplots(1, len(images), figsize=(7.6 * len(images), 8.8))
    figure.patch.set_facecolor("#ffffff")
    if len(images) == 1:
        axes = [axes]
    for axis, stage_name, image in zip(axes, stage_names, images, strict=True):
        axis.imshow(image)
        axis.axis("off")
    figure.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, wspace=0.03)
    output_path = OUTPUT_DIR / output_name
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return output_path.name


def write_outputs(distribution_table: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    distribution_table.to_csv(OUTPUT_CSV, index=False)
    stage_entries: list[dict[str, object]] = []
    for stage_name in STAGE_ORDER:
        stage_table = distribution_table[distribution_table["stage"] == stage_name].copy()
        if stage_table.empty:
            continue
        stage_entries.append(
            {
                "stage": stage_name,
                **draw_stage_boxplot(stage_name, stage_table),
            }
        )

    combined_outputs = [
        {
            "name": "S6_combined_row_boxplots.png",
            "stages": ["S6-fuzzer", "S6-UTA", "S6-MPG"],
        },
        {
            "name": "S8_combined_row1_boxplots.png",
            "stages": ["S8-claude", "S8-deepseek", "S8-gemini"],
        },
        {
            "name": "S8_combined_row2_boxplots.png",
            "stages": ["S8-gpt", "S8-llama", "S8-ministral"],
        },
    ]
    combined_entries: list[dict[str, object]] = []
    for entry in combined_outputs:
        combined_entries.append(
            {
                "output_png": combine_stage_figures(entry["name"], entry["stages"]),
                "stages": entry["stages"],
            }
        )

    metadata = {
        "selected_groups": [
            {"short_label": short_label, "left": left_name, "right": right_name}
            for left_name, right_name, short_label in SELECTED_GROUPS
        ],
        "source_note": "Each stage-specific box uses per-email pairwise values from HW-P FN or LLM-P FN rows, not heatmap means.",
        "stages_generated": stage_entries,
        "combined_outputs": combined_entries,
        "output_csv": OUTPUT_CSV.name,
        "rows": int(len(distribution_table)),
    }
    OUTPUT_METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    distribution_table = build_distribution_table()
    if distribution_table.empty:
        raise RuntimeError("No rows were generated for folder 6 stagewise boxplot data.")
    write_outputs(distribution_table)


if __name__ == "__main__":
    main()
