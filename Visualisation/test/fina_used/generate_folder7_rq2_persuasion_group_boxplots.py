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
VISUALIZATION_DIR = SCRIPT_DIR.parent.parent
RQ2_DIR = VISUALIZATION_DIR / "RQ2" / "scored_inputs"
PHISHING_DIR = VISUALIZATION_DIR / "test" / "A-I Differences"
OUTPUT_DIR = SCRIPT_DIR / "7"
OUTPUT_CSV = OUTPUT_DIR / "rq2_persuasion_group_boxplots.csv"
OUTPUT_METADATA = OUTPUT_DIR / "rq2_persuasion_group_boxplots_metadata.json"
PANEL_SPECS = [
    {
        "key": "llm_phishing",
        "title": "LLM phishing selected features",
        "source_note": "LLM phishing rows from merged_academic_industry_phishing_only.csv.",
        "color": "#1f4e79",
    },
    {
        "key": "vishing_single",
        "title": "Vishing single-turn",
        "source_note": "Combined HW and LLM rows from single-turn vishing scored inputs.",
        "color": "#b85a69",
    },
    {
        "key": "vishing_multi",
        "title": "Vishing multi-turn",
        "source_note": "Combined HW and LLM rows from multi-turn vishing scored inputs.",
        "color": "#2f5d50",
    },
]

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


def load_phishing_llm_frame() -> pd.DataFrame:
    csv_path = PHISHING_DIR / "merged_academic_industry_phishing_only.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Phishing input not found: {csv_path}")
    frame = pd.read_csv(csv_path, low_memory=False)
    if "source" not in frame.columns:
        raise FileNotFoundError("Phishing input does not contain a source column.")
    frame = frame[frame["source"].astype(str) == "LLM"].copy()
    if frame.empty:
        raise FileNotFoundError("No LLM phishing rows were found in merged_academic_industry_phishing_only.csv")
    frame["panel"] = "llm_phishing"
    return frame.reset_index(drop=True)


def load_vishing_turn(turn_type: str) -> pd.DataFrame:
    if turn_type == "single":
        input_files = [
            RQ2_DIR / "single_hw_vishing_persuasion.csv",
            RQ2_DIR / "single_llm_vishing_persuasion.csv",
        ]
        panel_key = "vishing_single"
    elif turn_type == "multi":
        input_files = [
            RQ2_DIR / "multi_hw_vishing_persuasion.csv",
            RQ2_DIR / "multi_llm_vishing_persuasion.csv",
        ]
        panel_key = "vishing_multi"
    else:
        raise ValueError(f"Unsupported turn_type: {turn_type}")

    frames: list[pd.DataFrame] = []
    for csv_path in input_files:
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path, low_memory=False)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No vishing rows were found for {turn_type}-turn inputs.")

    combined = pd.concat(frames, ignore_index=True)
    combined["panel"] = panel_key
    return combined.reset_index(drop=True)


def build_distribution_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    panel_frames = [
        load_phishing_llm_frame(),
        load_vishing_turn("single"),
        load_vishing_turn("multi"),
    ]

    for frame in panel_frames:
        panel_name = str(frame["panel"].iloc[0])
        for _, row in frame.iterrows():
            for left_name, right_name, short_label in SELECTED_GROUPS:
                left_value = pd.to_numeric(row[PRINCIPLE_COLUMNS[left_name]], errors="coerce")
                right_value = pd.to_numeric(row[PRINCIPLE_COLUMNS[right_name]], errors="coerce")

                if pd.isna(left_value) or pd.isna(right_value):
                    continue

                rows.append(
                    {
                        "panel": panel_name,
                        "principle_group": short_label,
                        "left_principle": left_name,
                        "right_principle": right_name,
                        "value": float(left_value * right_value),
                    }
                )
    
    return pd.DataFrame(rows)


def collect_box_data(panel_table: pd.DataFrame) -> list[np.ndarray]:
    box_data: list[np.ndarray] = []
    for _, _, short_label in SELECTED_GROUPS:
        values = panel_table[panel_table["principle_group"] == short_label]["value"].to_numpy(dtype=float)
        box_data.append(values)
    return box_data


def style_axis(axis: plt.Axes, group_centers: np.ndarray) -> None:
    axis.grid(axis="y", color="#dddddd", linewidth=0.85, alpha=0.9)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#555555")
    axis.spines["bottom"].set_color("#555555")
    axis.tick_params(axis="x", length=0, pad=12, labelsize=30, colors="#222222")
    axis.tick_params(axis="y", labelsize=30, colors="#222222")
    axis.set_xticks(group_centers)
    axis.set_xticklabels(
        [short_label for _, _, short_label in SELECTED_GROUPS],
        rotation=0,
        ha="center",
        fontsize=30,
        color="#222222",
    )


def draw_boxplot(data_table: pd.DataFrame) -> dict[str, int]:
    output_png = OUTPUT_DIR / "rq2_persuasion_group_boxplots.png"
    figure, axis = plt.subplots(figsize=(24, 10.5))
    figure.patch.set_facecolor("#ffffff")
    counts: dict[str, int] = {}

    axis.set_facecolor("#ffffff")
    group_centers = np.arange(len(SELECTED_GROUPS), dtype=float) * 2.4 + 1.0
    offsets = [-0.55, 0.0, 0.55]
    legend_handles: list[Patch] = []

    for offset, panel_spec in zip(offsets, PANEL_SPECS, strict=True):
        panel_table = data_table[data_table["panel"] == panel_spec["key"]].copy()
        counts[panel_spec["key"]] = int(len(panel_table["value"]))
        if panel_table.empty:
            continue

        boxplot = axis.boxplot(
            collect_box_data(panel_table),
            positions=group_centers + offset,
            widths=0.44,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#ffffff", "linewidth": 2.2},
            whiskerprops={"color": panel_spec["color"], "linewidth": 1.5},
            capprops={"color": panel_spec["color"], "linewidth": 1.5},
            boxprops={"edgecolor": panel_spec["color"], "linewidth": 1.7},
        )
        for patch in boxplot["boxes"]:
            patch.set_facecolor(panel_spec["color"])
            patch.set_alpha(0.92)

        legend_handles.append(
            Patch(
                facecolor=panel_spec["color"],
                edgecolor=panel_spec["color"],
                label=panel_spec["title"],
                alpha=0.92,
            )
        )

    style_axis(axis, group_centers)
    axis.set_ylabel("Per-email co-occurrence value", fontsize=34, color="#111111")
    axis.set_xlabel("Selected persuasion principle groups", fontsize=34, color="#111111", labelpad=18)
    axis.set_title("")
    axis.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.015, 0.03),
        ncol=1,
        frameon=True,
        fontsize=30,
        facecolor="#ffffff",
        edgecolor="#d0d0d0",
    )

    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.20, top=0.97)
    figure.savefig(output_png, dpi=260, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    
    return {
        "llm_phishing_count": counts.get("llm_phishing", 0),
        "vishing_single_count": counts.get("vishing_single", 0),
        "vishing_multi_count": counts.get("vishing_multi", 0),
        "output_png": output_png.name
    }


def combine_figures(output_name: str, image_names: list[str]) -> str:
    """Combine multiple PNG images horizontally"""
    image_paths = [OUTPUT_DIR / img_name for img_name in image_names]
    images = [mpimg.imread(path) for path in image_paths]
    
    figure, axes = plt.subplots(1, len(images), figsize=(7.6 * len(images), 8.8))
    figure.patch.set_facecolor("#ffffff")
    
    if len(images) == 1:
        axes = [axes]
    
    for axis, image in zip(axes, images, strict=True):
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
    
    boxplot_info = draw_boxplot(distribution_table)
    
    metadata = {
        "selected_groups": [
            {"short_label": short_label, "left": left_name, "right": right_name}
            for left_name, right_name, short_label in SELECTED_GROUPS
        ],
        "panels": [
            {
                "key": panel["key"],
                "title": panel["title"],
                "source_note": panel["source_note"],
                "count": boxplot_info[f"{panel['key']}_count"],
            }
            for panel in PANEL_SPECS
        ],
        "output_csv": OUTPUT_CSV.name,
        "output_png": boxplot_info["output_png"],
        "rows": int(len(distribution_table)),
    }
    OUTPUT_METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    distribution_table = build_distribution_table()
    if distribution_table.empty:
        raise RuntimeError("No rows were generated for folder 7 RQ2 persuasion boxplot data.")
    write_outputs(distribution_table)


if __name__ == "__main__":
    main()
